from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from kubernetes import client, config
from openai import OpenAI
from typing import Optional
import uvicorn
import json
import os
import time
import threading
from collections import deque
from pathlib import Path

app = FastAPI(title="Kubernetes API", version="1.0.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


def _load_dotenv(path: Path) -> None:
    """Load KEY=value lines from a .env file into os.environ if not already set."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv(Path(BASE_DIR) / ".env")

_openai_client: Optional[OpenAI] = None

def _get_openai_client() -> OpenAI:
    """Lazily create the OpenAI client so the app can start without a key (REST/k8s still work)."""
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not set. Export it or add OPENAI_API_KEY=... to .env in the project root.",
        )
    _openai_client = OpenAI(api_key=key)
    return _openai_client

# ---------------------------------------------------------------------------
# Client-side rate limiter — stays within OpenAI's 3 req/min free-tier limit
# Queues calls transparently so users never see a 429 error.
# ---------------------------------------------------------------------------
_rpm_limit   = 3          # max requests per rolling 60-second window
_window_secs = 60
_req_times: deque = deque()
_throttle_lock = threading.Lock()

def _throttled_create(**kwargs):
    """Rate-limit OpenAI calls to _rpm_limit per minute, blocking if needed."""
    with _throttle_lock:
        now = time.monotonic()
        # Drop timestamps outside the rolling window
        while _req_times and now - _req_times[0] >= _window_secs:
            _req_times.popleft()

        if len(_req_times) >= _rpm_limit:
            wait = _window_secs - (now - _req_times[0])
            if wait > 0:
                time.sleep(wait)
            # Re-prune after sleeping
            now = time.monotonic()
            while _req_times and now - _req_times[0] >= _window_secs:
                _req_times.popleft()

        _req_times.append(time.monotonic())

    return _get_openai_client().responses.create(**kwargs)

# ---------------------------------------------------------------------------
# Load kube config (uses ~/.kube/config locally, in-cluster config in a pod)
# ---------------------------------------------------------------------------
try:
    config.load_incluster_config()
except config.ConfigException:
    config.load_kube_config()

v1 = client.CoreV1Api()
networking_v1 = client.NetworkingV1Api()
apps_v1 = client.AppsV1Api()
batch_v1 = client.BatchV1Api()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class PodCreate(BaseModel):
    name: str
    namespace: str = "default"
    image: str
    labels: Optional[dict] = {"app": "fastapi-created"}


class DeploymentCreate(BaseModel):
    name: str
    image: str
    replicas: int
    cpu: str
    memory: str
    namespace: str = "default"
    labels: Optional[dict] = None


class ServiceCreate(BaseModel):
    name: str
    port: int
    target_port: int
    type: str = "ClusterIP"
    protocol: str = "TCP"
    selector: Optional[dict] = None
    namespace: str = "default"


class DaemonSetCreate(BaseModel):
    name: str
    image: str
    namespace: str = "default"
    labels: Optional[dict] = None


class CronJobCreate(BaseModel):
    name: str
    image: str
    schedule: str
    namespace: str = "default"
    command: Optional[list] = None


# ---------------------------------------------------------------------------
# Pods
# ---------------------------------------------------------------------------
@app.get("/pods", summary="List all pods across all namespaces")
def list_pods(namespace: Optional[str] = None):
    try:
        if namespace:
            pods = v1.list_namespaced_pod(namespace=namespace)
        else:
            pods = v1.list_pod_for_all_namespaces()

        return {
            "pods": [
                {
                    "name": pod.metadata.name,
                    "namespace": pod.metadata.namespace,
                    "status": pod.status.phase,
                    "node": pod.spec.node_name,
                    "ip": pod.status.pod_ip,
                    "containers": [c.name for c in pod.spec.containers],
                }
                for pod in pods.items
            ]
        }
    except client.exceptions.ApiException as e:
        raise HTTPException(status_code=e.status, detail=e.reason)


@app.post("/pods", summary="Create a pod", status_code=201)
def create_pod(body: PodCreate):
    pod_manifest = client.V1Pod(
        metadata=client.V1ObjectMeta(
            name=body.name,
            namespace=body.namespace,
            labels=body.labels,
        ),
        spec=client.V1PodSpec(
            containers=[
                client.V1Container(
                    name=body.name,
                    image=body.image,
                )
            ],
            restart_policy="Always",
        ),
    )
    try:
        pod = v1.create_namespaced_pod(namespace=body.namespace, body=pod_manifest)
        return {
            "message": "Pod created successfully",
            "name": pod.metadata.name,
            "namespace": pod.metadata.namespace,
            "status": pod.status.phase,
        }
    except client.exceptions.ApiException as e:
        raise HTTPException(status_code=e.status, detail=e.reason)


# ---------------------------------------------------------------------------
# Deployments
# ---------------------------------------------------------------------------
@app.post("/deployments", summary="Create a deployment", status_code=201)
def create_deployment(body: DeploymentCreate):
    labels = body.labels or {"app": body.name}
    resources = client.V1ResourceRequirements(
        requests={"cpu": body.cpu, "memory": body.memory},
        limits={"cpu": body.cpu, "memory": body.memory},
    )

    deployment_manifest = client.V1Deployment(
        metadata=client.V1ObjectMeta(name=body.name, namespace=body.namespace, labels=labels),
        spec=client.V1DeploymentSpec(
            replicas=body.replicas,
            selector=client.V1LabelSelector(match_labels=labels),
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(labels=labels),
                spec=client.V1PodSpec(
                    containers=[
                        client.V1Container(
                            name=body.name,
                            image=body.image,
                            resources=resources,
                        )
                    ],
                ),
            ),
        ),
    )
    try:
        d = apps_v1.create_namespaced_deployment(namespace=body.namespace, body=deployment_manifest)
        return {
            "message": "Deployment created successfully",
            "name": d.metadata.name,
            "namespace": d.metadata.namespace,
            "replicas": d.spec.replicas,
            "image": body.image,
            "cpu": body.cpu,
            "memory": body.memory,
        }
    except client.exceptions.ApiException as e:
        raise HTTPException(status_code=e.status, detail=e.reason)


# ---------------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------------
@app.get("/services", summary="List all services across all namespaces")
def list_services(namespace: Optional[str] = None):
    try:
        if namespace:
            services = v1.list_namespaced_service(namespace=namespace)
        else:
            services = v1.list_service_for_all_namespaces()

        return {
            "services": [
                {
                    "name": svc.metadata.name,
                    "namespace": svc.metadata.namespace,
                    "type": svc.spec.type,
                    "cluster_ip": svc.spec.cluster_ip,
                    "ports": [
                        {"port": p.port, "protocol": p.protocol, "target_port": str(p.target_port)}
                        for p in (svc.spec.ports or [])
                    ],
                }
                for svc in services.items
            ]
        }
    except client.exceptions.ApiException as e:
        raise HTTPException(status_code=e.status, detail=e.reason)


@app.post("/services", summary="Create a service", status_code=201)
def create_service(body: ServiceCreate):
    selector = body.selector or {"app": body.name}
    service_manifest = client.V1Service(
        metadata=client.V1ObjectMeta(name=body.name, namespace=body.namespace),
        spec=client.V1ServiceSpec(
            type=body.type,
            selector=selector,
            ports=[
                client.V1ServicePort(
                    port=body.port,
                    target_port=body.target_port,
                    protocol=body.protocol,
                )
            ],
        ),
    )
    try:
        svc = v1.create_namespaced_service(namespace=body.namespace, body=service_manifest)
        return {
            "message": "Service created successfully",
            "name": svc.metadata.name,
            "namespace": svc.metadata.namespace,
            "type": svc.spec.type,
            "cluster_ip": svc.spec.cluster_ip,
            "selector": svc.spec.selector,
            "ports": [
                {"port": p.port, "protocol": p.protocol, "target_port": str(p.target_port)}
                for p in (svc.spec.ports or [])
            ],
        }
    except client.exceptions.ApiException as e:
        raise HTTPException(status_code=e.status, detail=e.reason)


# ---------------------------------------------------------------------------
# DaemonSets
# ---------------------------------------------------------------------------
@app.get("/daemonsets", summary="List all daemonsets across all namespaces")
def list_daemonsets(namespace: Optional[str] = None):
    try:
        if namespace:
            daemonsets = apps_v1.list_namespaced_daemon_set(namespace=namespace)
        else:
            daemonsets = apps_v1.list_daemon_set_for_all_namespaces()

        return {
            "daemonsets": [
                {
                    "name": ds.metadata.name,
                    "namespace": ds.metadata.namespace,
                    "desired": ds.status.desired_number_scheduled,
                    "current": ds.status.current_number_scheduled,
                    "ready": ds.status.number_ready,
                    "containers": [c.name for c in ds.spec.template.spec.containers],
                }
                for ds in daemonsets.items
            ]
        }
    except client.exceptions.ApiException as e:
        raise HTTPException(status_code=e.status, detail=e.reason)


@app.post("/daemonsets", summary="Create a daemonset", status_code=201)
def create_daemonset(body: DaemonSetCreate):
    labels = body.labels or {"app": body.name}
    ds_manifest = client.V1DaemonSet(
        metadata=client.V1ObjectMeta(name=body.name, namespace=body.namespace, labels=labels),
        spec=client.V1DaemonSetSpec(
            selector=client.V1LabelSelector(match_labels=labels),
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(labels=labels),
                spec=client.V1PodSpec(
                    containers=[client.V1Container(name=body.name, image=body.image)],
                ),
            ),
        ),
    )
    try:
        ds = apps_v1.create_namespaced_daemon_set(namespace=body.namespace, body=ds_manifest)
        return {
            "message": "DaemonSet created successfully",
            "name": ds.metadata.name,
            "namespace": ds.metadata.namespace,
            "image": body.image,
        }
    except client.exceptions.ApiException as e:
        raise HTTPException(status_code=e.status, detail=e.reason)


# ---------------------------------------------------------------------------
# CronJobs
# ---------------------------------------------------------------------------
@app.get("/cronjobs", summary="List all cronjobs across all namespaces")
def list_cronjobs(namespace: Optional[str] = None):
    try:
        if namespace:
            cronjobs = batch_v1.list_namespaced_cron_job(namespace=namespace)
        else:
            cronjobs = batch_v1.list_cron_job_for_all_namespaces()

        return {
            "cronjobs": [
                {
                    "name": cj.metadata.name,
                    "namespace": cj.metadata.namespace,
                    "schedule": cj.spec.schedule,
                    "suspend": cj.spec.suspend,
                    "last_schedule_time": str(cj.status.last_schedule_time)
                    if cj.status and cj.status.last_schedule_time
                    else None,
                    "active": len(cj.status.active or []) if cj.status else 0,
                }
                for cj in cronjobs.items
            ]
        }
    except client.exceptions.ApiException as e:
        raise HTTPException(status_code=e.status, detail=e.reason)


@app.post("/cronjobs", summary="Create a cronjob", status_code=201)
def create_cronjob(body: CronJobCreate):
    container = client.V1Container(
        name=body.name,
        image=body.image,
        command=body.command,
    )
    job_template = client.V1JobTemplateSpec(
        spec=client.V1JobSpec(
            template=client.V1PodTemplateSpec(
                spec=client.V1PodSpec(
                    containers=[container],
                    restart_policy="OnFailure",
                ),
            ),
        ),
    )
    cj_manifest = client.V1CronJob(
        metadata=client.V1ObjectMeta(name=body.name, namespace=body.namespace),
        spec=client.V1CronJobSpec(
            schedule=body.schedule,
            job_template=job_template,
        ),
    )
    try:
        cj = batch_v1.create_namespaced_cron_job(namespace=body.namespace, body=cj_manifest)
        return {
            "message": "CronJob created successfully",
            "name": cj.metadata.name,
            "namespace": cj.metadata.namespace,
            "schedule": cj.spec.schedule,
            "image": body.image,
        }
    except client.exceptions.ApiException as e:
        raise HTTPException(status_code=e.status, detail=e.reason)


# ---------------------------------------------------------------------------
# Ingresses
# ---------------------------------------------------------------------------
@app.get("/ingresses", summary="List all ingresses across all namespaces")
def list_ingresses(namespace: Optional[str] = None):
    try:
        if namespace:
            ingresses = networking_v1.list_namespaced_ingress(namespace=namespace)
        else:
            ingresses = networking_v1.list_ingress_for_all_namespaces()

        return {
            "ingresses": [
                {
                    "name": ing.metadata.name,
                    "namespace": ing.metadata.namespace,
                    "ingress_class": ing.spec.ingress_class_name,
                    "rules": [
                        {
                            "host": rule.host,
                            "paths": [
                                {
                                    "path": p.path,
                                    "path_type": p.path_type,
                                    "backend_service": p.backend.service.name
                                    if p.backend and p.backend.service
                                    else None,
                                }
                                for p in (rule.http.paths if rule.http else [])
                            ],
                        }
                        for rule in (ing.spec.rules or [])
                    ],
                }
                for ing in ingresses.items
            ]
        }
    except client.exceptions.ApiException as e:
        raise HTTPException(status_code=e.status, detail=e.reason)


# ---------------------------------------------------------------------------
# OpenAI Tool definitions
# ---------------------------------------------------------------------------
K8S_TOOLS = [
    {
        "type": "function",
        "name": "list_pods",
        "description": "List all Kubernetes pods. Optionally filter by namespace.",
        "parameters": {
            "type": "object",
            "properties": {
                "namespace": {
                    "type": "string",
                    "description": "Kubernetes namespace to filter pods. Omit to list all namespaces.",
                }
            },
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "list_services",
        "description": "List all Kubernetes services. Optionally filter by namespace.",
        "parameters": {
            "type": "object",
            "properties": {
                "namespace": {
                    "type": "string",
                    "description": "Kubernetes namespace to filter services. Omit to list all namespaces.",
                }
            },
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "list_ingresses",
        "description": "List all Kubernetes ingresses. Optionally filter by namespace.",
        "parameters": {
            "type": "object",
            "properties": {
                "namespace": {
                    "type": "string",
                    "description": "Kubernetes namespace to filter ingresses. Omit to list all namespaces.",
                }
            },
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "create_pod",
        "description": "Create a new Kubernetes pod with the given name, image, and namespace.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the pod."},
                "image": {"type": "string", "description": "Container image to use (e.g. nginx:latest)."},
                "namespace": {
                    "type": "string",
                    "description": "Namespace to create the pod in. Defaults to 'default'.",
                },
            },
            "required": ["name", "image"],
        },
    },
    {
        "type": "function",
        "name": "create_deployment",
        "description": (
            "Create a Kubernetes Deployment. Always confirm the container image, "
            "the number of replicas, and the resource requirements (CPU and memory) "
            "with the user before calling this. Ask follow-up questions if any are missing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the deployment."},
                "image": {"type": "string", "description": "Container image (e.g. nginx:1.27)."},
                "replicas": {"type": "integer", "description": "Number of pod replicas to run.", "minimum": 1},
                "cpu": {"type": "string", "description": "CPU request/limit per pod (e.g. '100m', '0.5', '1')."},
                "memory": {"type": "string", "description": "Memory request/limit per pod (e.g. '128Mi', '512Mi', '1Gi')."},
                "namespace": {
                    "type": "string",
                    "description": "Namespace to create the deployment in. Defaults to 'default'.",
                },
            },
            "required": ["name", "image", "replicas", "cpu", "memory"],
        },
    },
    {
        "type": "function",
        "name": "create_service",
        "description": (
            "Create a Kubernetes Service. Always confirm the service name, the port the "
            "service should expose, and the target_port on the backing pods. Ask which "
            "type to use (ClusterIP, NodePort, LoadBalancer) if not provided."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the service."},
                "port": {"type": "integer", "description": "Port the service exposes."},
                "target_port": {"type": "integer", "description": "Container port the service routes to."},
                "type": {
                    "type": "string",
                    "description": "Service type. One of ClusterIP, NodePort, LoadBalancer. Defaults to ClusterIP.",
                    "enum": ["ClusterIP", "NodePort", "LoadBalancer"],
                },
                "protocol": {"type": "string", "description": "Protocol (TCP or UDP). Defaults to TCP."},
                "selector": {
                    "type": "object",
                    "description": "Label selector matching the target pods. Defaults to {'app': name}.",
                    "additionalProperties": {"type": "string"},
                },
                "namespace": {"type": "string", "description": "Namespace. Defaults to 'default'."},
            },
            "required": ["name", "port", "target_port"],
        },
    },
    {
        "type": "function",
        "name": "list_daemonsets",
        "description": "List all Kubernetes DaemonSets. Optionally filter by namespace.",
        "parameters": {
            "type": "object",
            "properties": {
                "namespace": {
                    "type": "string",
                    "description": "Kubernetes namespace to filter daemonsets. Omit to list all namespaces.",
                }
            },
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "create_daemonset",
        "description": (
            "Create a Kubernetes DaemonSet (one pod per node). Confirm the name and image "
            "with the user before calling this."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the daemonset."},
                "image": {"type": "string", "description": "Container image."},
                "namespace": {"type": "string", "description": "Namespace. Defaults to 'default'."},
            },
            "required": ["name", "image"],
        },
    },
    {
        "type": "function",
        "name": "list_cronjobs",
        "description": "List all Kubernetes CronJobs. Optionally filter by namespace.",
        "parameters": {
            "type": "object",
            "properties": {
                "namespace": {
                    "type": "string",
                    "description": "Kubernetes namespace to filter cronjobs. Omit to list all namespaces.",
                }
            },
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "create_cronjob",
        "description": (
            "Create a Kubernetes CronJob. Always confirm the name, image, and cron schedule "
            "(e.g. '*/5 * * * *' for every 5 minutes) with the user before calling this."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the cronjob."},
                "image": {"type": "string", "description": "Container image."},
                "schedule": {
                    "type": "string",
                    "description": "Cron expression (e.g. '0 * * * *' for hourly, '*/5 * * * *' for every 5 min).",
                },
                "command": {
                    "type": "array",
                    "description": "Optional container command, as an array of strings.",
                    "items": {"type": "string"},
                },
                "namespace": {"type": "string", "description": "Namespace. Defaults to 'default'."},
            },
            "required": ["name", "image", "schedule"],
        },
    },
]


def _dispatch_tool(tool_name: str, args: dict):
    """Call the matching local function and return a JSON-serialisable result."""
    if tool_name == "list_pods":
        return list_pods(namespace=args.get("namespace"))
    if tool_name == "list_services":
        return list_services(namespace=args.get("namespace"))
    if tool_name == "list_ingresses":
        return list_ingresses(namespace=args.get("namespace"))
    if tool_name == "create_pod":
        body = PodCreate(
            name=args["name"],
            image=args["image"],
            namespace=args.get("namespace", "default"),
        )
        return create_pod(body)
    if tool_name == "create_deployment":
        body = DeploymentCreate(
            name=args["name"],
            image=args["image"],
            replicas=int(args["replicas"]),
            cpu=args["cpu"],
            memory=args["memory"],
            namespace=args.get("namespace", "default"),
        )
        return create_deployment(body)
    if tool_name == "create_service":
        body = ServiceCreate(
            name=args["name"],
            port=int(args["port"]),
            target_port=int(args["target_port"]),
            type=args.get("type", "ClusterIP"),
            protocol=args.get("protocol", "TCP"),
            selector=args.get("selector"),
            namespace=args.get("namespace", "default"),
        )
        return create_service(body)
    if tool_name == "list_daemonsets":
        return list_daemonsets(namespace=args.get("namespace"))
    if tool_name == "create_daemonset":
        body = DaemonSetCreate(
            name=args["name"],
            image=args["image"],
            namespace=args.get("namespace", "default"),
        )
        return create_daemonset(body)
    if tool_name == "list_cronjobs":
        return list_cronjobs(namespace=args.get("namespace"))
    if tool_name == "create_cronjob":
        body = CronJobCreate(
            name=args["name"],
            image=args["image"],
            schedule=args["schedule"],
            command=args.get("command"),
            namespace=args.get("namespace", "default"),
        )
        return create_cronjob(body)
    raise ValueError(f"Unknown tool: {tool_name}")


class AIQuery(BaseModel):
    query: str


@app.post("/ai/query", summary="Ask the AI to interact with your Kubernetes cluster")
def ai_query(body: AIQuery):
    """
    Natural-language endpoint. The AI decides which Kubernetes tools to call,
    executes them, and returns a human-readable answer.

    Example queries:
      - "List all pods in the default namespace"
      - "Show me all services"
      - "Create a pod named test-pod using the nginx image"
    """
    messages = [{"role": "user", "content": body.query}]

    # First call — let the model decide which tools to invoke
    response = _throttled_create(
        model="gpt-5.4-mini",
        input=messages,
        tools=K8S_TOOLS,
        store=True,
    )

    tool_results = []

    # Execute every tool the model requested
    for item in response.output:
        if item.type == "function_call":
            args = json.loads(item.arguments)
            try:
                result = _dispatch_tool(item.name, args)
            except Exception as exc:
                result = {"error": str(exc)}

            tool_results.append({
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps(result),
            })

    # If tools were used, send results back for a final natural-language reply
    if tool_results:
        followup_input = messages + list(response.output) + tool_results
        response = _throttled_create(
            model="gpt-5.4-mini",
            input=followup_input,
            store=True,
        )

    final_text = next(
        (
            item.content[0].text
            for item in response.output
            if hasattr(item, "content") and item.content
        ),
        "No response generated.",
    )

    return {"answer": final_text, "tools_called": [t["call_id"] if "call_id" in t else t for t in tool_results]}


# ---------------------------------------------------------------------------
# Chat UI
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def chat_ui(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("k8s_api:app", host="0.0.0.0", port=8000, reload=True)
