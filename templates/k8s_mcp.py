from mcp.server.fastmcp import FastMCP
from kubernetes import client, config
from pydantic import BaseModel
from typing import Optional
import os

mcp = FastMCP("Kubernetes MCP Server")

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
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def list_namespaces() -> dict:
    """List all Kubernetes namespaces.

    Use this before creating resources in a non-default namespace to verify the
    namespace exists; if it does not, call create_namespace first.
    """
    try:
        namespaces = v1.list_namespace()
        return {
            "namespaces": [
                {
                    "name": ns.metadata.name,
                    "status": ns.status.phase,
                    "labels": ns.metadata.labels or {},
                }
                for ns in namespaces.items
            ]
        }
    except client.exceptions.ApiException as e:
        return {"error": e.reason, "status": e.status}


@mcp.tool()
def create_namespace(name: str, labels: Optional[dict] = None) -> dict:
    """Create a Kubernetes namespace.

    Call this when the user wants to create a resource in a namespace that does not
    yet exist (verify with list_namespaces).

    Args:
        name: Name of the namespace.
        labels: Optional labels to apply to the namespace.
    """
    ns_manifest = client.V1Namespace(
        metadata=client.V1ObjectMeta(name=name, labels=labels),
    )
    try:
        ns = v1.create_namespace(body=ns_manifest)
        return {
            "message": "Namespace created successfully",
            "name": ns.metadata.name,
            "status": ns.status.phase,
        }
    except client.exceptions.ApiException as e:
        return {"error": e.reason, "status": e.status}


@mcp.tool()
def list_pods(namespace: Optional[str] = None) -> dict:
    """List all Kubernetes pods. Optionally filter by namespace."""
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
        return {"error": e.reason, "status": e.status}


@mcp.tool()
def list_services(namespace: Optional[str] = None) -> dict:
    """List all Kubernetes services. Optionally filter by namespace."""
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
        return {"error": e.reason, "status": e.status}


@mcp.tool()
def list_ingresses(namespace: Optional[str] = None) -> dict:
    """List all Kubernetes ingresses. Optionally filter by namespace."""
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
        return {"error": e.reason, "status": e.status}


@mcp.tool()
def create_pod(name: str, image: str, namespace: str = "default", labels: Optional[dict] = None) -> dict:
    """Create a new Kubernetes pod.

    Args:
        name: Name of the pod.
        image: Container image to use (e.g. nginx:latest).
        namespace: Namespace to create the pod in. Defaults to 'default'.
        labels: Optional labels to apply to the pod.
    """
    pod_manifest = client.V1Pod(
        metadata=client.V1ObjectMeta(
            name=name,
            namespace=namespace,
            labels=labels or {"app": "mcp-created"},
        ),
        spec=client.V1PodSpec(
            containers=[client.V1Container(name=name, image=image)],
            restart_policy="Always",
        ),
    )
    try:
        pod = v1.create_namespaced_pod(namespace=namespace, body=pod_manifest)
        return {
            "message": "Pod created successfully",
            "name": pod.metadata.name,
            "namespace": pod.metadata.namespace,
            "status": pod.status.phase,
        }
    except client.exceptions.ApiException as e:
        return {"error": e.reason, "status": e.status}


@mcp.tool()
def create_deployment(
    name: str,
    image: str,
    replicas: int,
    cpu: str,
    memory: str,
    namespace: str = "default",
    labels: Optional[dict] = None,
) -> dict:
    """Create a Kubernetes Deployment.

    Always confirm image, replica count, and resource requirements (CPU + memory)
    with the user before calling this. Ask follow-up questions if any are missing.

    Args:
        name: Name of the deployment.
        image: Container image (e.g. nginx:1.27).
        replicas: Number of pod replicas to run.
        cpu: CPU request/limit per pod (e.g. '100m', '0.5', '1').
        memory: Memory request/limit per pod (e.g. '128Mi', '512Mi', '1Gi').
        namespace: Namespace to create the deployment in. Defaults to 'default'.
        labels: Optional labels for the deployment and its pods.
    """
    pod_labels = labels or {"app": name}
    resources = client.V1ResourceRequirements(
        requests={"cpu": cpu, "memory": memory},
        limits={"cpu": cpu, "memory": memory},
    )

    deployment_manifest = client.V1Deployment(
        metadata=client.V1ObjectMeta(name=name, namespace=namespace, labels=pod_labels),
        spec=client.V1DeploymentSpec(
            replicas=replicas,
            selector=client.V1LabelSelector(match_labels=pod_labels),
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(labels=pod_labels),
                spec=client.V1PodSpec(
                    containers=[
                        client.V1Container(
                            name=name,
                            image=image,
                            resources=resources,
                        )
                    ],
                ),
            ),
        ),
    )
    try:
        d = apps_v1.create_namespaced_deployment(namespace=namespace, body=deployment_manifest)
        return {
            "message": "Deployment created successfully",
            "name": d.metadata.name,
            "namespace": d.metadata.namespace,
            "replicas": d.spec.replicas,
            "image": image,
            "cpu": cpu,
            "memory": memory,
        }
    except client.exceptions.ApiException as e:
        return {"error": e.reason, "status": e.status}


@mcp.tool()
def create_service(
    name: str,
    port: int,
    target_port: int,
    type: str = "ClusterIP",
    protocol: str = "TCP",
    selector: Optional[dict] = None,
    namespace: str = "default",
) -> dict:
    """Create a Kubernetes Service.

    Always confirm the service name, the exposed port, and the target_port on the
    backing pods with the user before calling this.

    Args:
        name: Name of the service.
        port: Port the service exposes.
        target_port: Container port the service routes to.
        type: Service type (ClusterIP, NodePort, LoadBalancer). Defaults to ClusterIP.
        protocol: Protocol (TCP or UDP). Defaults to TCP.
        selector: Label selector matching backing pods. Defaults to {'app': name}.
        namespace: Namespace. Defaults to 'default'.
    """
    svc_selector = selector or {"app": name}
    service_manifest = client.V1Service(
        metadata=client.V1ObjectMeta(name=name, namespace=namespace),
        spec=client.V1ServiceSpec(
            type=type,
            selector=svc_selector,
            ports=[client.V1ServicePort(port=port, target_port=target_port, protocol=protocol)],
        ),
    )
    try:
        svc = v1.create_namespaced_service(namespace=namespace, body=service_manifest)
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
        return {"error": e.reason, "status": e.status}


@mcp.tool()
def list_daemonsets(namespace: Optional[str] = None) -> dict:
    """List all Kubernetes DaemonSets. Optionally filter by namespace."""
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
        return {"error": e.reason, "status": e.status}


@mcp.tool()
def create_daemonset(
    name: str,
    image: str,
    namespace: str = "default",
    labels: Optional[dict] = None,
) -> dict:
    """Create a Kubernetes DaemonSet (one pod per node).

    Confirm the name and image with the user before calling this.

    Args:
        name: Name of the daemonset.
        image: Container image.
        namespace: Namespace. Defaults to 'default'.
        labels: Optional labels.
    """
    pod_labels = labels or {"app": name}
    ds_manifest = client.V1DaemonSet(
        metadata=client.V1ObjectMeta(name=name, namespace=namespace, labels=pod_labels),
        spec=client.V1DaemonSetSpec(
            selector=client.V1LabelSelector(match_labels=pod_labels),
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(labels=pod_labels),
                spec=client.V1PodSpec(
                    containers=[client.V1Container(name=name, image=image)],
                ),
            ),
        ),
    )
    try:
        ds = apps_v1.create_namespaced_daemon_set(namespace=namespace, body=ds_manifest)
        return {
            "message": "DaemonSet created successfully",
            "name": ds.metadata.name,
            "namespace": ds.metadata.namespace,
            "image": image,
        }
    except client.exceptions.ApiException as e:
        return {"error": e.reason, "status": e.status}


@mcp.tool()
def list_cronjobs(namespace: Optional[str] = None) -> dict:
    """List all Kubernetes CronJobs. Optionally filter by namespace."""
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
        return {"error": e.reason, "status": e.status}


@mcp.tool()
def create_cronjob(
    name: str,
    image: str,
    schedule: str,
    namespace: str = "default",
    command: Optional[list] = None,
) -> dict:
    """Create a Kubernetes CronJob.

    Always confirm the name, image, and cron schedule (e.g. '*/5 * * * *' for every
    5 minutes) with the user before calling this.

    Args:
        name: Name of the cronjob.
        image: Container image.
        schedule: Cron expression (e.g. '0 * * * *' hourly, '*/5 * * * *' every 5 min).
        namespace: Namespace. Defaults to 'default'.
        command: Optional container command as a list of strings.
    """
    container = client.V1Container(name=name, image=image, command=command)
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
        metadata=client.V1ObjectMeta(name=name, namespace=namespace),
        spec=client.V1CronJobSpec(schedule=schedule, job_template=job_template),
    )
    try:
        cj = batch_v1.create_namespaced_cron_job(namespace=namespace, body=cj_manifest)
        return {
            "message": "CronJob created successfully",
            "name": cj.metadata.name,
            "namespace": cj.metadata.namespace,
            "schedule": cj.spec.schedule,
            "image": image,
        }
    except client.exceptions.ApiException as e:
        return {"error": e.reason, "status": e.status}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
# Default: streamable-http (so the FastAPI chat UI can connect over HTTP without extra env vars).
# Override with MCP_TRANSPORT=stdio for clients (e.g. Cursor) that spawn this process directly,
# or MCP_TRANSPORT=sse for the legacy SSE transport. Port is controlled by FASTMCP_PORT
# (applied directly to mcp.settings since FastMCP doesn't always pick up the env var on its own).
if __name__ == "__main__":
    transport = os.environ.get("MCP_TRANSPORT", "streamable-http").strip().lower()
    port_env = os.environ.get("FASTMCP_PORT") or os.environ.get("MCP_PORT")
    if port_env:
        try:
            mcp.settings.port = int(port_env)
        except ValueError:
            raise SystemExit(f"Invalid FASTMCP_PORT: {port_env!r}")
    host_env = os.environ.get("FASTMCP_HOST") or os.environ.get("MCP_HOST")
    if host_env:
        mcp.settings.host = host_env

    if transport in ("sse", "streamable-http"):
        mcp.run(transport=transport)  # type: ignore[arg-type]
    elif transport == "stdio":
        mcp.run()
    else:
        raise SystemExit(f"Unknown MCP_TRANSPORT: {transport!r} (expected stdio, sse, or streamable-http)")
