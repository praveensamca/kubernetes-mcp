from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from openai import OpenAI
from typing import Optional
import uvicorn
import json
import os
import time
import threading
from collections import deque
from pathlib import Path

import anyio
import mcp.types as mcp_types
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

app = FastAPI(title="Kubernetes AI Chat (MCP-only)", version="2.0.0")

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

DEFAULT_MCP_SSE_URL = os.environ.get("MCP_SSE_URL", "http://127.0.0.1:8767/sse")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")

_openai_client: Optional[OpenAI] = None


def _get_openai_client() -> OpenAI:
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
# ---------------------------------------------------------------------------
_rpm_limit = 3
_window_secs = 60
_req_times: deque = deque()
_throttle_lock = threading.Lock()


def _throttled_create(**kwargs):
    with _throttle_lock:
        now = time.monotonic()
        while _req_times and now - _req_times[0] >= _window_secs:
            _req_times.popleft()

        if len(_req_times) >= _rpm_limit:
            wait = _window_secs - (now - _req_times[0])
            if wait > 0:
                time.sleep(wait)
            now = time.monotonic()
            while _req_times and now - _req_times[0] >= _window_secs:
                _req_times.popleft()

        _req_times.append(time.monotonic())

    return _get_openai_client().responses.create(**kwargs)


# ---------------------------------------------------------------------------
# MCP helpers
# ---------------------------------------------------------------------------
def _normalize_mcp_sse_url(raw: str) -> str:
    u = raw.strip()
    if not u:
        raise HTTPException(status_code=400, detail="mcp_sse_url is empty")
    if not u.startswith(("http://", "https://")):
        u = "http://" + u
    u = u.rstrip("/")
    if not u.endswith("/sse"):
        u = u + "/sse"
    return u


def _mcp_tools_to_openai(lt: mcp_types.ListToolsResult) -> list:
    out = []
    for t in lt.tools:
        schema = t.inputSchema if isinstance(t.inputSchema, dict) else {"type": "object", "properties": {}}
        out.append({
            "type": "function",
            "name": t.name,
            "description": (t.description or "")[:8000],
            "parameters": schema,
        })
    return out


def _call_tool_result_to_plain(ct: mcp_types.CallToolResult) -> dict:
    try:
        return ct.model_dump(mode="json")
    except Exception:
        return {"error": "serialization_failed", "repr": repr(ct)}


def _response_final_text(response) -> str:
    return next(
        (
            item.content[0].text
            for item in response.output
            if hasattr(item, "content") and item.content
        ),
        "No response generated.",
    )


async def _ai_query_via_mcp_sse(sse_url: str, user_query: str) -> dict:
    norm = _normalize_mcp_sse_url(sse_url)
    messages = [{"role": "user", "content": user_query}]
    async with sse_client(norm, timeout=60.0, sse_read_timeout=600.0) as streams:
        read_s, write_s = streams
        async with ClientSession(read_s, write_s) as session:
            await session.initialize()
            lt = await session.list_tools()
            openai_tools = _mcp_tools_to_openai(lt)
            if not openai_tools:
                raise HTTPException(status_code=503, detail="MCP server returned no tools.")

            resp1 = await anyio.to_thread.run_sync(
                lambda: _throttled_create(
                    model=OPENAI_MODEL,
                    input=messages,
                    tools=openai_tools,
                    store=True,
                )
            )

            tool_results = []
            for item in resp1.output:
                if item.type == "function_call":
                    try:
                        args = json.loads(item.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    try:
                        ctr = await session.call_tool(item.name, args)
                        payload = _call_tool_result_to_plain(ctr)
                    except Exception as exc:
                        payload = {"error": str(exc)}
                    tool_results.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps(payload),
                    })

            if tool_results:
                resp_final = await anyio.to_thread.run_sync(
                    lambda r=resp1: _throttled_create(
                        model=OPENAI_MODEL,
                        input=messages + list(r.output) + tool_results,
                        store=True,
                    )
                )
            else:
                resp_final = resp1

            return {
                "answer": _response_final_text(resp_final),
                "tools_called": [t["call_id"] for t in tool_results],
                "mcp_sse_url": norm,
                "via_mcp": True,
            }


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
class AIQuery(BaseModel):
    query: str
    mcp_sse_url: Optional[str] = None


@app.post("/ai/query", summary="Ask the AI to interact with your Kubernetes cluster (via MCP)")
def ai_query(body: AIQuery):
    """
    Natural-language endpoint. Always routes tool calls through the MCP server.
    Falls back to MCP_SSE_URL / DEFAULT_MCP_SSE_URL if no mcp_sse_url is supplied.
    """
    sse_url = (body.mcp_sse_url or DEFAULT_MCP_SSE_URL).strip()
    return anyio.run(_ai_query_via_mcp_sse, sse_url, body.query)


# ---------------------------------------------------------------------------
# Chat UI
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def chat_ui(request: Request):
    return templates.TemplateResponse(
        "chat.html",
        {"request": request, "default_mcp_sse_url": DEFAULT_MCP_SSE_URL},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("k8s_api:app", host="0.0.0.0", port=8000, reload=True)
