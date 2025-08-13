from __future__ import annotations

import os
import asyncio
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SearchParams
import google.generativeai as genai
import sqlite3

# Load environment
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
N8N_URL = os.getenv("N8N_URL", "http://localhost:5678")
N8N_API_KEY = os.getenv("N8N_API_KEY")
N8N_BASIC_AUTH_USER = os.getenv("N8N_BASIC_AUTH_USER")
N8N_BASIC_AUTH_PASSWORD = os.getenv("N8N_BASIC_AUTH_PASSWORD")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Configure Gemini if key provided
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# App
app = FastAPI(title="MCP Tools Server (skeleton)", version="0.2.3")

# --- Memory (SQLite) ---
DB_PATH = os.path.join(os.path.dirname(__file__), "data.db")

def _get_db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT,
            text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    return conn

class SaveNoteBody(BaseModel):
    topic: str
    text: str

@app.post("/tool/memory/save_note")
async def save_note(body: SaveNoteBody) -> Dict[str, Any]:
    conn = _get_db_conn()
    with conn:
        conn.execute("INSERT INTO notes(topic, text) VALUES(?, ?)", (body.topic, body.text))
    return {"status": "ok"}

@app.get("/tool/memory/search")
async def search_notes(query: str, limit: int = 20) -> Dict[str, Any]:
    conn = _get_db_conn()
    cur = conn.execute(
        "SELECT id, topic, text, created_at FROM notes WHERE topic LIKE ? OR text LIKE ? ORDER BY id DESC LIMIT ?",
        (f"%{query}%", f"%{query}%", limit),
    )
    rows = [
        {"id": r[0], "topic": r[1], "text": r[2], "created_at": r[3]}
        for r in cur.fetchall()
    ]
    return {"results": rows}

# --- n8n ---
class RunWorkflowBody(BaseModel):
    workflow_id: str
    payload: Dict[str, Any] = {}

class RunWebhookBody(BaseModel):
    path: str
    method: str = "POST"
    headers: Dict[str, str] = {}
    body: Dict[str, Any] = {}
    retries: int = 5
    delay_seconds: float = 2.0

class RunByNameBody(BaseModel):
    name: str
    method: str = "POST"
    headers: Dict[str, str] = {}
    body: Dict[str, Any] = {}
    prefer_webhook: bool = True

def _n8n_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if N8N_API_KEY:
        headers["X-N8N-API-KEY"] = N8N_API_KEY
    return headers

_def_auth = (N8N_BASIC_AUTH_USER, N8N_BASIC_AUTH_PASSWORD) if N8N_BASIC_AUTH_USER and N8N_BASIC_AUTH_PASSWORD else None

@app.get("/tool/n8n/workflows")
async def list_workflows() -> Dict[str, Any]:
    url = f"{N8N_URL.rstrip('/')}/api/v1/workflows"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, headers=_n8n_headers(), auth=_def_auth)
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return resp.json()

@app.get("/tool/n8n/ready")
async def n8n_ready() -> Dict[str, Any]:
    url = f"{N8N_URL.rstrip('/')}/api/v1/workflows"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url, headers=_n8n_headers(), auth=_def_auth)
            return {"ok": resp.status_code == 200}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

@app.post("/tool/n8n/activate/{workflow_id}")
async def activate_workflow(workflow_id: str) -> Dict[str, Any]:
    url = f"{N8N_URL.rstrip('/')}/api/v1/workflows/{workflow_id}/activate"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, headers=_n8n_headers(), auth=_def_auth)
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return {"status": "ok"}

@app.post("/tool/n8n/run_webhook")
async def run_webhook(body: RunWebhookBody) -> Dict[str, Any]:
    method = (body.method or "POST").upper().strip()
    url = f"{N8N_URL.rstrip('/')}/webhook/{body.path.lstrip('/')}"
    attempts = max(1, min(body.retries, 10))
    delay = max(0.5, min(body.delay_seconds, 10.0))

    last_error_text: Optional[str] = None

    async with httpx.AsyncClient(timeout=60.0) as client:
        for attempt in range(1, attempts + 1):
            try:
                request = client.build_request(
                    method,
                    url,
                    headers=body.headers or {"Content-Type": "application/json"},
                    json=body.body or {},
                )
                resp = await client.send(request)
                if 200 <= resp.status_code < 300:
                    try:
                        return {"status": "ok", "data": resp.json()}
                    except Exception:
                        return {"status": "ok", "text": resp.text}
                if resp.status_code == 404 and "webhook" in (resp.text or "").lower():
                    last_error_text = resp.text
                    await asyncio.sleep(delay)
                    continue
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as exc:
                last_error_text = str(exc)
                await asyncio.sleep(delay)
                continue

    raise HTTPException(status_code=503, detail=last_error_text or "Webhook call failed after retries")

@app.post("/tool/n8n/run_workflow")
async def run_workflow(body: RunWorkflowBody) -> Dict[str, Any]:
    run_url = f"{N8N_URL.rstrip('/')}/api/v1/workflows/{body.workflow_id}/run"
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(run_url, headers=_n8n_headers(), auth=_def_auth, json={"payload": body.payload})
        if resp.status_code == 404:
            raise HTTPException(status_code=501, detail="Direct run not available. Use /tool/n8n/run_webhook with a Webhook node path.")
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return resp.json()

@app.post("/tool/n8n/run_by_name")
async def run_by_name(body: RunByNameBody) -> Dict[str, Any]:
    # Resolve workflow by name
    async with httpx.AsyncClient(timeout=30.0) as client:
        list_url = f"{N8N_URL.rstrip('/')}/api/v1/workflows"
        resp = await client.get(list_url, headers=_n8n_headers(), auth=_def_auth)
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        data = resp.json()
        items = data.get("data", data)
        target = None
        for w in items:
            if (w.get("name") or "").lower().strip() == body.name.lower().strip():
                target = w
                break
        if not target:
            raise HTTPException(status_code=404, detail=f"Workflow not found: {body.name}")

        # Try webhook if present and preferred
        if body.prefer_webhook:
            nodes = target.get("nodes") or []
            webhook_path = None
            for n in nodes:
                if n.get("type") == "n8n-nodes-base.webhook":
                    params = n.get("parameters") or {}
                    webhook_path = params.get("path")
                    break
            if webhook_path:
                # Delegate to run_webhook
                return await run_webhook(RunWebhookBody(path=webhook_path, method=body.method, headers=body.headers, body=body.body))

        # Fallback to direct run
        wid = target.get("id")
        return await run_workflow(RunWorkflowBody(workflow_id=wid, payload=body.body))

# --- RAG with Qdrant ---
COLLECTION = "trends"
EMBED_DIM = 768  # text-embedding-004

_qdrant: Optional[QdrantClient] = None

def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        # Support Qdrant Cloud with API key, or local without key
        _qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY) if QDRANT_API_KEY else QdrantClient(url=QDRANT_URL)
        try:
            collections = [c.name for c in _qdrant.get_collections().collections]
            if COLLECTION not in collections:
                _qdrant.create_collection(
                    collection_name=COLLECTION,
                    vectors=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
                )
        except Exception:
            pass
    return _qdrant

class UpsertDoc(BaseModel):
    id: Optional[str] = None
    text: str
    source: Optional[str] = None
    metadata: Dict[str, Any] = {}

@app.post("/tool/rag/upsert")
async def rag_upsert(body: UpsertDoc) -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")
    q = _get_qdrant()
    emb = genai.embed_content(model="text-embedding-004", content=body.text)
    vector = emb["embedding"]["values"] if isinstance(emb, dict) else emb.embedding.values
    payload = {"text": body.text, "source": body.source, **(body.metadata or {})}
    q.upsert(collection_name=COLLECTION, points=[{"id": body.id or None, "vector": vector, "payload": payload}])
    return {"status": "ok"}

@app.get("/tool/rag/search")
async def rag_search(query: str, top_k: int = 5) -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")
    q = _get_qdrant()
    emb = genai.embed_content(model="text-embedding-004", content=query)
    vector = emb["embedding"]["values"] if isinstance(emb, dict) else emb.embedding.values
    res = q.search(collection_name=COLLECTION, query_vector=vector, limit=top_k, search_params=SearchParams(hnsw_ef=64))
    results = [{"id": str(p.id), "score": p.score, "payload": p.payload} for p in res]
    return {"results": results}

# Entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastmcp_server.server:app", host="0.0.0.0", port=8000, reload=False)
