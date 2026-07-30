from __future__ import annotations

import os
import tempfile
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from reg_rag.answer import answer_question
from reg_rag.product import DEMO_PRESETS, MAX_UPLOAD_BYTES, ProductRuntime
from reg_rag.retrieval import retrieve_records


class LegacyQueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=8, ge=1, le=50)
    mode: Literal["simple", "hybrid"] = "hybrid"
    generator: Literal["rule", "mock", "mock_llm", "openai"] = "rule"
    reranker: Literal["none", "heuristic", "page_aware", "section_aware", "table_aware"] = "none"


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=8, ge=1, le=50)
    mode: Literal["simple", "hybrid"] = "hybrid"
    reranker: Literal["none", "heuristic", "page_aware", "section_aware", "table_aware"] = "none"


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    requested_mode: Literal["auto", "reference_v1", "table_specialist", "cross_file"] = "auto"


class PublicRateLimiter:
    def __init__(
        self,
        *,
        window_seconds: float = 60.0,
        per_client_limit: int = 30,
        global_limit: int = 120,
    ) -> None:
        self.window_seconds = window_seconds
        self.per_client_limit = per_client_limit
        self.global_limit = global_limit
        self._clients: dict[str, deque[float]] = defaultdict(deque)
        self._global: deque[float] = deque()
        self._lock = threading.Lock()

    def allow(self, client: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            while self._global and self._global[0] <= cutoff:
                self._global.popleft()
            client_requests = self._clients[client]
            while client_requests and client_requests[0] <= cutoff:
                client_requests.popleft()
            if len(self._global) >= self.global_limit or len(client_requests) >= self.per_client_limit:
                return False
            self._global.append(now)
            client_requests.append(now)
            return True


def create_app(
    records_path: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
    upload_root: str | Path | None = None,
    public_mode: bool | None = None,
) -> FastAPI:
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    raw_records = Path(records_path or os.getenv("REG_RAG_RECORDS", "data/synthetic/records.jsonl"))
    resolved_records = raw_records if raw_records.is_absolute() else root / raw_records
    raw_uploads = Path(upload_root or os.getenv("REG_RAG_UPLOAD_ROOT", "data/user_uploads"))
    resolved_uploads = raw_uploads if raw_uploads.is_absolute() else root / raw_uploads
    is_public = (
        public_mode
        if public_mode is not None
        else os.getenv("COMPETITION_PUBLIC_MODE", "").strip().lower() in {"1", "true", "yes"}
    )
    state: dict[str, Any] = {"runtime": None, "demo_runtime": None, "temporary_root": None}
    answer_lock = threading.Lock()
    public_config = root / "configs/system/public_demo.yaml"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not resolved_records.exists():
            raise RuntimeError(f"Records file not found: {resolved_records.name}")
        runtime_uploads = resolved_uploads
        if is_public:
            temporary_root = tempfile.TemporaryDirectory(prefix="reg-rag-public-")
            state["temporary_root"] = temporary_root
            runtime_uploads = Path(temporary_root.name) / "main"
        state["runtime"] = ProductRuntime(
            root,
            resolved_records,
            runtime_uploads,
            system_config=public_config if is_public else None,
            public_mode=is_public,
        )
        if is_public:
            demo_path = root / "data/public_demo/new_document_demo.docx"
            if not demo_path.exists():
                raise RuntimeError(f"Public demo asset not found: {demo_path.name}")
            demo_runtime = ProductRuntime(
                root,
                resolved_records,
                Path(state["temporary_root"].name) / "new_document",
                system_config=public_config,
                public_mode=True,
            )
            demo_document = demo_runtime.store.upload(demo_path.name, demo_path.read_bytes())
            if demo_document["status"] != "INDEXED":
                raise RuntimeError("Public new-document demo failed to index")
            demo_runtime.refresh_user_records()
            state["demo_runtime"] = demo_runtime
        yield
        temporary_root = state.get("temporary_root")
        if temporary_root is not None:
            temporary_root.cleanup()

    app = FastAPI(
        title="Trusted Banking Regulatory RAG",
        description="银行监管可信知识助手产品 API",
        version="0.3.0",
        lifespan=lifespan,
        docs_url=None if is_public else "/docs",
        redoc_url=None if is_public else "/redoc",
        openapi_url=None if is_public else "/openapi.json",
    )
    app.state.public_mode = is_public
    app.state.public_rate_limiter = PublicRateLimiter()
    static_dir = Path(__file__).with_name("web_static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    if is_public:
        _install_public_handlers(app)

        @app.middleware("http")
        async def public_security_middleware(request: Request, call_next):
            if _is_rate_limited_path(request.url.path):
                client = _client_identity(request)
                if not app.state.public_rate_limiter.allow(client):
                    response = JSONResponse(
                        status_code=429,
                        content={"detail": "请求过于频繁，请稍后再试"},
                        headers={"Retry-After": "60"},
                    )
                    return _add_security_headers(response)
            response = await call_next(request)
            return _add_security_headers(response)

    @app.get("/", include_in_schema=False)
    def home(request: Request) -> Response:
        if is_public and request.query_params.get("demo") != "1":
            return RedirectResponse(url="/?demo=1", status_code=307)
        return FileResponse(static_dir / "index.html")

    @app.get("/health")
    def health() -> dict:
        return _runtime(state).health()

    @app.post("/answer")
    def product_answer(req: AnswerRequest) -> dict:
        if is_public:
            with answer_lock:
                return _runtime(state).answer(req.question, req.requested_mode)
        return _runtime(state).answer(req.question, req.requested_mode)

    if not is_public:

        @app.post("/search")
        def search(req: SearchRequest) -> dict:
            runtime = _runtime(state)
            hits = retrieve_records(
                runtime.system.records,
                req.query,
                top_k=req.top_k,
                mode=req.mode,
                reranker=req.reranker,
            )
            return {
                "query": req.query,
                "hits": [
                    {
                        "score": hit.score,
                        "locator": hit.record.get("chunk_id") or hit.record.get("cell_id"),
                        "doc_id": hit.record.get("doc_id"),
                        "text": hit.record.get("text") or hit.record.get("value"),
                    }
                    for hit in hits
                ],
            }

        @app.post("/query")
        def legacy_query(req: LegacyQueryRequest) -> dict:
            runtime = _runtime(state)
            return answer_question(
                runtime.system.records,
                req.question,
                top_k=req.top_k,
                mode=req.mode,
                generator=req.generator,
                reranker=req.reranker,
            )

    @app.post("/documents/upload", status_code=201)
    async def upload_document(
        request: Request,
        x_filename: str = Header(alias="X-Filename"),
    ) -> dict:
        if is_public:
            raise HTTPException(status_code=403, detail="公网演示使用隔离预置文档")
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="文件超过 20 MB 限制")
        content = await request.body()
        runtime = _runtime(state)
        try:
            item = runtime.store.upload(unquote(x_filename), content)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        runtime.refresh_user_records()
        return {"document": item}

    @app.get("/documents")
    def documents() -> dict:
        return _runtime(state).documents()

    @app.get("/documents/{doc_id}")
    def document(doc_id: str) -> dict:
        runtime = _runtime(state)
        try:
            return runtime.document(doc_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="文档不存在") from exc

    @app.post("/documents/{doc_id}/index")
    def index_document(doc_id: str) -> dict:
        if is_public:
            raise HTTPException(status_code=403, detail="公网演示知识库为只读")
        runtime = _runtime(state)
        if not doc_id.startswith("user_"):
            raise HTTPException(status_code=403, detail="官方文档为只读资源")
        try:
            item = runtime.store.index(doc_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="用户文档不存在") from exc
        runtime.refresh_user_records()
        return {"document": item}

    @app.delete("/documents/{doc_id}", status_code=204, response_class=Response)
    def delete_document(doc_id: str) -> Response:
        if is_public:
            raise HTTPException(status_code=403, detail="公网演示知识库为只读")
        runtime = _runtime(state)
        if not doc_id.startswith("user_"):
            raise HTTPException(status_code=403, detail="官方文档为只读资源")
        try:
            runtime.store.delete(doc_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="仅允许删除用户上传文档") from exc
        runtime.refresh_user_records()
        return Response(status_code=204)

    if not is_public:

        @app.get("/trace/{request_id}")
        def trace(request_id: str) -> dict:
            runtime = _runtime(state)
            if request_id not in runtime.request_traces:
                raise HTTPException(status_code=404, detail="Trace 不存在")
            return {"request_id": request_id, "trace": runtime.request_traces[request_id]}

    @app.get("/metrics/summary")
    def metrics_summary() -> dict:
        return _runtime(state).metrics()

    @app.get("/demo/presets")
    def demo_presets() -> dict:
        return {"presets": DEMO_PRESETS}

    if is_public:

        @app.post("/demo/new-document")
        def new_document_demo() -> dict:
            preset = next(item for item in DEMO_PRESETS if item["id"] == "new_document")
            with answer_lock:
                return _runtime(state, key="demo_runtime").answer(
                    preset["question"],
                    preset["mode"],
                )

    return app


def _runtime(state: dict[str, Any], *, key: str = "runtime") -> ProductRuntime:
    runtime = state.get(key)
    if runtime is None:
        raise HTTPException(status_code=503, detail="系统正在启动")
    return runtime


def _is_rate_limited_path(path: str) -> bool:
    return path in {"/answer", "/documents/upload", "/demo/new-document"} or (
        path.startswith("/documents/") and path.endswith("/index")
    )


def _client_identity(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "").split(",", 1)[0].strip()
    if forwarded and len(forwarded) <= 64:
        return forwarded
    return request.client.host if request.client else "unknown"


def _add_security_headers(response: Response) -> Response:
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self' data:; connect-src 'self'; font-src 'self'; "
        "object-src 'none'; base-uri 'self'; frame-ancestors 'none'; form-action 'self'"
    )
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Cache-Control"] = "no-store" if response.status_code >= 400 else "no-cache"
    return response


def _install_public_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, _exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": "请求参数无效"})

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        messages = {
            404: "页面或接口不存在",
            405: "请求方法不受支持",
            413: "文件超过大小限制",
            429: "请求过于频繁，请稍后再试",
            503: "服务正在启动，请稍后再试",
        }
        detail = messages.get(exc.status_code)
        if detail is None and exc.status_code == 403:
            detail = "公网演示为只读模式"
        if detail is None:
            detail = "请求未能处理"
        return JSONResponse(status_code=exc.status_code, content={"detail": detail})

    @app.exception_handler(Exception)
    async def internal_error(_request: Request, _exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": "服务暂时不可用，请稍后再试"})


app = create_app()
