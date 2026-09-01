"""Local FastAPI laboratory for inspecting one-shot tool-dispatch behavior."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .backends import (
    DEFAULT_XLAM_MODEL,
    DispatcherBackend,
    FixtureBackend,
    create_backend,
)
from .benchmark import BenchmarkRunner, load_cases
from .dispatcher import Dispatcher
from .registry import ToolRegistry

app = FastAPI(title="Yuki Tool Dispatcher Lab", version="0.1.0")
registry = ToolRegistry()
_active_backend: DispatcherBackend | None = None
_model_lock = threading.Lock()
_inference_lock = threading.Lock()


class StrictBody(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoadModelBody(StrictBody):
    backend: Literal["mlx", "transformers", "openai-compatible-local"] = "mlx"
    model_id: str = DEFAULT_XLAM_MODEL
    base_url: str | None = None
    api_key: str = "local"
    constrain_json: bool = True


class DispatchBody(StrictBody):
    request: str = Field(min_length=1)
    group: str = "all"
    selected: list[str] | None = None
    output_mode: Literal["native", "canonical"] = "native"
    execute: bool = False
    allow_side_effects: bool = False
    shell_mode: Literal["dry_run", "strict"] = "dry_run"


class ExecuteBody(StrictBody):
    tool: str
    arguments: dict
    group: str = "all"
    selected: list[str] | None = None
    allow_side_effects: bool = False
    shell_mode: Literal["dry_run", "strict"] = "dry_run"


class BenchmarkBody(StrictBody):
    group: str = "all"
    selected: list[str] | None = None
    router_mode: Literal[
        "model_only", "regex_only", "regex_then_model"
    ] = "model_only"
    output_mode: Literal["native", "canonical"] = "native"
    execute: bool = False
    allow_side_effects: bool = False
    shell_mode: Literal["dry_run", "strict"] = "dry_run"
    limit: int | None = Field(default=None, ge=1)


def _dispatcher(require_model: bool = True) -> Dispatcher:
    if _active_backend is None:
        if require_model:
            raise HTTPException(status_code=409, detail="Load a model first.")
        return Dispatcher(registry, FixtureBackend([]))
    return Dispatcher(registry, _active_backend)


@app.get("/api/status")
def status():
    return {
        "ready": _active_backend is not None,
        "model": _active_backend.status() if _active_backend else None,
        "tools": registry.describe(),
    }


@app.post("/api/model/load")
def load_model(body: LoadModelBody):
    global _active_backend
    try:
        with _model_lock:
            backend = create_backend(
                body.backend,
                body.model_id,
                base_url=body.base_url,
                api_key=body.api_key,
                constrain_json=body.constrain_json,
            )
            _active_backend = backend
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"{type(exc).__name__}: {exc}"
        ) from exc
    return {"ready": True, "model": backend.status()}


@app.get("/api/tools")
def tools(group: str = "all"):
    try:
        names = registry.resolve_names(group)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "group": group,
        "names": list(names),
        "schemas": registry.strict_openai_schemas(names),
        "xlam_schemas": registry.xlam_schemas(names),
    }


@app.post("/api/dispatch")
def dispatch(body: DispatchBody):
    dispatcher = _dispatcher()
    try:
        with _inference_lock:
            return dispatcher.dispatch(
                body.request,
                group=body.group,
                selected=body.selected,
                output_mode=body.output_mode,
                execute=body.execute,
                allow_side_effects=body.allow_side_effects,
                shell_mode=body.shell_mode,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/execute")
def execute(body: ExecuteBody):
    dispatcher = _dispatcher()
    try:
        return dispatcher.execute_validated(
            {"tool": body.tool, "arguments": body.arguments},
            group=body.group,
            selected=body.selected,
            allow_side_effects=body.allow_side_effects,
            shell_mode=body.shell_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/benchmark")
def benchmark(body: BenchmarkBody):
    dispatcher = _dispatcher(require_model=body.router_mode != "regex_only")
    runner = BenchmarkRunner(dispatcher)
    try:
        with _inference_lock:
            return runner.run(
                load_cases(),
                group=body.group,
                selected=body.selected,
                router_mode=body.router_mode,
                output_mode=body.output_mode,
                execute=body.execute,
                allow_side_effects=body.allow_side_effects,
                shell_mode=body.shell_mode,
                limit=body.limit,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


STATIC_DIR = Path(__file__).with_name("static")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
