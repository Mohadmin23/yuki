"""FastAPI application for the local Yuki dataset review lab."""
from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .dataset_store import DatasetStore
from .review_store import REASONS, VERDICTS, ReviewStore

HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE.parent / "finetune" / "datasets" / "yuki_clean_v4.jsonl"
DEFAULT_DB = HERE / "data" / "reviews.sqlite3"
DEFAULT_EXPORTS = HERE / "exports"
DEFAULT_STATIC = HERE / "frontend" / "dist"


class ReviewPayload(BaseModel):
    sample_id: str
    verdict: Literal["good", "bad", "fix", "unsure"]
    reasons: list[str] = Field(default_factory=list)
    note: str = ""
    corrected_text: str = ""
    duration_ms: int = 0


class DraftPayload(BaseModel):
    sample_id: str
    verdict: Literal["bad", "fix"] | None = None
    reasons: list[str] = Field(default_factory=list)
    note: str = ""
    corrected_text: str = ""


class ClearPayload(BaseModel):
    confirmation: str


def create_app(
    dataset_path: Path | None = None,
    db_path: Path | None = None,
    export_dir: Path | None = None,
    static_dir: Path | None = None,
) -> FastAPI:
    configured_dataset = Path(
        dataset_path or os.environ.get("YUKI_REVIEW_DATASET", DEFAULT_DATASET)
    )
    configured_db = Path(db_path or os.environ.get("YUKI_REVIEW_DB", DEFAULT_DB))
    configured_exports = Path(
        export_dir or os.environ.get("YUKI_REVIEW_EXPORTS", DEFAULT_EXPORTS)
    )
    configured_static = Path(static_dir or DEFAULT_STATIC)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        dataset = DatasetStore(configured_dataset)
        reviews = ReviewStore(configured_db, dataset)
        configured_exports.mkdir(parents=True, exist_ok=True)
        application.state.dataset = dataset
        application.state.reviews = reviews
        application.state.export_dir = configured_exports.resolve()
        yield

    application = FastAPI(
        title="Yuki Dataset Review Lab",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
    )

    def services(request: Request) -> tuple[DatasetStore, ReviewStore]:
        return request.app.state.dataset, request.app.state.reviews

    def sample_response(dataset: DatasetStore, reviews: ReviewStore, index: int) -> dict:
        try:
            sample = dataset.describe(index)
        except IndexError as exc:
            raise HTTPException(status_code=404, detail="Sample index is outside the dataset") from exc
        sample["review"] = reviews.get_review(sample["id"])
        sample["draft"] = reviews.get_draft(sample["id"])
        return sample

    @application.get("/api/health")
    def health(request: Request) -> dict:
        dataset, _reviews = services(request)
        return {"ok": True, "dataset": dataset.name, "total": len(dataset)}

    @application.get("/api/dataset")
    def dataset_info(request: Request) -> dict:
        dataset, reviews = services(request)
        return {
            **dataset.metadata(),
            "counts": reviews.counts(),
            "reason_categories": list(REASONS),
            "verdicts": list(VERDICTS),
            "review_database": str(reviews.path),
            "clear_confirmation": dataset.fingerprint[:8],
        }

    @application.get("/api/samples/{index}")
    def get_sample(index: int, request: Request) -> dict:
        dataset, reviews = services(request)
        return sample_response(dataset, reviews, index)

    @application.get("/api/navigate")
    def navigate(
        request: Request,
        current: int = -1,
        direction: Literal["next", "previous"] = "next",
        verdict: str = "unreviewed",
        search: str = "",
    ) -> dict:
        dataset, reviews = services(request)
        try:
            index, wrapped = reviews.matching_index(current, direction, verdict, search)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "sample": sample_response(dataset, reviews, index) if index is not None else None,
            "wrapped": wrapped,
            "filter": verdict,
            "search": search,
        }

    @application.post("/api/reviews")
    def save_review(payload: ReviewPayload, request: Request) -> dict:
        _dataset, reviews = services(request)
        corrected = None
        if payload.corrected_text.strip():
            try:
                corrected = json.loads(payload.corrected_text)
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"Corrected sample is not valid JSON: {exc.msg}",
                ) from exc
            if not isinstance(corrected, dict):
                raise HTTPException(
                    status_code=422,
                    detail="Corrected sample must be one complete JSON object",
                )
        try:
            review = reviews.save_review(
                payload.sample_id,
                payload.verdict,
                payload.reasons,
                payload.note,
                corrected if payload.verdict == "fix" else None,
                payload.duration_ms,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown sample identifier") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"review": review, "counts": reviews.counts()}

    @application.put("/api/drafts")
    def save_draft(payload: DraftPayload, request: Request) -> dict:
        _dataset, reviews = services(request)
        try:
            return reviews.save_draft(
                payload.sample_id,
                payload.verdict,
                payload.reasons,
                payload.note,
                payload.corrected_text,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown sample identifier") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.post("/api/reviews/undo")
    def undo_review(request: Request) -> dict:
        _dataset, reviews = services(request)
        result = reviews.undo()
        return {"undone": result, "counts": reviews.counts()}

    @application.get("/api/reviews/history")
    def review_history(request: Request, limit: int = Query(100, ge=1, le=500)) -> dict:
        _dataset, reviews = services(request)
        return {"events": reviews.history(limit)}

    @application.get("/api/dashboard")
    def dashboard(request: Request) -> dict:
        _dataset, reviews = services(request)
        return reviews.dashboard()

    @application.post("/api/reviews/clear")
    def clear_reviews(payload: ClearPayload, request: Request) -> dict:
        _dataset, reviews = services(request)
        try:
            removed = reviews.clear_reviews(payload.confirmation)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"removed": removed, "counts": reviews.counts()}

    def atomic_jsonl(path: Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def atomic_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    @application.post("/api/exports")
    def create_exports(request: Request) -> dict:
        dataset, reviews = services(request)
        export_root: Path = request.app.state.export_dir
        all_reviews = reviews.all_reviews()
        by_verdict: dict[str, list[dict]] = {verdict: [] for verdict in VERDICTS}
        corrected: list[dict] = []
        report_rows = []
        for pointer in dataset.pointers:
            review = all_reviews.get(pointer.sample_id)
            if review is None:
                continue
            original = dataset.get_record(pointer.index)
            by_verdict[review["verdict"]].append(original)
            if review["verdict"] == "good":
                corrected.append(original)
            elif review["verdict"] == "fix" and review["corrected_content"] is not None:
                corrected.append(review["corrected_content"])
            report_rows.append({
                "sample_id": pointer.sample_id,
                "original_index": pointer.index,
                "row_hash": pointer.row_hash,
                **review,
            })
        stem = f"{dataset.path.stem}-{dataset.fingerprint[:8]}"
        paths = {
            "good": export_root / f"{stem}-good.jsonl",
            "bad": export_root / f"{stem}-bad.jsonl",
            "fix": export_root / f"{stem}-needs-fix.jsonl",
            "unsure": export_root / f"{stem}-unsure.jsonl",
            "corrected": export_root / f"{stem}-corrected.jsonl",
            "report": export_root / f"{stem}-review-report.json",
        }
        for verdict in VERDICTS:
            atomic_jsonl(paths[verdict], by_verdict[verdict])
        atomic_jsonl(paths["corrected"], corrected)
        atomic_json(paths["report"], {
            "dataset": dataset.metadata(),
            "counts": reviews.counts(),
            "generated_at": time.time(),
            "reviews": report_rows,
        })
        return {
            "files": {
                name: {
                    "filename": path.name,
                    "path": str(path),
                    "url": f"/api/exports/{path.name}",
                    "records": (
                        len(corrected) if name == "corrected"
                        else len(report_rows) if name == "report"
                        else len(by_verdict[name])
                    ),
                }
                for name, path in paths.items()
            },
            "source_fingerprint": dataset.fingerprint,
        }

    @application.get("/api/exports/{filename}")
    def download_export(filename: str, request: Request):
        export_root: Path = request.app.state.export_dir
        if Path(filename).name != filename:
            raise HTTPException(status_code=400, detail="Invalid export filename")
        path = export_root / filename
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Export not found")
        return FileResponse(path, filename=filename)

    assets = configured_static / "assets"
    if assets.is_dir():
        application.mount("/assets", StaticFiles(directory=assets), name="assets")

    @application.get("/{spa_path:path}", include_in_schema=False)
    def frontend(spa_path: str):
        index = configured_static / "index.html"
        if not index.is_file():
            raise HTTPException(
                status_code=503,
                detail="Frontend is not built. Run npm install && npm run build in frontend/.",
            )
        return FileResponse(index)

    return application


app = create_app()


if __name__ == "__main__":
    from granian import Granian

    print("Yuki Dataset Review Lab → http://127.0.0.1:7870")
    Granian(
        "subprojects.dataset_reviewer.app:app",
        address="127.0.0.1",
        port=7870,
        interface="asgi",
    ).serve()
