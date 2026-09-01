from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

from subprojects.dataset_reviewer.app import create_app
from subprojects.dataset_reviewer.dataset_store import DatasetStore


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_dataset(path: Path) -> list[dict]:
    records = [
        {
            "messages": [
                {"role": "system", "content": "You are Yuki."},
                {"role": "user", "content": "Say hello."},
                {"role": "assistant", "content": "Hello!"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are Yuki."},
                {"role": "user", "content": "Say hello."},
                {"role": "assistant", "content": "Hello!"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": "Needle: explain this."},
                {"role": "assistant", "content": "A rough answer."},
            ],
            "source": "synthetic-test",
        },
        {
            "messages": [
                {"role": "system", "content": "You are Yuki."},
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Follow-up"},
                {"role": "assistant", "content": "Follow-up answer"},
            ]
        },
    ]
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return records


def _client(dataset: Path, database: Path, exports: Path, static: Path) -> TestClient:
    return TestClient(create_app(dataset, database, exports, static))


def test_index_is_stable_lazy_and_duplicate_safe(tmp_path: Path) -> None:
    source = tmp_path / "samples.jsonl"
    _write_dataset(source)

    first = DatasetStore(source)
    second = DatasetStore(source)

    assert first.fingerprint == second.fingerprint
    assert [pointer.sample_id for pointer in first.pointers] == [
        pointer.sample_id for pointer in second.pointers
    ]
    assert first.pointers[0].row_hash == first.pointers[1].row_hash
    assert first.pointers[0].sample_id != first.pointers[1].sample_id
    assert first.describe(3)["messages"][4]["content"] == "Follow-up answer"


def test_review_restart_undo_navigation_and_exports(tmp_path: Path) -> None:
    source = tmp_path / "samples.jsonl"
    records = _write_dataset(source)
    source_before = _hash(source)
    database = tmp_path / "review.sqlite3"
    export_dir = tmp_path / "exports"
    static_dir = tmp_path / "missing-static"

    with _client(source, database, export_dir, static_dir) as client:
        dataset = client.get("/api/dataset").json()
        sample_zero = client.get("/api/samples/0").json()
        sample_one = client.get("/api/samples/1").json()
        assert sample_zero["id"] != sample_one["id"]
        assert dataset["counts"]["remaining"] == 4

        response = client.put("/api/drafts", json={
            "sample_id": sample_zero["id"],
            "verdict": "bad",
            "reasons": ["bad reasoning"],
            "note": "Needs a clearer answer.",
            "corrected_text": "",
        })
        assert response.status_code == 200

    with _client(source, database, export_dir, static_dir) as client:
        restored = client.get("/api/samples/0").json()["draft"]
        assert restored["verdict"] == "bad"
        assert restored["note"] == "Needs a clearer answer."

        sample_zero = client.get("/api/samples/0").json()
        sample_one = client.get("/api/samples/1").json()
        sample_two = client.get("/api/samples/2").json()

        assert client.post("/api/reviews", json={
            "sample_id": sample_zero["id"], "verdict": "good", "duration_ms": 420,
        }).status_code == 200
        assert client.post("/api/reviews", json={
            "sample_id": sample_one["id"], "verdict": "bad",
            "reasons": ["duplicate"], "note": "Exact duplicate.", "duration_ms": 300,
        }).status_code == 200

        corrected = {
            **records[2],
            "messages": [*records[2]["messages"][:-1], {
                "role": "assistant", "content": "A corrected, useful answer."
            }],
        }
        assert client.post("/api/reviews", json={
            "sample_id": sample_two["id"], "verdict": "fix",
            "reasons": ["too short"], "note": "Expand it.",
            "corrected_text": json.dumps(corrected), "duration_ms": 900,
        }).status_code == 200

        # Changing a verdict creates a revision; undo restores the prior state.
        assert client.post("/api/reviews", json={
            "sample_id": sample_one["id"], "verdict": "unsure", "duration_ms": 120,
        }).status_code == 200
        undone = client.post("/api/reviews/undo").json()["undone"]
        assert undone["index"] == 1
        assert client.get("/api/samples/1").json()["review"]["verdict"] == "bad"

        next_unreviewed = client.get(
            "/api/navigate", params={
                "current": -1, "direction": "next", "verdict": "unreviewed", "search": "",
            },
        ).json()["sample"]
        assert next_unreviewed["index"] == 3
        search_match = client.get(
            "/api/navigate", params={
                "current": -1, "direction": "next", "verdict": "all", "search": "needle",
            },
        ).json()["sample"]
        assert search_match["index"] == 2

        export_response = client.post("/api/exports")
        assert export_response.status_code == 200
        files = export_response.json()["files"]
        assert files["good"]["records"] == 1
        assert files["bad"]["records"] == 1
        assert files["fix"]["records"] == 1
        assert files["corrected"]["records"] == 2

        corrected_rows = [
            json.loads(line) for line in Path(files["corrected"]["path"]).read_text().splitlines()
        ]
        assert corrected_rows == [records[0], corrected]
        report = json.loads(Path(files["report"]["path"]).read_text())
        assert report["dataset"]["fingerprint"] == dataset["fingerprint"]
        assert len(report["reviews"]) == 3

    # A third application process sees the same review state.
    with _client(source, database, export_dir, static_dir) as client:
        counts = client.get("/api/dataset").json()["counts"]
        assert counts == {
            "good": 1, "bad": 1, "fix": 1, "unsure": 0,
            "reviewed": 3, "remaining": 1, "total": 4, "percent": 75.0,
        }

    assert _hash(source) == source_before


def test_validation_and_guarded_clear(tmp_path: Path) -> None:
    source = tmp_path / "samples.jsonl"
    _write_dataset(source)
    database = tmp_path / "review.sqlite3"

    with _client(source, database, tmp_path / "exports", tmp_path / "static") as client:
        dataset = client.get("/api/dataset").json()
        sample = client.get("/api/samples/0").json()
        invalid_reason = client.post("/api/reviews", json={
            "sample_id": sample["id"], "verdict": "bad", "reasons": ["made up"],
        })
        assert invalid_reason.status_code == 422
        invalid_fix = client.post("/api/reviews", json={
            "sample_id": sample["id"], "verdict": "fix", "corrected_text": "not-json",
        })
        assert invalid_fix.status_code == 422
        assert client.post("/api/reviews", json={
            "sample_id": sample["id"], "verdict": "good",
        }).status_code == 200
        assert client.post("/api/reviews/clear", json={"confirmation": "wrong"}).status_code == 422
        cleared = client.post(
            "/api/reviews/clear", json={"confirmation": dataset["clear_confirmation"]},
        )
        assert cleared.status_code == 200
        assert cleared.json()["counts"]["reviewed"] == 0
