"""Read-only, byte-offset indexed access to JSONL training datasets."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SamplePointer:
    index: int
    offset: int
    length: int
    row_hash: str
    sample_id: str


class DatasetStore:
    """Index a JSONL file without retaining every sample object in memory."""

    def __init__(self, path: Path):
        self.path = path.resolve(strict=True)
        if not self.path.is_file():
            raise ValueError(f"Dataset is not a file: {self.path}")
        self.fingerprint = self._fingerprint()
        self.pointers = self._build_index()
        self.name = self.path.name
        self.format = self._detect_format()

    def _fingerprint(self) -> str:
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _build_index(self) -> list[SamplePointer]:
        pointers: list[SamplePointer] = []
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                index = len(pointers)
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at source line {index + 1}: {exc}") from exc
                if not isinstance(record, dict):
                    raise TypeError(f"Dataset record {index + 1} is not a JSON object")
                row_hash = hashlib.sha256(stripped).hexdigest()
                identity = f"{self.fingerprint}:{index}:{row_hash}".encode()
                sample_id = hashlib.sha256(identity).hexdigest()[:32]
                pointers.append(SamplePointer(index, offset, len(line), row_hash, sample_id))
        if not pointers:
            raise ValueError("Dataset contains no JSON records")
        return pointers

    def _detect_format(self) -> str:
        record = self.get_record(0)
        if isinstance(record.get("messages"), list):
            return "chatml"
        if isinstance(record.get("conversations"), list):
            return "sharegpt"
        if "prompt" in record and "completion" in record:
            return "prompt_completion"
        if "instruction" in record and "output" in record:
            return "alpaca"
        return "generic"

    def __len__(self) -> int:
        return len(self.pointers)

    def get_record(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self.pointers):
            raise IndexError(index)
        pointer = self.pointers[index]
        with self.path.open("rb") as handle:
            handle.seek(pointer.offset)
            raw = handle.read(pointer.length).strip()
        record = json.loads(raw)
        if not isinstance(record, dict):
            raise TypeError(f"Dataset record {index + 1} is not an object")
        return record

    def searchable_text(self, index: int) -> str:
        return json.dumps(self.get_record(index), ensure_ascii=False, sort_keys=True)

    def describe(self, index: int) -> dict[str, Any]:
        record = self.get_record(index)
        messages, metadata = self._presentation(record)
        pointer = self.pointers[index]
        return {
            "id": pointer.sample_id,
            "index": index,
            "number": index + 1,
            "total": len(self),
            "row_hash": pointer.row_hash,
            "format": self.format,
            "messages": messages,
            "metadata": metadata,
            "raw": record,
        }

    def _presentation(
        self, record: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if isinstance(record.get("messages"), list):
            messages = [self._message(message, "unknown") for message in record["messages"]]
            return messages, {key: value for key, value in record.items() if key != "messages"}
        if isinstance(record.get("conversations"), list):
            role_map = {"human": "user", "gpt": "assistant"}
            messages = []
            for message in record["conversations"]:
                if isinstance(message, dict):
                    role = role_map.get(str(message.get("from", "unknown")), message.get("from", "unknown"))
                    messages.append(self._message(
                        {"role": role, "content": message.get("value", ""), **{
                            key: value for key, value in message.items() if key not in {"from", "value"}
                        }},
                        "unknown",
                    ))
            return messages, {key: value for key, value in record.items() if key != "conversations"}
        if "prompt" in record or "completion" in record:
            messages = [
                self._message({"role": "user", "content": record.get("prompt", "")}, "user"),
                self._message({"role": "assistant", "content": record.get("completion", "")}, "assistant"),
            ]
            return messages, {key: value for key, value in record.items() if key not in {"prompt", "completion"}}
        if "instruction" in record or "output" in record:
            user = record.get("instruction", "")
            if record.get("input"):
                user = f"{user}\n\n{record['input']}"
            messages = [
                self._message({"role": "user", "content": user}, "user"),
                self._message({"role": "assistant", "content": record.get("output", "")}, "assistant"),
            ]
            return messages, {
                key: value for key, value in record.items()
                if key not in {"instruction", "input", "output"}
            }
        return [], record

    @staticmethod
    def _message(message: Any, fallback_role: str) -> dict[str, Any]:
        if not isinstance(message, dict):
            return {"role": fallback_role, "content": message, "extra": {}}
        role = str(message.get("role", fallback_role))
        content = message.get("content", "")
        extra = {key: value for key, value in message.items() if key not in {"role", "content"}}
        return {"role": role, "content": content, "extra": extra}

    def metadata(self) -> dict[str, Any]:
        stat = self.path.stat()
        return {
            "name": self.name,
            "path": str(self.path),
            "fingerprint": self.fingerprint,
            "fingerprint_short": self.fingerprint[:12],
            "total": len(self),
            "format": self.format,
            "bytes": stat.st_size,
        }
