from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4


@dataclass
class Job:
    id: UUID
    status: str = "queued"
    progress: int = 0
    input_files: list[str] | None = None
    meta: dict[str, Any] | None = None


class FileJobStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _job_path(self, job_id: UUID) -> Path:
        return self.root / f"{job_id}.json"

    def create(self, meta: dict[str, Any] | None = None) -> Job:
        job = Job(id=uuid4(), status="queued", progress=0, input_files=[], meta=meta or {})
        self.save(job)
        return job

    def save(self, job: Job) -> None:
        path = self._job_path(job.id)
        payload = {
            "id": str(job.id),
            "status": job.status,
            "progress": job.progress,
            "input_files": job.input_files or [],
            "meta": job.meta or {},
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    def load(self, job_id: UUID) -> Job | None:
        path = self._job_path(job_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return Job(
            id=UUID(data["id"]),
            status=data.get("status", "queued"),
            progress=int(data.get("progress", 0)),
            input_files=list(data.get("input_files", [])),
            meta=data.get("meta", {}),
        )


