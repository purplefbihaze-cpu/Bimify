"""Storage abstraction (S3/MinIO or local filesystem fallback)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ObjectStorage(Protocol):
    def put_bytes(self, key: str, data: bytes) -> str:  # returns uri
        ...

    def put_file(self, key: str, src_path: Path) -> str:  # returns uri
        ...



