from __future__ import annotations

from pathlib import Path
from typing import Optional

import boto3


class S3Storage:
    def __init__(self, bucket: str, prefix: str = "", region: Optional[str] = None) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        session = boto3.session.Session(region_name=region) if region else boto3.session.Session()
        self.client = session.client("s3")

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def put_bytes(self, key: str, data: bytes) -> str:
        s3_key = self._key(key)
        self.client.put_object(Bucket=self.bucket, Key=s3_key, Body=data)
        return f"s3://{self.bucket}/{s3_key}"

    def put_file(self, key: str, src_path: Path) -> str:
        s3_key = self._key(key)
        self.client.upload_file(str(src_path), self.bucket, s3_key)
        return f"s3://{self.bucket}/{s3_key}"

    def get_presigned_url(self, key: str, expires: int = 3600) -> str:
        s3_key = self._key(key)
        return self.client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": self.bucket, "Key": s3_key},
            ExpiresIn=expires,
        )


__all__ = ["S3Storage"]


