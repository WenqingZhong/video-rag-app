import logging
from pathlib import Path
from typing import Any, BinaryIO

from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)


class StorageClient:
    """S3-compatible object storage for raw videos, extracted frames and cut clips.

    Key layout (one bucket, prefixes per artifact type):
        raw/{video_id}/source.mp4
        frames/{video_id}/{frame_ms}.jpg
        clips/{video_id}/{start_ms}-{end_ms}.mp4
    """

    def __init__(self, s3_client, bucket: str, presign_client=None, presigned_url_ttl: int = 3600):
        self.s3 = s3_client
        self.bucket = bucket
        # Presigned URLs must be signed for the host the browser will call, which may differ
        # from the in-cluster endpoint (e.g. http://seaweedfs:8333 vs http://localhost:8333).
        self.presign_s3 = presign_client or s3_client
        self.presigned_url_ttl = presigned_url_ttl

    def ensure_bucket(self) -> None:
        try:
            self.s3.head_bucket(Bucket=self.bucket)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") not in ("404", "NoSuchBucket"):
                raise
            logger.info("Creating bucket %s", self.bucket)
            self.s3.create_bucket(Bucket=self.bucket)

    def upload_file(self, local_path: str | Path, key: str, content_type: str | None = None) -> str:
        extra = {"ContentType": content_type} if content_type else None
        self.s3.upload_file(str(local_path), self.bucket, key, ExtraArgs=extra)
        return key

    def upload_fileobj(self, fileobj: BinaryIO, key: str, content_type: str | None = None) -> str:
        extra = {"ContentType": content_type} if content_type else None
        self.s3.upload_fileobj(fileobj, self.bucket, key, ExtraArgs=extra)
        return key

    def download_file(self, key: str, local_path: str | Path) -> Path:
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.s3.download_file(self.bucket, key, str(local_path))
        return local_path

    def exists(self, key: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def presigned_url(self, key: str, expires_in: int | None = None) -> str:
        return self.presign_s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_in or self.presigned_url_ttl,
        )

    def health_check(self) -> dict[str, Any]:
        try:
            self.s3.head_bucket(Bucket=self.bucket)
            return {"status": "healthy", "message": f"Bucket '{self.bucket}' reachable"}
        except (BotoCoreError, ClientError) as exc:
            return {"status": "unhealthy", "message": f"Object storage check failed: {exc}"}
