import boto3
from botocore.config import Config

from src.config import Settings
from src.services.storage.client import StorageClient


def _make_s3(settings: Settings, endpoint_url):
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        # No keys on AWS: boto3 falls back to the instance's IAM role
        aws_access_key_id=settings.s3_access_key or None,
        aws_secret_access_key=settings.s3_secret_key or None,
        region_name=settings.s3_region,
        config=Config(
            signature_version="s3v4",
            # A local S3 server needs path-style URLs; real S3 prefers bucket.s3.<region>.amazonaws.com
            s3={"addressing_style": "path" if endpoint_url else "virtual"},
            connect_timeout=3,
            read_timeout=30,
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    )


def make_storage_client(settings: Settings) -> StorageClient:
    """Create an S3 client. An empty endpoint URL means real AWS S3."""
    s3 = _make_s3(settings, settings.s3_endpoint_url or None)
    presign = _make_s3(settings, settings.s3_public_endpoint_url) if settings.s3_public_endpoint_url else None
    return StorageClient(s3, settings.s3_bucket, presign_client=presign, presigned_url_ttl=settings.s3_presigned_url_ttl)
