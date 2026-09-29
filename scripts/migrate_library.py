"""Copy the shared library (videos without an owner) from the local stack to the AWS deployment.

    uv run python scripts/migrate_library.py --bucket <bucket> --instance-id <i-...> [--region us-east-1] [--dry-run]

1. Files: each video's source file and keyframes, from the local S3 server (SeaweedFS) to the AWS bucket.
2. Rows: the videos and their segments (transcripts, captions, CLIP vectors) as CSV, uploaded to the bucket.
3. On the server (SSM Run Command): load the rows into Postgres (existing ids are skipped, so re-running is safe)
   and rebuild the search index from them. No model is called: nothing is re-transcribed or re-captioned.

Users' private uploads are never copied. Needs AWS credentials locally (AWS_PROFILE or `aws configure`), with the
stack running (Postgres and SeaweedFS on localhost).
"""

import argparse
import io
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import Settings
from src.services.storage.factory import _make_s3

LIBRARY = "owner_id IS NULL AND status = 'ready'"
PREFIX = "deploy/library"


def copy_files(settings: Settings, keys: list[str], target, bucket: str, dry_run: bool) -> int:
    local = _make_s3(settings, settings.s3_endpoint_url)
    copied = 0
    for key in keys:
        try:
            target.head_object(Bucket=bucket, Key=key)
            continue  # already there
        except ClientError:
            pass
        if not dry_run:
            body = io.BytesIO(local.get_object(Bucket=settings.s3_bucket, Key=key)["Body"].read())
            target.upload_fileobj(body, bucket, key)
        copied += 1
    return copied


def export_csv(engine, query: str) -> bytes:
    raw = engine.raw_connection()
    try:
        out = io.StringIO()
        raw.cursor().copy_expert(f"COPY ({query}) TO STDOUT WITH CSV HEADER", out)
        return out.getvalue().encode()
    finally:
        raw.close()


SERVER_SCRIPT = """set -euo pipefail
cd /opt/video-rag
mkdir -p /tmp/library
aws s3 cp s3://{bucket}/{prefix}/videos.csv /tmp/library/videos.csv
aws s3 cp s3://{bucket}/{prefix}/segments.csv /tmp/library/segments.csv
docker compose -f compose.prod.yml cp /tmp/library/videos.csv postgres:/tmp/videos.csv
docker compose -f compose.prod.yml cp /tmp/library/segments.csv postgres:/tmp/segments.csv
vcols=$(head -1 /tmp/library/videos.csv); scols=$(head -1 /tmp/library/segments.csv)
docker compose -f compose.prod.yml exec -T postgres psql -v ON_ERROR_STOP=1 -U video_rag -d video_rag <<SQL
CREATE TEMP TABLE v (LIKE videos);
\\\\copy v($vcols) FROM '/tmp/videos.csv' CSV HEADER
INSERT INTO videos SELECT * FROM v ON CONFLICT (id) DO NOTHING;
CREATE TEMP TABLE s (LIKE segments);
\\\\copy s($scols) FROM '/tmp/segments.csv' CSV HEADER
INSERT INTO segments SELECT * FROM s ON CONFLICT (id) DO NOTHING;
SELECT count(*) AS library_videos FROM videos WHERE owner_id IS NULL;
SQL
token=$(grep '^ADMIN_TOKEN=' .env | cut -d= -f2)
curl -fsS -X POST -H "x-admin-token: $token" http://127.0.0.1:8000/api/v1/admin/reindex
"""


def run_on_server(ssm, instance_id: str, script: str) -> None:
    command = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Comment="load the library",
        Parameters={"commands": [script]},
    )["Command"]["CommandId"]
    for _ in range(60):
        time.sleep(5)
        try:
            result = ssm.get_command_invocation(CommandId=command, InstanceId=instance_id)
        except ClientError:
            continue
        if result["Status"] in ("Success", "Failed", "Cancelled", "TimedOut"):
            print(result["StandardOutputContent"], result["StandardErrorContent"], sep="\n")
            if result["Status"] != "Success":
                sys.exit(f"server step {result['Status']}")
            return
    sys.exit("server step still running after 5 minutes: check SSM Run Command history")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--dry-run", action="store_true", help="count what would be copied; change nothing")
    args = parser.parse_args()

    settings = Settings()
    engine = create_engine(settings.postgres_database_url)
    with engine.connect() as db:
        videos = db.execute(text(f"SELECT id, s3_key FROM videos WHERE {LIBRARY}")).all()
        frames = db.execute(text(
            f"SELECT frame_key FROM segments WHERE frame_key IS NOT NULL AND video_id IN (SELECT id FROM videos WHERE {LIBRARY})"
        )).scalars().all()  # fmt: skip
    keys = [v.s3_key for v in videos if v.s3_key] + list(frames)
    print(f"library: {len(videos)} videos, {len(keys)} files")

    s3 = boto3.client("s3", region_name=args.region)
    copied = copy_files(settings, keys, s3, args.bucket, args.dry_run)
    print(f"files: {copied} {'to copy' if args.dry_run else 'copied'}, {len(keys) - copied} already there")
    if args.dry_run:
        return

    ids = ", ".join(f"'{v.id}'" for v in videos)
    for name, query in {
        "videos": f"SELECT * FROM videos WHERE {LIBRARY}",
        "segments": f"SELECT * FROM segments WHERE video_id IN ({ids})",
    }.items():
        s3.put_object(Bucket=args.bucket, Key=f"{PREFIX}/{name}.csv", Body=export_csv(engine, query))
    print("rows uploaded; loading them on the server and rebuilding the index…")
    run_on_server(boto3.client("ssm", region_name=args.region), args.instance_id,
                  SERVER_SCRIPT.format(bucket=args.bucket, prefix=PREFIX))  # fmt: skip


if __name__ == "__main__":
    main()
