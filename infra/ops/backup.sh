#!/bin/sh
# Nightly Postgres backup to s3://$BACKUP_BUCKET/backups/ (kept 14 days: an S3 lifecycle rule deletes older ones).
# Restore: aws s3 cp s3://.../video_rag-<date>.sql.gz - | gunzip | psql -h postgres -U video_rag video_rag
set -eu
key="backups/video_rag-$(date -u +%Y%m%dT%H%M%SZ).sql.gz"
pg_dump -h postgres -U video_rag --no-owner video_rag | gzip | aws s3 cp - "s3://$BACKUP_BUCKET/$key"
echo "backup written to s3://$BACKUP_BUCKET/$key"
