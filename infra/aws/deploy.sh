#!/bin/bash
# Runs ON the server (sent by GitHub Actions through SSM Run Command, as root):
#   deploy.sh <image tag>
# The bundle (compose.prod.yml + infra/) for that tag has already been unpacked into /opt/video-rag.
set -euo pipefail
TAG="$1"
APP=/opt/video-rag
PROJECT=video-rag
REGION=$(TOKEN=$(curl -fsS -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60") \
  && curl -fsS -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/placement/region)
cd "$APP"

# 1. Settings: every parameter under /video-rag/ becomes a line of .env (readable by root only)
umask 077
aws ssm get-parameters-by-path --region "$REGION" --path "/$PROJECT/" --with-decryption \
  --query "Parameters[].[Name,Value]" --output text \
  | while IFS=$'\t' read -r name value; do echo "${name##*/}=${value}"; done > .env.new
echo "IMAGE_TAG=$TAG" >> .env.new
for required in SITE_ADDRESS IMAGE_REGISTRY S3_BUCKET SESSION_SECRET ADMIN_TOKEN POSTGRES_PASSWORD PEXELS_API_KEY TELEGRAM_BOT_TOKEN; do
  grep -q "^$required=" .env.new || { echo "missing SSM parameter /$PROJECT/$required" >&2; exit 1; }
done
if grep -q '^LLM_PROVIDER=anthropic' .env.new && ! grep -q '^ANTHROPIC_API_KEY=' .env.new; then
  echo "missing SSM parameter /$PROJECT/ANTHROPIC_API_KEY (LLM_PROVIDER=anthropic)" >&2; exit 1
fi
mv .env.new .env
REGISTRY=$(grep '^IMAGE_REGISTRY=' .env | cut -d= -f2)

# 2. Images for this tag, then replace the containers that changed
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"
docker compose -f compose.prod.yml pull --quiet
docker compose -f compose.prod.yml up -d --remove-orphans

# 3. Wait for the API (it runs database migrations on start), then report
for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8000/api/v1/ping > /dev/null; then
    echo "deployed $TAG"
    docker image prune -af --filter "until=168h" > /dev/null || true
    exit 0
  fi
  sleep 5
done
echo "API did not become healthy" >&2
docker compose -f compose.prod.yml ps
docker compose -f compose.prod.yml logs --tail 50 api >&2
exit 1
