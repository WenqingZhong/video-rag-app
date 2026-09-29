#!/bin/bash
# First boot of the server (EC2 user data, Amazon Linux 2023 on ARM). Installs Docker and prepares the machine;
# the app itself arrives with the first deploy (deploy.sh, run by GitHub Actions over SSM).
set -euxo pipefail

dnf install -y docker
systemctl enable --now docker

# Docker Compose v2 plugin (not packaged for AL2023)
mkdir -p /usr/local/lib/docker/cli-plugins
curl -fsSL "https://github.com/docker/compose/releases/download/v2.39.2/docker-compose-linux-aarch64" \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

# OpenSearch needs more memory-mapped areas than the default
echo "vm.max_map_count=262144" > /etc/sysctl.d/99-opensearch.conf
sysctl --system

# 2 GB of swap: a cushion for memory peaks (Whisper and CLIP loading at once), not working memory
if [ ! -f /swapfile ]; then
  fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  echo "/swapfile none swap sw 0 0" >> /etc/fstab
  echo "vm.swappiness=10" > /etc/sysctl.d/99-swap.conf && sysctl --system
fi

mkdir -p /opt/video-rag
