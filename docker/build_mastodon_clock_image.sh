#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INNER_IMAGE="mobileworld/mastodon:v4.3.7-faketime"
INNER_TAR="$ROOT_DIR/docker/images/mastodon-v4.3.7-faketime.tar"
OUTER_IMAGE="${1:-mobile_world:midscene-dind-fix-clock}"

docker buildx build \
  --platform linux/amd64 \
  --load \
  --file "$ROOT_DIR/docker/mastodon-docker/Dockerfile.faketime" \
  --tag "$INNER_IMAGE" \
  "$ROOT_DIR/docker/mastodon-docker"

mkdir -p "$(dirname "$INNER_TAR")"
docker save --output "$INNER_TAR" "$INNER_IMAGE"

docker build \
  --platform linux/amd64 \
  --file "$ROOT_DIR/docker/Dockerfile.update" \
  --tag "$OUTER_IMAGE" \
  "$ROOT_DIR"

echo "Built $OUTER_IMAGE"
