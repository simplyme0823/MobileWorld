#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTER_IMAGE="${1:-mobile_world:midscene-dind-fix-clock}"

docker build \
  --platform linux/amd64 \
  --file "$ROOT_DIR/docker/Dockerfile.update" \
  --tag "$OUTER_IMAGE" \
  "$ROOT_DIR"

echo "Built $OUTER_IMAGE"
