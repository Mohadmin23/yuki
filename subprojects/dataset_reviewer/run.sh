#!/usr/bin/env bash
set -euo pipefail

REVIEWER_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$REVIEWER_DIR/../.." && pwd)"

cd "$PROJECT_DIR"
exec uv run granian \
  --interface asgi \
  --host 127.0.0.1 \
  --port "${YUKI_REVIEW_PORT:-7870}" \
  subprojects.dataset_reviewer.app:app
