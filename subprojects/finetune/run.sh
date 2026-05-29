#!/usr/bin/env bash
# Convenience wrapper that runs ON the Thunder Compute box.
#
# It activates the .venv for you (so you never hit "No module named unsloth"
# from forgetting to `source .venv/bin/activate`) and launches train or chat.
#
# Usage (on the remote box, after bash remote_setup.sh):
#   bash run.sh train [--epochs N ...]   # fine-tune on yuki_clean_v4.jsonl
#   bash run.sh chat  [--base ...]       # test the adapter in the TUI

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "${HERE}"

if [[ ! -d .venv ]]; then
    echo "no .venv here -- run 'bash remote_setup.sh' first" >&2
    exit 1
fi
# shellcheck source=/dev/null
source .venv/bin/activate

case "${1:-chat}" in
    train)
        shift || true
        exec python train.py \
            --dataset "${HERE}/yuki_clean_v4.jsonl" \
            --output "${HERE}/output" \
            "$@"
        ;;
    chat)
        shift || true
        exec python chat.py "$@"
        ;;
    *)
        echo "usage: bash run.sh {train|chat} [args...]" >&2
        exit 1
        ;;
esac
