#!/usr/bin/env bash
# Local driver. Runs on the M1 Air to provision a Thunder Compute box,
# upload the dataset + training code, and open an SSH session.
#
# Prereqs (one-time):
#   brew tap Thunder-Compute/tnr && brew install tnr
#   tnr login --token $THUNDER_API_TOKEN   # or: tnr login (opens browser)
#
# Usage:
#   ./launch.sh provision   # create H100, upload files, print connect cmd
#   ./launch.sh upload      # re-upload code/dataset to instance 0
#   ./launch.sh connect     # SSH in
#   ./launch.sh pull        # copy adapter back to ./output/
#   ./launch.sh stop        # delete instance (STOPS BILLING)

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
DATASET="${HERE}/datasets/yuki_clean_v2.jsonl"
INSTANCE_ID="${INSTANCE_ID:-0}"
REMOTE_DIR="/home/ubuntu/yuki-finetune"

require_dataset() {
    if [[ ! -f "${DATASET}" ]]; then
        echo "dataset not found: ${DATASET}" >&2
        exit 1
    fi
}

cmd_provision() {
    require_dataset
    echo "[launch] creating H100 instance..."
    tnr create --gpu h100 --vcpus 16 --primary-disk 250 --mode production
    echo "[launch] waiting for instance to come up..."
    tnr status
    cmd_upload
    echo
    echo "[launch] next steps:"
    echo "  ./launch.sh connect"
    echo "  # on the remote box:"
    echo "  cd ${REMOTE_DIR} && bash remote_setup.sh"
}

cmd_upload() {
    require_dataset
    # tnr scp does not auto-create the parent dir, and tnr connect can't run
    # one-shot commands. So make sure the remote dir exists first:
    #   tnr connect ${INSTANCE_ID}
    #   mkdir -p ${REMOTE_DIR} && exit
    echo "[launch] uploading to instance ${INSTANCE_ID}:${REMOTE_DIR}/ ..."
    echo "[launch] (if scp fails with 'No such file', SSH in and mkdir -p ${REMOTE_DIR})"
    tnr scp "${HERE}/train.py"          "${INSTANCE_ID}:${REMOTE_DIR}/train.py"
    tnr scp "${HERE}/requirements.txt"  "${INSTANCE_ID}:${REMOTE_DIR}/requirements.txt"
    tnr scp "${HERE}/remote_setup.sh"   "${INSTANCE_ID}:${REMOTE_DIR}/remote_setup.sh"
    tnr scp "${DATASET}"                "${INSTANCE_ID}:${REMOTE_DIR}/yuki_clean_v2.jsonl"
}

cmd_connect() {
    exec tnr connect "${INSTANCE_ID}"
}

cmd_pull() {
    mkdir -p "${HERE}/output"
    echo "[launch] pulling adapter from instance ${INSTANCE_ID}..."
    tnr scp "${INSTANCE_ID}:${REMOTE_DIR}/output/yuki-qwen3-32b-lora" "${HERE}/output/"
}

cmd_stop() {
    echo "[launch] deleting instance ${INSTANCE_ID} (this stops billing)..."
    read -r -p "type DELETE to confirm: " ans
    [[ "${ans}" == "DELETE" ]] || { echo "aborted"; exit 1; }
    tnr delete "${INSTANCE_ID}"
}

case "${1:-}" in
    provision) cmd_provision ;;
    upload)    cmd_upload ;;
    connect)   cmd_connect ;;
    pull)      cmd_pull ;;
    stop)      cmd_stop ;;
    *)
        echo "usage: $0 {provision|upload|connect|pull|stop}"
        exit 1
        ;;
esac
