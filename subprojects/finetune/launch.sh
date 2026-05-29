#!/usr/bin/env bash
# Local driver. Runs on the M1 Air to provision a Thunder Compute box,
# upload the dataset + training code, and open an SSH session.
#
# Prereqs (one-time):
#   brew tap Thunder-Compute/tnr && brew install tnr
#   tnr login --token $THUNDER_API_TOKEN   # or: tnr login (opens browser)
#
# Usage:
#   ./launch.sh provision      # create A100, upload files, print connect cmd
#   ./launch.sh upload         # re-upload code/dataset to instance 0
#   ./launch.sh push-adapter   # upload local adapter to test without retraining
#   ./launch.sh connect        # SSH in
#   ./launch.sh pull           # copy adapter back to ./output/
#   ./launch.sh stop           # delete instance (STOPS BILLING)

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
DATASET="${HERE}/datasets/yuki_clean_v4.jsonl"
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
    echo "[launch] creating A100 instance..."
    tnr create --gpu a100 --vcpus 16 --primary-disk 250 --mode production
    echo "[launch] waiting for instance to come up..."
    tnr status
    cmd_upload
    echo
    echo "[launch] next steps:"
    echo "  ./launch.sh connect"
    echo "  # on the remote box:"
    echo "  cd ${REMOTE_DIR} && bash remote_setup.sh"
    echo "  bash run.sh train --epochs 3     # or: bash run.sh chat"
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
    tnr scp "${HERE}/chat.py"           "${INSTANCE_ID}:${REMOTE_DIR}/chat.py"
    tnr scp "${HERE}/run.sh"            "${INSTANCE_ID}:${REMOTE_DIR}/run.sh"
    tnr scp "${HERE}/requirements.txt"  "${INSTANCE_ID}:${REMOTE_DIR}/requirements.txt"
    tnr scp "${HERE}/remote_setup.sh"   "${INSTANCE_ID}:${REMOTE_DIR}/remote_setup.sh"
    tnr scp "${DATASET}"                "${INSTANCE_ID}:${REMOTE_DIR}/yuki_clean_v4.jsonl"
}

cmd_connect() {
    exec tnr connect "${INSTANCE_ID}"
}

cmd_pull() {
    mkdir -p "${HERE}/output"
    echo "[launch] pulling adapter from instance ${INSTANCE_ID}..."
    tnr scp "${INSTANCE_ID}:${REMOTE_DIR}/output/yuki-qwen3-32b-lora" "${HERE}/output/"
}

# Upload an already-trained adapter to test it without retraining (e.g. a box
# you spun up just to run `run.sh chat`). The remote output dir must exist:
#   ./launch.sh connect ; mkdir -p ${REMOTE_DIR}/output ; exit
cmd_push_adapter() {
    local adapter="${HERE}/output/yuki-qwen3-32b-lora"
    if [[ ! -d "${adapter}" ]]; then
        echo "no local adapter at ${adapter} -- train + pull first" >&2
        exit 1
    fi
    echo "[launch] uploading adapter to instance ${INSTANCE_ID} (for testing)..."
    echo "[launch] (if this fails, SSH in and: mkdir -p ${REMOTE_DIR}/output)"
    tnr scp "${adapter}" "${INSTANCE_ID}:${REMOTE_DIR}/output/"
}

cmd_stop() {
    echo "[launch] deleting instance ${INSTANCE_ID} (this stops billing)..."
    read -r -p "type DELETE to confirm: " ans
    [[ "${ans}" == "DELETE" ]] || { echo "aborted"; exit 1; }
    tnr delete "${INSTANCE_ID}"
}

case "${1:-}" in
    provision)     cmd_provision ;;
    upload)        cmd_upload ;;
    push-adapter)  cmd_push_adapter ;;
    connect)       cmd_connect ;;
    pull)          cmd_pull ;;
    stop)          cmd_stop ;;
    *)
        echo "usage: $0 {provision|upload|push-adapter|connect|pull|stop}"
        echo
        echo "  provision     create box + upload code/dataset"
        echo "  upload        re-upload code/dataset to instance ${INSTANCE_ID}"
        echo "  push-adapter  upload local adapter to test it (no retrain)"
        echo "  connect       SSH in"
        echo "  pull          copy trained adapter back to ./output/"
        echo "  stop          delete instance (STOPS BILLING)"
        exit 1
        ;;
esac
