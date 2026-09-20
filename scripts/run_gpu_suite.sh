#!/usr/bin/env bash
set -euo pipefail

GRAPHKV_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GRAPHKV_PYTHON_BIN="${GRAPHKV_PYTHON_BIN:-python}"
GRAPHKV_HOST="${GRAPHKV_HOST:-127.0.0.1}"
GRAPHKV_PORT="${GRAPHKV_PORT:-7648}"
GRAPHKV_MODEL="${GRAPHKV_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
GRAPHKV_MODEL_REVISION="${GRAPHKV_MODEL_REVISION:-7ae557604adf67be50417f59c2c2f167def9a775}"
GRAPHKV_PROMPT_TOKENS="${GRAPHKV_PROMPT_TOKENS:-512}"
GRAPHKV_REPETITIONS="${GRAPHKV_REPETITIONS:-3}"
GRAPHKV_RESULTS_DIR="${GRAPHKV_RESULTS_DIR:-${GRAPHKV_REPO_DIR}/results}"
GRAPHKV_SERVER_PID=""

stop_server() {
    if [[ -n "${GRAPHKV_SERVER_PID}" ]] && kill -0 "${GRAPHKV_SERVER_PID}" 2>/dev/null; then
        kill "${GRAPHKV_SERVER_PID}"
        wait "${GRAPHKV_SERVER_PID}" 2>/dev/null || true
    fi
    GRAPHKV_SERVER_PID=""
}

start_clean_server() {
    local policy="$1"
    stop_server
    "${GRAPHKV_PYTHON_BIN}" -m graphkv.server \
        --host "${GRAPHKV_HOST}" \
        --port "${GRAPHKV_PORT}" \
        --max-gib 1 \
        >"${GRAPHKV_RESULTS_DIR}/server-${policy}.log" 2>&1 &
    GRAPHKV_SERVER_PID=$!

    local ready=0
    for _ in {1..50}; do
        if "${GRAPHKV_PYTHON_BIN}" -c \
            "from graphkv.store import TcpStateStoreClient; assert TcpStateStoreClient('${GRAPHKV_HOST}', ${GRAPHKV_PORT}).ping()" \
            >/dev/null 2>&1; then
            ready=1
            break
        fi
        sleep 0.1
    done
    if [[ "${ready}" -ne 1 ]]; then
        echo "GraphKV server did not become ready; see server-${policy}.log" >&2
        exit 1
    fi
}

trap stop_server EXIT INT TERM
mkdir -p "${GRAPHKV_RESULTS_DIR}"
cd "${GRAPHKV_REPO_DIR}"

"${GRAPHKV_PYTHON_BIN}" -c \
    "import torch; assert torch.cuda.is_available(), 'CUDA is not visible'; print(torch.cuda.get_device_name(0))"

for policy in reset whole_graph local_layout; do
    echo "Running ${policy} with an empty state server"
    start_clean_server "${policy}"
    "${GRAPHKV_PYTHON_BIN}" scripts/run_gpu_benchmark.py \
        --server "${GRAPHKV_HOST}:${GRAPHKV_PORT}" \
        --model "${GRAPHKV_MODEL}" \
        --model-revision "${GRAPHKV_MODEL_REVISION}" \
        --prompt-tokens "${GRAPHKV_PROMPT_TOKENS}" \
        --repetitions "${GRAPHKV_REPETITIONS}" \
        --placement round_robin \
        --policy "${policy}" \
        --output "${GRAPHKV_RESULTS_DIR}/gpu-${policy}.jsonl" \
        --quiet
    stop_server
done

"${GRAPHKV_PYTHON_BIN}" scripts/summarize_results.py \
    "${GRAPHKV_RESULTS_DIR}/gpu-reset.jsonl" \
    "${GRAPHKV_RESULTS_DIR}/gpu-whole_graph.jsonl" \
    "${GRAPHKV_RESULTS_DIR}/gpu-local_layout.jsonl"
