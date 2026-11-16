
#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/app/data/CODE-file/fresco_code}"
CONFIG_PATH="${CONFIG_PATH:-configs/nersemble/460/full.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/checkpoints/fresco_460_100ep}"
RUN_NAME="${RUN_NAME:-subject460_100ep}"
GPU_ID="${GPU_ID:-0}"
NUM_WORKERS="${NUM_WORKERS:-2}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-2000}"
RESUME="${RESUME:-}"
SESSION_NAME="${SESSION_NAME:-fresco460_100ep}"
MODE="${1:---run}"

if command -v python >/dev/null 2>&1; then
    HOST_PYTHON=python
elif command -v python3 >/dev/null 2>&1; then
    HOST_PYTHON=python3
else
    echo "ERROR: neither python nor python3 is available." >&2
    exit 2
fi

if ! [[ "${NUM_WORKERS}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: NUM_WORKERS must be a non-negative integer." >&2
    exit 2
fi

if [[ "${MODE}" == "--screen" ]]; then
    mkdir -p "${OUTPUT_ROOT}/${RUN_NAME}"
    exec screen -dmS "${SESSION_NAME}" bash "${BASH_SOURCE[0]}" --run
fi

if [[ "${MODE}" != "--run" && "${MODE}" != "--dry-run" ]]; then
    echo "Usage: $0 [--run|--screen|--dry-run]" >&2
    exit 2
fi

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUTPUT_ROOT}/${RUN_NAME}"

if [[ -n "${RESUME}" && ! -f "${RESUME}" ]]; then
    echo "ERROR: RESUME checkpoint does not exist: ${RESUME}" >&2
    exit 2
fi

AUTO_CHECKPOINT="${OUTPUT_ROOT}/${RUN_NAME}/checkpoint_latest.pth"

if [[ -z "${RESUME}" && -f "${AUTO_CHECKPOINT}" ]]; then
    RESUME="${AUTO_CHECKPOINT}"
    echo "Auto-resume checkpoint detected: ${RESUME}"
fi

EXTRA_CONFIG="$(
"${HOST_PYTHON}" - "${RESUME}" "${CHECKPOINT_INTERVAL}" "${NUM_WORKERS}" <<'PY'
import json
import sys

resume = sys.argv[1] or None
checkpoint_interval = int(sys.argv[2])
num_workers = int(sys.argv[3])

print(json.dumps({
    "training.gpus": "0",
    "training.epochs": 100,
    "training.checkpoint_interval": checkpoint_interval,
    "training.pretrained_checkpoint_path": resume,
    "training.resume_keep_configured_lrs": False,
    "training.require_resume_state": True,

    # DataLoader workers
    "data.num_workers": num_workers,

    # FReSCO curriculum
    "exp.fresco_schedule.enable": True,
    "exp.fresco_schedule.name": "subject460_100ep",

    # Initial stage: Epoch 0-9
    "exp.lf_img.enable": True,
    "exp.lf_img.lambda": 0.25,

    "exp.lf_uv.enable": False,
    "exp.lf_uv.loss.enable": False,
    "exp.lf_uv.loss.weight": 0.0,
    "exp.lf_uv.every": 1,
    "exp.lf_uv.loss.scale_by_every": False,

    "exp.lf_uv.seam.enable": False,
    "exp.lf_uv.seam.weight_pair": 0.0,

    "exp.hf_img.enable": False,
    "exp.hf_img.lambda": 0.0,

    # HF internal parameters
    "exp.hf_img.kernel1": 3,
    "exp.hf_img.sigma1": 0.8,
    "exp.hf_img.kernel2": 5,
    "exp.hf_img.sigma2": 1.2,
    "exp.hf_img.lambda_h1": 1.0,
    "exp.hf_img.lambda_h2": 1.0,
}))
PY
)"

echo "Project: ${PROJECT_DIR}"
echo "Output: ${OUTPUT_ROOT}/${RUN_NAME}"
echo "Physical GPU: ${GPU_ID} (visible cuda:0)"
echo "DataLoader workers: ${NUM_WORKERS}"
echo "Resume: ${RESUME:-fresh training}"
echo "Checkpoint interval: ${CHECKPOINT_INTERVAL} steps"

"${HOST_PYTHON}" tools/print_fresco_460_schedule.py

PYTHON_CMD=(
    python train.py
    --config_path "${CONFIG_PATH}"
    --workspace "${OUTPUT_ROOT}"
    --version "${RUN_NAME}"
    --extra_config "${EXTRA_CONFIG}"
)

if [[ "${MODE}" == "--run" && "${CONDA_DEFAULT_ENV:-}" != "mega" ]]; then
    CONDA_BIN="${CONDA_BIN:-conda}"
    if ! command -v "${CONDA_BIN}" >/dev/null 2>&1; then
        echo "ERROR: conda is unavailable; set CONDA_BIN or activate the mega environment." >&2
        exit 2
    fi
    CMD=(
        "${CONDA_BIN}" run --no-capture-output -n mega
        "${PYTHON_CMD[@]}"
    )
else
    CMD=("${PYTHON_CMD[@]}")
fi

printf 'CUDA_VISIBLE_DEVICES=%q ' "${GPU_ID}"
printf '%q ' "${CMD[@]}"
printf '\n'

if [[ "${MODE}" == "--dry-run" ]]; then
    exit 0
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

"${CMD[@]}" 2>&1 | tee -a "${OUTPUT_ROOT}/${RUN_NAME}/launcher.log"
