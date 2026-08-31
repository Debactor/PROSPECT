#!/bin/bash
set -euo pipefail

# Usage:
#   bash scripts/prospect_eval.sh r2r
#   bash scripts/prospect_eval.sh rxr
#
# Optional env vars:
#   CHECKPOINT=checkpoints/prospect/checkpoint-XXXX
#   EVAL_SPLIT=val_unseen
#   DATA_ROOT=./data

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
MASTER_PORT=$((RANDOM % 101 + 20000))
REPO_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
BENCHMARK="${1:-r2r}"
EVAL_SPLIT="${EVAL_SPLIT:-val_unseen}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/checkpoints/prospect}"

case "${BENCHMARK}" in
  r2r)
    HABITAT_CONFIG="${REPO_ROOT}/config/vln_r2r_eval.yaml"
    OUTPUT_PATH="${REPO_ROOT}/results/r2r_${EVAL_SPLIT}"
    ;;
  rxr)
    HABITAT_CONFIG="${REPO_ROOT}/config/vln_rxr_eval.yaml"
    OUTPUT_PATH="${REPO_ROOT}/results/rxr_${EVAL_SPLIT}"
    ;;
  *)
    echo "Unknown benchmark: ${BENCHMARK}. Use r2r | rxr"
    exit 1
    ;;
esac

if [[ ! -f "${HABITAT_CONFIG}" ]]; then
  echo "Error: habitat config not found: ${HABITAT_CONFIG}"
  exit 1
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NPROC=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
else
  NPROC=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -n1)
  if [[ -z "$NPROC" || "$NPROC" -eq 0 ]]; then
    echo "Error: no GPU detected"
    exit 1
  fi
fi

mkdir -p "${OUTPUT_PATH}"

echo "Benchmark : ${BENCHMARK}"
echo "Split     : ${EVAL_SPLIT}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Config    : ${HABITAT_CONFIG}"
echo "Output    : ${OUTPUT_PATH}"

torchrun --nproc_per_node=${NPROC} --master_port="${MASTER_PORT}" \
  -m streamvln.streamvln_eval \
  --model_path "${CHECKPOINT}" \
  --habitat_config_path "${HABITAT_CONFIG}" \
  --eval_split "${EVAL_SPLIT}" \
  --output_path "${OUTPUT_PATH}" \
  2>&1 | tee "${OUTPUT_PATH}.log"
