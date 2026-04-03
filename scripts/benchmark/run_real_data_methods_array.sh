#!/usr/bin/env bash
# SLURM array task script: one method per task for real-data benchmarking.
#
# Required env vars:
#   MANIFEST_METHODS   text file with one method per line
#   STAGED_MANIFEST    text file with one staged h5ad path per line
#   OUTPUT_DIR         result root
#   REPO_ROOT          repository root
#
# Optional env vars:
#   PATIENT_KEY        default: patient
#   PSEUDOTIME_KEY     default: s_local
#   CELL_TYPE_KEY      default: cell_type_final2
#   CELL_TYPE_PATTERN  default: LUAD|SCLC-A
#   SEED               default: 1
#   TUNE_MODE          default: auto
#   TUNE_TRIALS        default: 30
#   TUNE_SEED          default: 42
#   PURITY_N_NEIGHBORS default: 50
#   PURITY_SWEEP       default: 1 (set 0 to disable)

#SBATCH --job-name=scdebussy_real
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --partition=cpu

set -euo pipefail

TASK_ID=${SLURM_ARRAY_TASK_ID}
METHOD=$(sed -n "$((TASK_ID + 1))p" "${MANIFEST_METHODS}")

if [[ -z "${METHOD}" ]]; then
    echo "[ERROR] No method found for task ${TASK_ID} in ${MANIFEST_METHODS}"
    exit 1
fi

if [[ ! -f "${STAGED_MANIFEST}" ]]; then
    echo "[ERROR] STAGED_MANIFEST not found: ${STAGED_MANIFEST}"
    exit 1
fi

mapfile -t ADATA_PATHS < <(awk 'NF' "${STAGED_MANIFEST}")
if [[ ${#ADATA_PATHS[@]} -eq 0 ]]; then
    echo "[ERROR] No staged paths found in ${STAGED_MANIFEST}"
    exit 1
fi

PATIENT_KEY="${PATIENT_KEY:-patient}"
PSEUDOTIME_KEY="${PSEUDOTIME_KEY:-s_local}"
CELL_TYPE_KEY="${CELL_TYPE_KEY:-cell_type_final2}"
CELL_TYPE_PATTERN="${CELL_TYPE_PATTERN:-LUAD|SCLC-A}"
SEED="${SEED:-1}"
TUNE_MODE="${TUNE_MODE:-auto}"
TUNE_TRIALS="${TUNE_TRIALS:-30}"
TUNE_SEED="${TUNE_SEED:-42}"
PURITY_N_NEIGHBORS="${PURITY_N_NEIGHBORS:-50}"
PURITY_SWEEP="${PURITY_SWEEP:-1}"
METHOD_PARAMS="${METHOD_PARAMS:-}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"

echo "[task ${TASK_ID}] method=${METHOD}"
echo "[task ${TASK_ID}] n_inputs=${#ADATA_PATHS[@]}"

echo "[task ${TASK_ID}] running run_real_data.py"
CMD=(
    python scripts/benchmark/run_real_data.py
    --adata "${ADATA_PATHS[@]}"
    --output-dir "${OUTPUT_DIR}"
    --method "${METHOD}"
    --seed "${SEED}"
    --patient-key "${PATIENT_KEY}"
    --pseudotime-key "${PSEUDOTIME_KEY}"
    --cell-type-key "${CELL_TYPE_KEY}"
    --cell-type-pattern "${CELL_TYPE_PATTERN}"
    --tune "${TUNE_MODE}"
    --tune-trials "${TUNE_TRIALS}"
    --tune-seed "${TUNE_SEED}"
    --purity-n-neighbors "${PURITY_N_NEIGHBORS}"
)

if [[ "${PURITY_SWEEP}" == "1" ]]; then
    CMD+=(--purity-sweep)
fi

if [[ -n "${METHOD_PARAMS}" ]]; then
    CMD+=(--method-params "${METHOD_PARAMS}")
fi

"${CMD[@]}"

echo "[task ${TASK_ID}] DONE"
