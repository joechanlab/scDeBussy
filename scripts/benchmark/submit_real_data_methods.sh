#!/usr/bin/env bash
# Submit one SLURM array where each task runs a different method on real data.

# Typical workflow:
# 1) Prepare staged files:
#    python scripts/benchmark/prepare_real_data_inputs.py ...
# 2) Submit array:
# conda activate /usersoftware/chanj3/scdebussy-py311/
# bash /data1/chanj3/wangm10/scDeBussy/scripts/benchmark/submit_real_data_methods.sh \
#   /scratch/chanj3/wangm10/HTAN/preprocessed/staged_paths.txt \
#   /scratch/chanj3/wangm10/HTAN/results \
#   scdebussy,identity \
#   'LUAD|SCLC-A|SCLC-N'
# conda activate /usersoftware/chanj3/Lamian
# bash /data1/chanj3/wangm10/scDeBussy/scripts/benchmark/submit_real_data_methods.sh \
#   /scratch/chanj3/wangm10/HTAN/preprocessed/staged_paths.txt \
#   /scratch/chanj3/wangm10/HTAN/results \
#   cellalign_fixed_reference,cellalign_consensus \
#   'LUAD|SCLC-A|SCLC-N' \
#   patient \
#   s_local \
#   cell_type_final2 \
#   1 \
#   off \
#   1 \
#   42 \
#   50 \
#   1 \
#   '{"rscript_bin": "/usersoftware/chanj3/Lamian/bin/Rscript"}'
# conda activate /usersoftware/chanj3/g2g_env
# bash /data1/chanj3/wangm10/scDeBussy/scripts/benchmark/submit_real_data_methods.sh \
#   /scratch/chanj3/wangm10/HTAN/preprocessed/staged_paths.txt \
#   /scratch/chanj3/wangm10/HTAN/results \
#   genes2genes_consensus,genes2genes_fixed_reference \
#   'LUAD|SCLC-A|SCLC-N' \
#   patient \
#   s_local \
#   cell_type_final2 \
#   1 \
#   off \
#   1 \
#   42

set -euo pipefail

DRY_RUN=0
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL_ARGS[@]}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

STAGED_MANIFEST="${1:-}"
OUTPUT_DIR="${2:-${REPO_ROOT}/results/real_data_compare}"
METHODS_CSV="${3:-scdebussy,identity,cellalign,genes2genes_consensus}"
CELL_TYPE_PATTERN="${4:-LUAD|SCLC-A}"
PATIENT_KEY="${5:-patient}"
PSEUDOTIME_KEY="${6:-s_local}"
CELL_TYPE_KEY="${7:-cell_type_final2}"
SEED="${8:-1}"
TUNE_MODE="${9:-auto}"
TUNE_TRIALS="${10:-30}"
TUNE_SEED="${11:-42}"
PURITY_N_NEIGHBORS="${12:-50}"
PURITY_SWEEP="${13:-1}"

if [[ -z "${STAGED_MANIFEST}" ]]; then
    echo "[ERROR] Missing required STAGED_MANIFEST argument."
    echo "Usage: bash scripts/benchmark/submit_real_data_methods.sh STAGED_MANIFEST [OUTPUT_DIR] [METHODS_CSV] [CELL_TYPE_PATTERN] [PATIENT_KEY] [PSEUDOTIME_KEY] [CELL_TYPE_KEY] [SEED] [TUNE_MODE] [TUNE_TRIALS] [TUNE_SEED] [PURITY_N_NEIGHBORS] [PURITY_SWEEP]"
    exit 1
fi

if [[ ! -f "${STAGED_MANIFEST}" ]]; then
    echo "[ERROR] STAGED_MANIFEST does not exist: ${STAGED_MANIFEST}"
    exit 1
fi

if [[ ! "${TUNE_MODE}" =~ ^(off|auto|force)$ ]]; then
    echo "[ERROR] TUNE_MODE must be one of: off, auto, force"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/logs"

MANIFEST_METHODS="${OUTPUT_DIR}/manifest_methods_$(date +%Y%m%d_%H%M%S)_$$.txt"
IFS=',' read -r -a METHODS <<< "${METHODS_CSV}"

if [[ ${#METHODS[@]} -eq 0 ]]; then
    echo "[ERROR] No methods were parsed from METHODS_CSV=${METHODS_CSV}"
    exit 1
fi

: > "${MANIFEST_METHODS}"
for m in "${METHODS[@]}"; do
    m_trimmed="${m//[[:space:]]/}"
    [[ -z "${m_trimmed}" ]] && continue
    echo "${m_trimmed}" >> "${MANIFEST_METHODS}"
done

N_METHODS=$(awk 'NF' "${MANIFEST_METHODS}" | wc -l)
if [[ "${N_METHODS}" -le 0 ]]; then
    echo "[ERROR] Method manifest is empty: ${MANIFEST_METHODS}"
    exit 1
fi

ARRAY_MAX=$((N_METHODS - 1))
ARRAY_SCRIPT="${SCRIPT_DIR}/run_real_data_methods_array.sh"

echo "[real-submit] Repo root        : ${REPO_ROOT}"
echo "[real-submit] Staged manifest  : ${STAGED_MANIFEST}"
echo "[real-submit] Output dir       : ${OUTPUT_DIR}"
echo "[real-submit] Methods manifest : ${MANIFEST_METHODS}"
echo "[real-submit] Methods count    : ${N_METHODS}"
echo "[real-submit] Array range      : 0-${ARRAY_MAX}"
echo "[real-submit] Cell pattern     : ${CELL_TYPE_PATTERN}"
echo "[real-submit] Tune             : mode=${TUNE_MODE} trials=${TUNE_TRIALS} seed=${TUNE_SEED}"
echo "[real-submit] Dry run          : ${DRY_RUN}"

SBATCH_CMD=(
    sbatch
    --array "0-${ARRAY_MAX}"
    --output "${OUTPUT_DIR}/logs/real_%A_%a.out"
    --error "${OUTPUT_DIR}/logs/real_%A_%a.err"
    --export "ALL,REPO_ROOT=${REPO_ROOT},MANIFEST_METHODS=${MANIFEST_METHODS},STAGED_MANIFEST=${STAGED_MANIFEST},OUTPUT_DIR=${OUTPUT_DIR},PATIENT_KEY=${PATIENT_KEY},PSEUDOTIME_KEY=${PSEUDOTIME_KEY},CELL_TYPE_KEY=${CELL_TYPE_KEY},CELL_TYPE_PATTERN=${CELL_TYPE_PATTERN},SEED=${SEED},TUNE_MODE=${TUNE_MODE},TUNE_TRIALS=${TUNE_TRIALS},TUNE_SEED=${TUNE_SEED},PURITY_N_NEIGHBORS=${PURITY_N_NEIGHBORS},PURITY_SWEEP=${PURITY_SWEEP}"
    "${ARRAY_SCRIPT}"
)

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[real-submit] Dry run command:"
    printf ' %q' "${SBATCH_CMD[@]}"
    echo
    exit 0
fi

"${SBATCH_CMD[@]}"
