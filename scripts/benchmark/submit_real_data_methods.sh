#!/usr/bin/env bash
# Submit one SLURM array where each task runs a different method on real data.
#
# All arguments are named flags so they can be supplied in any order and
# omitted when the default is acceptable.
#
# Required:
#   --staged-manifest PATH
#
# Typical workflow:
# 1) Prepare staged files:
#    python scripts/benchmark/prepare_real_data_inputs.py ...
# 2) Submit array:
# conda activate /usersoftware/chanj3/scdebussy-py311/
# bash /data1/chanj3/wangm10/scDeBussy/scripts/benchmark/submit_real_data_methods.sh \
#   --staged-manifest /scratch/chanj3/wangm10/HTAN_transformed_combined/preprocessed/staged_paths.txt \
#   --output-dir /scratch/chanj3/wangm10/HTAN_transformed_combined/results \
#   --methods scdebussy,identity \
#   --use-soft-em \
#   --cell-type-pattern 'LUAD|SCLC-A|SCLC-N' \
#   --patient-key patient \
#   --pseudotime-key s_local \
#   --cell-type-key cell_type_final2
# conda activate /usersoftware/chanj3/Lamian
# bash /data1/chanj3/wangm10/scDeBussy/scripts/benchmark/submit_real_data_methods.sh \
#   --staged-manifest /scratch/chanj3/wangm10/HTAN_transformed_combined/preprocessed/staged_paths.txt \
#   --output-dir /scratch/chanj3/wangm10/HTAN_transformed_combined/results \
#   --methods cellalign_fixed_reference,cellalign_consensus \
#   --cell-type-pattern 'LUAD|SCLC-A|SCLC-N' \
#   --patient-key patient \
#   --pseudotime-key s_local \
#   --cell-type-key cell_type_final2 \
#   --purity-n-neighbors 50
# conda activate /usersoftware/chanj3/g2g_env
# bash /data1/chanj3/wangm10/scDeBussy/scripts/benchmark/submit_real_data_methods.sh \
#   --staged-manifest /scratch/chanj3/wangm10/HTAN_transformed_combined/preprocessed/staged_paths.txt \
#   --output-dir /scratch/chanj3/wangm10/HTAN_transformed_combined/results \
#   --methods genes2genes_consensus,genes2genes_fixed_reference \
#   --cell-type-pattern 'LUAD|SCLC-A|SCLC-N'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
STAGED_MANIFEST=""
OUTPUT_DIR="${REPO_ROOT}/results/real_data_compare"
METHODS_CSV="scdebussy,identity,cellalign,genes2genes_consensus"
CELL_TYPE_PATTERN="LUAD|SCLC-A"
PATIENT_KEY="patient"
PSEUDOTIME_KEY="s_local"
CELL_TYPE_KEY="cell_type_final2"
SEED="1"
TUNE_MODE="auto"
TUNE_TRIALS="30"
TUNE_SEED="42"
PURITY_N_NEIGHBORS="50"
PURITY_SWEEP="1"
SAVE_ADATA="0"
GROUP_KEY=""
GENE_FILTER="0"
INFORMATIVE_GENE_FILTER="0"
HVG_TOP_PER_PATIENT="3000"
HVG_MIN_PATIENTS=""
HVG_LAYER="counts"
USE_SOFT_EM="0"
DRY_RUN=0

# ---------------------------------------------------------------------------
# Argument parsing — named flags, any order
# ---------------------------------------------------------------------------
usage() {
    cat <<EOF
Usage: bash submit_real_data_methods.sh --staged-manifest PATH [OPTIONS]

Required:
  --staged-manifest PATH      Text file with one staged .h5ad path per line

Options:
  --output-dir DIR            Result root directory (default: ${REPO_ROOT}/results/real_data_compare)
  --methods CSV               Comma-separated method list (default: scdebussy,identity,cellalign,genes2genes_consensus)
  --cell-type-pattern REGEX   Regex filter applied to cell-type column (default: LUAD|SCLC-A)
  --patient-key KEY           obs column for patient labels (default: patient)
  --pseudotime-key KEY        obs column for pseudotime (default: s_local)
  --cell-type-key KEY         obs column for cell types (default: cell_type_final2)
  --seed INT                  Random seed (default: 1)
  --tune-mode MODE            off|auto|force (default: auto)
  --tune-trials INT           Optuna trial count (default: 30)
  --tune-seed INT             Optuna sampler seed (default: 42)
  --purity-n-neighbors INT    k for cross-batch KNN purity (default: 50)
  --purity-sweep              Enable purity sweep (default: enabled)
  --no-purity-sweep           Disable purity sweep
  --save-adata                Save aligned AnnData alongside JSON result
  --group-key KEY             obs column for patient group/condition labels
  --gene-filter               Apply recurrent HVG filter before alignment
  --informative-gene-filter   Apply informative-gene filter before alignment
  --hvg-top-per-patient INT   Top N genes per patient for HVG filter (default: 3000)
  --hvg-min-patients INT      Min patients a gene must be HVG in (default: ceil(n/2))
  --hvg-layer LAYER           AnnData layer for HVG computation (default: counts)
  --use-soft-em               Use soft-EM (forward-backward) for scDeBussy alignment
  --dry-run                   Print sbatch command without submitting
  -h, --help                  Show this message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --staged-manifest)    STAGED_MANIFEST="${2:-}";        shift 2 ;;
        --output-dir)         OUTPUT_DIR="${2:-}";             shift 2 ;;
        --methods)            METHODS_CSV="${2:-}";            shift 2 ;;
        --cell-type-pattern)  CELL_TYPE_PATTERN="${2:-}";      shift 2 ;;
        --patient-key)        PATIENT_KEY="${2:-}";            shift 2 ;;
        --pseudotime-key)     PSEUDOTIME_KEY="${2:-}";         shift 2 ;;
        --cell-type-key)      CELL_TYPE_KEY="${2:-}";          shift 2 ;;
        --seed)               SEED="${2:-}";                   shift 2 ;;
        --tune-mode)          TUNE_MODE="${2:-}";              shift 2 ;;
        --tune-trials)        TUNE_TRIALS="${2:-}";            shift 2 ;;
        --tune-seed)          TUNE_SEED="${2:-}";              shift 2 ;;
        --purity-n-neighbors) PURITY_N_NEIGHBORS="${2:-}";     shift 2 ;;
        --purity-sweep)       PURITY_SWEEP="1";                shift ;;
        --no-purity-sweep)    PURITY_SWEEP="0";                shift ;;
        --save-adata)         SAVE_ADATA="1";                  shift ;;
        --group-key)          GROUP_KEY="${2:-}";              shift 2 ;;
        --gene-filter)        GENE_FILTER="1";                 shift ;;
        --informative-gene-filter) INFORMATIVE_GENE_FILTER="1"; shift ;;
        --hvg-top-per-patient) HVG_TOP_PER_PATIENT="${2:-}";  shift 2 ;;
        --hvg-min-patients)   HVG_MIN_PATIENTS="${2:-}";       shift 2 ;;
        --hvg-layer)          HVG_LAYER="${2:-}";              shift 2 ;;
        --use-soft-em)        USE_SOFT_EM="1";                 shift ;;
        --dry-run)            DRY_RUN=1;                       shift ;;
        -h|--help)            usage; exit 0 ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [[ -z "${STAGED_MANIFEST}" ]]; then
    echo "[ERROR] --staged-manifest is required."
    usage
    exit 1
fi

if [[ ! -f "${STAGED_MANIFEST}" ]]; then
    echo "[ERROR] STAGED_MANIFEST does not exist: ${STAGED_MANIFEST}"
    exit 1
fi

if [[ ! "${TUNE_MODE}" =~ ^(off|auto|force)$ ]]; then
    echo "[ERROR] --tune-mode must be one of: off, auto, force"
    exit 1
fi

# ---------------------------------------------------------------------------
# Build per-method manifest
# ---------------------------------------------------------------------------
mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/logs"

MANIFEST_METHODS="${OUTPUT_DIR}/manifest_methods_$(date +%Y%m%d_%H%M%S)_$$.txt"
IFS=',' read -r -a METHODS <<< "${METHODS_CSV}"

if [[ ${#METHODS[@]} -eq 0 ]]; then
    echo "[ERROR] No methods were parsed from --methods=${METHODS_CSV}"
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
echo "[real-submit] Save adata       : ${SAVE_ADATA}"
echo "[real-submit] Group key        : ${GROUP_KEY:-<none>}"
echo "[real-submit] Gene filter      : recurrent_hvg=${GENE_FILTER} informative=${INFORMATIVE_GENE_FILTER} top_per_patient=${HVG_TOP_PER_PATIENT} min_patients=${HVG_MIN_PATIENTS:-default} layer=${HVG_LAYER}"
echo "[real-submit] Use soft-EM      : ${USE_SOFT_EM}"
echo "[real-submit] Dry run          : ${DRY_RUN}"

SBATCH_CMD=(
    sbatch
    --array "0-${ARRAY_MAX}"
    --output "${OUTPUT_DIR}/logs/real_%A_%a.out"
    --error "${OUTPUT_DIR}/logs/real_%A_%a.err"
    --export "ALL,REPO_ROOT=${REPO_ROOT},MANIFEST_METHODS=${MANIFEST_METHODS},STAGED_MANIFEST=${STAGED_MANIFEST},OUTPUT_DIR=${OUTPUT_DIR},PATIENT_KEY=${PATIENT_KEY},PSEUDOTIME_KEY=${PSEUDOTIME_KEY},CELL_TYPE_KEY=${CELL_TYPE_KEY},CELL_TYPE_PATTERN=${CELL_TYPE_PATTERN},SEED=${SEED},TUNE_MODE=${TUNE_MODE},TUNE_TRIALS=${TUNE_TRIALS},TUNE_SEED=${TUNE_SEED},PURITY_N_NEIGHBORS=${PURITY_N_NEIGHBORS},PURITY_SWEEP=${PURITY_SWEEP},SAVE_ADATA=${SAVE_ADATA},GROUP_KEY=${GROUP_KEY},GENE_FILTER=${GENE_FILTER},INFORMATIVE_GENE_FILTER=${INFORMATIVE_GENE_FILTER},HVG_TOP_PER_PATIENT=${HVG_TOP_PER_PATIENT},HVG_MIN_PATIENTS=${HVG_MIN_PATIENTS},HVG_LAYER=${HVG_LAYER},USE_SOFT_EM=${USE_SOFT_EM}"
    "${ARRAY_SCRIPT}"
)

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[real-submit] Dry run command:"
    printf ' %q' "${SBATCH_CMD[@]}"
    echo
    exit 0
fi

"${SBATCH_CMD[@]}"
