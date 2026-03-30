#!/usr/bin/env bash
# run_array_job.sh — SLURM array job script for the scDeBussy benchmark.
#
# Each task reads one line from MANIFEST and runs run_scenario.py.
#
# DO NOT submit this script directly.  Use submit_benchmark.sh instead,
# which generates the manifest and sets the --array range automatically.
#
# Required environment variables (set by submit_benchmark.sh via --export):
#   MANIFEST     Path to the manifest file.
#                New format: "method scenario seed" per line.
#                Backward compatible with old "scenario seed" format.
#   OUTPUT_DIR   Root directory for JSON result files.
#   REPO_ROOT    Absolute path to the repository root (used for sys.path).
# Optional environment variables for method-specific overrides (base64-encoded JSON):
#   BASELINE_METHOD_PARAMS_B64    base64-encoded JSON used when METHOD=identity
#   SCDEBUSSY_METHOD_PARAMS_B64   base64-encoded JSON used when METHOD=scdebussy
#   CELLALIGN_METHOD_PARAMS_B64   base64-encoded JSON used for cellalign* methods
# Optional tuning environment variables:
#   TUNE_MODE                     one of off, auto, force (default: auto)
#   TUNE_TRIALS                   Optuna trial budget (default: 30)
#   TUNE_SEED                     Optuna random seed (default: 42)
#   METHOD_TUNE_TRIALS_MAP        optional CSV mapping, e.g. method1=3,method2=1
# (base64 encoding avoids SLURM --export comma-splitting of JSON values)
#
# Edit the SBATCH directives below to match your cluster configuration:
#   --partition   : cluster partition name          (REQUIRED — fill in)
#   --account     : project account if needed       (optional — uncomment)

#SBATCH --job-name=scdebussy_bench
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --partition=cpu        # ← EDIT THIS before submitting

set -euo pipefail

# ---- Resolve task -----------------------------------------------------------
TASK_ID=${SLURM_ARRAY_TASK_ID}
LINE=$(sed -n "$((TASK_ID + 1))p" "${MANIFEST}")

if [[ -z "${LINE}" ]]; then
    echo "[ERROR] No line at index ${TASK_ID} in ${MANIFEST}"
    exit 1
fi

set -- ${LINE}
if [[ "$#" -eq 3 ]]; then
    METHOD="$1"
    SCENARIO="$2"
    SEED="$3"
elif [[ "$#" -eq 2 ]]; then
    METHOD="scdebussy"
    SCENARIO="$1"
    SEED="$2"
else
    echo "[ERROR] Invalid manifest line format at task ${TASK_ID}: '${LINE}'"
    exit 1
fi

echo "[task ${TASK_ID}] method=${METHOD}  scenario=${SCENARIO}  seed=${SEED}"
echo "[task ${TASK_ID}] output_dir=${OUTPUT_DIR}"

TUNE_MODE="${TUNE_MODE:-auto}"
TUNE_TRIALS="${TUNE_TRIALS:-30}"
TUNE_SEED="${TUNE_SEED:-42}"
METHOD_TUNE_TRIALS_MAP="${METHOD_TUNE_TRIALS_MAP:-}"

# Optional per-method override for tuning trial budget.
if [[ -n "${METHOD_TUNE_TRIALS_MAP}" ]]; then
    IFS=',' read -r -a _mt_entries <<< "${METHOD_TUNE_TRIALS_MAP}"
    for _entry in "${_mt_entries[@]}"; do
        _entry_trimmed="${_entry//[[:space:]]/}"
        [[ -z "${_entry_trimmed}" ]] && continue
        _k="${_entry_trimmed%%=*}"
        _v="${_entry_trimmed#*=}"
        if [[ "${_k}" == "${METHOD}" ]] && [[ "${_v}" =~ ^[0-9]+$ ]] && [[ "${_v}" -gt 0 ]]; then
            TUNE_TRIALS="${_v}"
            break
        fi
    done
fi

echo "[task ${TASK_ID}] tune_mode=${TUNE_MODE}  tune_trials=${TUNE_TRIALS}  tune_seed=${TUNE_SEED}"

# ---- Activate environment (edit if using conda or a specific venv) ---------
# Example for conda:
#   source activate scdebussy
# Example for a venv at the repo root:
#   source "${REPO_ROOT}/.venv/bin/activate"

# ---- Run -------------------------------------------------------------------
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"

METHOD_PARAMS_B64=""
case "${METHOD}" in
    identity)
        METHOD_PARAMS_B64="${BASELINE_METHOD_PARAMS_B64:-}"
        ;;
    scdebussy)
        METHOD_PARAMS_B64="${SCDEBUSSY_METHOD_PARAMS_B64:-}"
        ;;
    cellalign|cellalign_fixed_reference|cellalign_pairwise|cellalign_consensus)
        METHOD_PARAMS_B64="${CELLALIGN_METHOD_PARAMS_B64:-}"
        ;;
esac

# Decode base64 → JSON if a value was provided
METHOD_PARAMS_JSON=""
if [[ -n "${METHOD_PARAMS_B64}" ]]; then
    METHOD_PARAMS_JSON=$(printf '%s' "${METHOD_PARAMS_B64}" | base64 -d)
fi

CMD=(
    python scripts/benchmark/run_scenario.py
    --method "${METHOD}"
    --scenario "${SCENARIO}"
    --seed "${SEED}"
    --output_dir "${OUTPUT_DIR}"
    --tune "${TUNE_MODE}"
    --tune_trials "${TUNE_TRIALS}"
    --tune_seed "${TUNE_SEED}"
)

if [[ -n "${METHOD_PARAMS_JSON}" ]]; then
    CMD+=(--method_params "${METHOD_PARAMS_JSON}")
fi

"${CMD[@]}"

echo "[task ${TASK_ID}] DONE"
