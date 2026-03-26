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
)

if [[ -n "${METHOD_PARAMS_JSON}" ]]; then
    CMD+=(--method_params "${METHOD_PARAMS_JSON}")
fi

"${CMD[@]}"

echo "[task ${TASK_ID}] DONE"
