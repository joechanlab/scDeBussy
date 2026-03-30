#!/usr/bin/env bash
# submit_compare_methods.sh — submit a benchmark array to compare:
#   1) identity baseline
#   2) scdebussy
#   3) cellalign_fixed_reference (reference_policy=medoid)
#   4) cellalign_consensus
#   5) genes2genes_fixed_reference (reference_policy=medoid)
#   6) genes2genes_consensus
#
# Usage:
#   bash scripts/benchmark/submit_compare_methods.sh \
#       [--dry-run] [--skip-existing] [--methods m1,m2,...] [--method-tune-trials m1=n1,m2=n2,...] \
#       [OUTPUT_DIR] [RSCRIPT_BIN] [SCENARIOS_CSV] [TUNE_MODE] [TUNE_TRIALS] [TUNE_SEED]
#
# Arguments:
#   OUTPUT_DIR     Root directory for outputs.
#                  Default: <repo>/results/benchmark_compare
#   RSCRIPT_BIN    Rscript executable with cellAlign installed.
#                  Default: /usersoftware/chanj3/Lamian/bin/Rscript
#   SCENARIOS_CSV  Optional comma-separated scenario names.
#                  Default: all scenarios in scripts/benchmark/scenarios.py
#   TUNE_MODE      Tuning mode passed to run_scenario.py.
#                  Choices: off, auto, force. Default: auto
#   TUNE_TRIALS    Optuna trial budget for tuned methods. Default: 30
#   TUNE_SEED      Optuna random seed for tuning. Default: 42
# Environment override:
#   ALLOW_DUPLICATE_SUBMIT=1 to bypass duplicate-submit protection.
# Flags:
#   --dry-run       Build and validate manifest, print submission plan, do not call sbatch.
#   --skip-existing Skip jobs whose output JSON already exists under OUTPUT_DIR.
#   --methods       Comma-separated method list (default: built-in compare set).
#   --method-tune-trials
#                   Comma-separated per-method Optuna trial overrides.
#                   Example: genes2genes_fixed_reference=1,genes2genes_consensus=1
#
# Example:
# bash /data1/chanj3/wangm10/scDeBussy/scripts/benchmark/submit_compare_methods.sh \
#   --methods genes2genes_fixed_reference,genes2genes_consensus \
#   --skip-existing \
#   /scratch/chanj3/wangm10/compare_run_tuning \
#   /usersoftware/chanj3/Lamian/bin/Rscript \
#   all off 1 42
#   all auto 30 42
# To aggregate compare-only summary after completion:
#   sbatch --dependency=afterok:16231968 --wrap="python /data1/chanj3/wangm10/scDeBussy/scripts/benchmark/aggregate_results.py --results_dir /scratch/chanj3/wangm10/compare_run_tuning --output /scratch/chanj3/wangm10/compare_run_tuning/summary_compare.csv --methods identity scdebussy cellalign_fixed_reference cellalign_consensus genes2genes_fixed_reference genes2genes_consensus --parquet"

# To compare tuned runs only after completion:
#   python /data1/chanj3/wangm10/scDeBussy/scripts/benchmark/compare_methods.py --results_dir /scratch/chanj3/wangm10/compare_run_tuning --output /scratch/chanj3/wangm10/compare_run_tuning/comparison_summary_tuned.csv --methods scdebussy cellalign_fixed_reference cellalign_consensus genes2genes_fixed_reference genes2genes_consensus --tuning all --parquet

set -euo pipefail

DRY_RUN=0
SKIP_EXISTING=0
METHODS_OVERRIDE_CSV=""
METHOD_TUNE_TRIALS_MAP=""
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --skip-existing)
            SKIP_EXISTING=1
            shift
            ;;
        --methods)
            shift
            if [[ $# -eq 0 ]]; then
                echo "[ERROR] --methods requires a comma-separated value."
                exit 1
            fi
            METHODS_OVERRIDE_CSV="$1"
            shift
            ;;
        --methods=*)
            METHODS_OVERRIDE_CSV="${1#*=}"
            shift
            ;;
        --method-tune-trials)
            shift
            if [[ $# -eq 0 ]]; then
                echo "[ERROR] --method-tune-trials requires a comma-separated mapping value."
                exit 1
            fi
            METHOD_TUNE_TRIALS_MAP="$1"
            shift
            ;;
        --method-tune-trials=*)
            METHOD_TUNE_TRIALS_MAP="${1#*=}"
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

OUTPUT_DIR="${1:-${REPO_ROOT}/results/benchmark_compare}"
RSCRIPT_BIN="${2:-/usersoftware/chanj3/Lamian/bin/Rscript}"
SCENARIOS_CSV="${3:-all}"
TUNE_MODE="${4:-auto}"
TUNE_TRIALS="${5:-30}"
TUNE_SEED="${6:-42}"

LOG_DIR="${OUTPUT_DIR}/logs"
MANIFEST_BASENAME="manifest_compare_$(date +%Y%m%d_%H%M%S)_$$.txt"
MANIFEST="${OUTPUT_DIR}/${MANIFEST_BASENAME}"
ARRAY_SCRIPT="${SCRIPT_DIR}/run_array_job.sh"

METHODS=(
    identity
    scdebussy
    cellalign_fixed_reference
    cellalign_consensus
    genes2genes_fixed_reference
    genes2genes_consensus
)
if [[ -n "${METHODS_OVERRIDE_CSV}" ]]; then
    IFS=',' read -r -a METHODS <<< "${METHODS_OVERRIDE_CSV}"
fi
METHODS_CSV="$(IFS=,; echo "${METHODS[*]}")"

echo "[compare-submit] Repo root      : ${REPO_ROOT}"
echo "[compare-submit] Output dir     : ${OUTPUT_DIR}"
echo "[compare-submit] Methods        : ${METHODS_CSV}"
echo "[compare-submit] Rscript        : ${RSCRIPT_BIN}"
echo "[compare-submit] Scenarios      : ${SCENARIOS_CSV}"
echo "[compare-submit] Tune mode      : ${TUNE_MODE}"
echo "[compare-submit] Tune trials    : ${TUNE_TRIALS}"
echo "[compare-submit] Tune seed      : ${TUNE_SEED}"
echo "[compare-submit] Tune overrides : ${METHOD_TUNE_TRIALS_MAP:-<none>}"
echo "[compare-submit] Skip existing  : ${SKIP_EXISTING}"
echo "[compare-submit] Dry run        : ${DRY_RUN}"
echo "[compare-submit] Manifest       : ${MANIFEST}"

if [[ -z "${METHODS_CSV//,/}" ]]; then
    echo "[ERROR] No methods provided."
    exit 1
fi

if [[ "${DRY_RUN}" -ne 1 ]] && [[ ! -x "${RSCRIPT_BIN}" ]]; then
    echo "[ERROR] RSCRIPT_BIN is not executable: ${RSCRIPT_BIN}"
    exit 1
fi

if [[ ! "${TUNE_MODE}" =~ ^(off|auto|force)$ ]]; then
    echo "[ERROR] TUNE_MODE must be one of: off, auto, force"
    exit 1
fi

if ! [[ "${TUNE_TRIALS}" =~ ^[0-9]+$ ]] || [[ "${TUNE_TRIALS}" -le 0 ]]; then
    echo "[ERROR] TUNE_TRIALS must be a positive integer"
    exit 1
fi

if ! [[ "${TUNE_SEED}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] TUNE_SEED must be a non-negative integer"
    exit 1
fi

if [[ -n "${METHOD_TUNE_TRIALS_MAP}" ]]; then
    IFS=',' read -r -a _mt_entries <<< "${METHOD_TUNE_TRIALS_MAP}"
    for _entry in "${_mt_entries[@]}"; do
        _entry_trimmed="${_entry//[[:space:]]/}"
        if [[ -z "${_entry_trimmed}" ]]; then
            continue
        fi
        if [[ "${_entry_trimmed}" != *=* ]]; then
            echo "[ERROR] Invalid --method-tune-trials entry: ${_entry_trimmed}"
            echo "[HINT] Expected format: method=integer,method2=integer"
            exit 1
        fi
        _val="${_entry_trimmed#*=}"
        if ! [[ "${_val}" =~ ^[0-9]+$ ]] || [[ "${_val}" -le 0 ]]; then
            echo "[ERROR] Invalid trials value in --method-tune-trials entry: ${_entry_trimmed}"
            exit 1
        fi
    done
fi

# Protect against accidental double-submit into the same OUTPUT_DIR.
if [[ "${ALLOW_DUPLICATE_SUBMIT:-0}" != "1" ]] && command -v squeue >/dev/null 2>&1 && command -v scontrol >/dev/null 2>&1; then
    mapfile -t _active_jobids < <(squeue -h -u "${USER}" -n scdebussy_bench -o "%A" | awk 'NF')
    _matching_jobids=()
    for _jid in "${_active_jobids[@]}"; do
        if scontrol show job "${_jid}" 2>/dev/null | grep -Fq "StdOut=${LOG_DIR}/compare_%A_%a.out"; then
            _matching_jobids+=("${_jid}")
        fi
    done
    if [[ ${#_matching_jobids[@]} -gt 0 ]]; then
        echo "[ERROR] Found active compare submissions for this OUTPUT_DIR: ${_matching_jobids[*]}"
        echo "[ERROR] Refusing to submit a duplicate array."
        echo "[HINT] Wait for completion/cancel existing jobs, or set ALLOW_DUPLICATE_SUBMIT=1 to override."
        exit 1
    fi
fi

# ---- 1. Generate manifest -------------------------------------------------
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

cd "${REPO_ROOT}"
METHODS_CSV="${METHODS_CSV}" SCENARIOS_CSV="${SCENARIOS_CSV}" OUTPUT_DIR="${OUTPUT_DIR}" SKIP_EXISTING="${SKIP_EXISTING}" DRY_RUN="${DRY_RUN}" python - <<'EOF' > "${MANIFEST}"
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "src"))
sys.path.insert(0, os.getcwd())

from scripts.benchmark.scenarios import build_manifest, SCENARIO_REGISTRY

methods = [m.strip() for m in os.environ["METHODS_CSV"].split(",") if m.strip()]
if os.environ.get("DRY_RUN", "0") != "1":
    from scripts.benchmark.methods import available_methods

    known_methods = set(available_methods())
    unknown_methods = [m for m in methods if m not in known_methods]
    if unknown_methods:
        raise SystemExit(
            f"Unknown methods requested: {unknown_methods}. Available: {sorted(known_methods)}"
        )

scenarios_csv = os.environ.get("SCENARIOS_CSV", "all")
if scenarios_csv == "all":
    wanted_scenarios = None
else:
    wanted_scenarios = [s.strip() for s in scenarios_csv.split(",") if s.strip()]
    unknown_scenarios = [s for s in wanted_scenarios if s not in SCENARIO_REGISTRY]
    if unknown_scenarios:
        raise SystemExit(
            f"Unknown scenarios requested: {unknown_scenarios}. Available: {sorted(SCENARIO_REGISTRY)}"
        )

pairs = build_manifest()
output_dir = os.environ["OUTPUT_DIR"]
skip_existing = os.environ.get("SKIP_EXISTING", "0") == "1"

for method in methods:
    for scenario, seed in pairs:
        if wanted_scenarios is not None and scenario not in wanted_scenarios:
            continue
        if skip_existing:
            out_path = os.path.join(output_dir, method, scenario, f"seed_{seed}.json")
            if os.path.exists(out_path):
                continue
        print(f"{method} {scenario} {seed}")
EOF

N=$(wc -l < "${MANIFEST}")
echo "[compare-submit] Manifest has ${N} jobs."
if [[ "${N}" -le 0 ]]; then
    echo "[ERROR] Manifest is empty: ${MANIFEST}"
    exit 1
fi

# ---- 2. Build method-specific params -------------------------------------
# identity baseline copies s_local to aligned output.
BASELINE_METHOD_PARAMS_JSON='{"pseudotime_key":"s_local","key_added":"aligned_pseudotime_identity"}'

# scDeBussy uses scenario-provided parameters by default; keep override empty.
SCDEBUSSY_METHOD_PARAMS_JSON=''

# CellAlign fixed-reference with medoid selection.
CELLALIGN_METHOD_PARAMS_JSON=''
if [[ "${DRY_RUN}" -ne 1 ]]; then
    CELLALIGN_METHOD_PARAMS_JSON=$(RSCRIPT_BIN="${RSCRIPT_BIN}" python - <<'EOF'
import json
import os

params = {
    "rscript_bin": os.environ["RSCRIPT_BIN"],
    "reference_policy": "medoid",
    "patient_key": "patient",
    "pseudotime_key": "s_local",
    "timeout_s": 1200,
}
print(json.dumps(params, separators=(",", ":")))
EOF
)
fi

# Base64-encode JSON strings so SLURM --export comma-splitting cannot corrupt them.
BASELINE_METHOD_PARAMS_B64=$(printf '%s' "${BASELINE_METHOD_PARAMS_JSON}" | base64 -w 0)
SCDEBUSSY_METHOD_PARAMS_B64=$(printf '%s' "${SCDEBUSSY_METHOD_PARAMS_JSON}" | base64 -w 0)
CELLALIGN_METHOD_PARAMS_B64=$(printf '%s' "${CELLALIGN_METHOD_PARAMS_JSON}" | base64 -w 0)

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[compare-submit] DRY RUN: no sbatch submission will be made."
    echo "[compare-submit] Planned array size: ${N}"
    echo "[compare-submit] Planned log dir   : ${LOG_DIR}"
    echo "[compare-submit] First 10 manifest rows:"
    sed -n '1,10p' "${MANIFEST}"
    exit 0
fi

# ---- 3. Submit array ------------------------------------------------------
JOBID=$(sbatch \
    --array=0-$((N - 1)) \
    --chdir="${OUTPUT_DIR}" \
    --export=ALL,MANIFEST="${MANIFEST}",OUTPUT_DIR="${OUTPUT_DIR}",REPO_ROOT="${REPO_ROOT}",BASELINE_METHOD_PARAMS_B64="${BASELINE_METHOD_PARAMS_B64}",SCDEBUSSY_METHOD_PARAMS_B64="${SCDEBUSSY_METHOD_PARAMS_B64}",CELLALIGN_METHOD_PARAMS_B64="${CELLALIGN_METHOD_PARAMS_B64}",TUNE_MODE="${TUNE_MODE}",TUNE_TRIALS="${TUNE_TRIALS}",TUNE_SEED="${TUNE_SEED}",METHOD_TUNE_TRIALS_MAP="${METHOD_TUNE_TRIALS_MAP}" \
    --output="${LOG_DIR}/compare_%A_%a.out" \
    --error="${LOG_DIR}/compare_%A_%a.err" \
    "${ARRAY_SCRIPT}" | awk '{print $NF}')

echo "[compare-submit] Submitted array job JOBID=${JOBID} (${N} tasks)"

echo
echo "To aggregate compare-only summary after completion:"
echo "  sbatch --dependency=afterok:${JOBID} --wrap=\"python ${REPO_ROOT}/scripts/benchmark/aggregate_results.py --results_dir ${OUTPUT_DIR} --output ${OUTPUT_DIR}/summary_compare.csv --methods identity scdebussy cellalign_fixed_reference cellalign_consensus genes2genes_fixed_reference genes2genes_consensus --parquet\""
echo
echo "To compare tuned runs only after completion:"
echo "  python ${REPO_ROOT}/scripts/benchmark/compare_methods.py --results_dir ${OUTPUT_DIR} --output ${OUTPUT_DIR}/comparison_summary_tuned.csv --methods scdebussy cellalign_fixed_reference cellalign_consensus genes2genes_fixed_reference genes2genes_consensus --tuning true --parquet"
