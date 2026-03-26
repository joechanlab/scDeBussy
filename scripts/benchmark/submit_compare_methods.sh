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
#       [OUTPUT_DIR] [RSCRIPT_BIN] [SCENARIOS_CSV]
#
# Arguments:
#   OUTPUT_DIR     Root directory for outputs.
#                  Default: <repo>/results/benchmark_compare
#   RSCRIPT_BIN    Rscript executable with cellAlign installed.
#                  Default: /usersoftware/chanj3/Lamian/bin/Rscript
#   SCENARIOS_CSV  Optional comma-separated scenario names.
#                  Default: all scenarios in scripts/benchmark/scenarios.py
#
# Example:
# bash scripts/benchmark/submit_compare_methods.sh \
#     /scratch/chanj3/wangm10/compare_run \
#     /usersoftware/chanj3/Lamian/bin/Rscript

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

OUTPUT_DIR="${1:-${REPO_ROOT}/results/benchmark_compare}"
RSCRIPT_BIN="${2:-/usersoftware/chanj3/Lamian/bin/Rscript}"
SCENARIOS_CSV="${3:-all}"

MANIFEST="${SCRIPT_DIR}/manifest_compare.txt"
LOG_DIR="${OUTPUT_DIR}/logs"
ARRAY_SCRIPT="${SCRIPT_DIR}/run_array_job.sh"

METHODS=(
    identity
    scdebussy
    cellalign_fixed_reference
    cellalign_consensus
    genes2genes_fixed_reference
    genes2genes_consensus
)
METHODS_CSV="$(IFS=,; echo "${METHODS[*]}")"

echo "[compare-submit] Repo root      : ${REPO_ROOT}"
echo "[compare-submit] Output dir     : ${OUTPUT_DIR}"
echo "[compare-submit] Methods        : ${METHODS_CSV}"
echo "[compare-submit] Rscript        : ${RSCRIPT_BIN}"
echo "[compare-submit] Scenarios      : ${SCENARIOS_CSV}"
echo "[compare-submit] Manifest       : ${MANIFEST}"

if [[ ! -x "${RSCRIPT_BIN}" ]]; then
    echo "[ERROR] RSCRIPT_BIN is not executable: ${RSCRIPT_BIN}"
    exit 1
fi

# ---- 1. Generate manifest -------------------------------------------------
cd "${REPO_ROOT}"
METHODS_CSV="${METHODS_CSV}" SCENARIOS_CSV="${SCENARIOS_CSV}" python - <<'EOF' > "${MANIFEST}"
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "src"))
sys.path.insert(0, os.getcwd())

from scripts.benchmark.scenarios import build_manifest, SCENARIO_REGISTRY
from scripts.benchmark.methods import available_methods

methods = [m.strip() for m in os.environ["METHODS_CSV"].split(",") if m.strip()]
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
for method in methods:
    for scenario, seed in pairs:
        if wanted_scenarios is not None and scenario not in wanted_scenarios:
            continue
        print(f"{method} {scenario} {seed}")
EOF

N=$(wc -l < "${MANIFEST}")
echo "[compare-submit] Manifest has ${N} jobs."
if [[ "${N}" -le 0 ]]; then
    echo "[ERROR] Manifest is empty: ${MANIFEST}"
    exit 1
fi

# ---- 2. Create log directory ---------------------------------------------
mkdir -p "${LOG_DIR}"

# ---- 3. Build method-specific params -------------------------------------
# identity baseline copies s_local to aligned output.
BASELINE_METHOD_PARAMS_JSON='{"pseudotime_key":"s_local","key_added":"aligned_pseudotime_identity"}'

# scDeBussy uses scenario-provided parameters by default; keep override empty.
SCDEBUSSY_METHOD_PARAMS_JSON=''

# CellAlign fixed-reference with medoid selection.
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

# Base64-encode JSON strings so SLURM --export comma-splitting cannot corrupt them.
BASELINE_METHOD_PARAMS_B64=$(printf '%s' "${BASELINE_METHOD_PARAMS_JSON}" | base64 -w 0)
SCDEBUSSY_METHOD_PARAMS_B64=$(printf '%s' "${SCDEBUSSY_METHOD_PARAMS_JSON}" | base64 -w 0)
CELLALIGN_METHOD_PARAMS_B64=$(printf '%s' "${CELLALIGN_METHOD_PARAMS_JSON}" | base64 -w 0)

# ---- 4. Submit array ------------------------------------------------------
JOBID=$(sbatch \
    --array=0-$((N - 1)) \
    --export=ALL,MANIFEST="${MANIFEST}",OUTPUT_DIR="${OUTPUT_DIR}",REPO_ROOT="${REPO_ROOT}",BASELINE_METHOD_PARAMS_B64="${BASELINE_METHOD_PARAMS_B64}",SCDEBUSSY_METHOD_PARAMS_B64="${SCDEBUSSY_METHOD_PARAMS_B64}",CELLALIGN_METHOD_PARAMS_B64="${CELLALIGN_METHOD_PARAMS_B64}" \
    --output="${LOG_DIR}/compare_%A_%a.out" \
    --error="${LOG_DIR}/compare_%A_%a.err" \
    "${ARRAY_SCRIPT}" | awk '{print $NF}')

echo "[compare-submit] Submitted array job JOBID=${JOBID} (${N} tasks)"

echo
echo "To aggregate compare-only summary after completion:"
echo "  sbatch --dependency=afterok:${JOBID} --wrap=\"python ${REPO_ROOT}/scripts/benchmark/aggregate_results.py --results_dir ${OUTPUT_DIR} --output ${OUTPUT_DIR}/summary_compare.csv --methods identity scdebussy cellalign_fixed_reference cellalign_consensus genes2genes_fixed_reference genes2genes_consensus --parquet\""
