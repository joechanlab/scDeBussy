"""Snapshot of Codex's strict-convergence wrapper for the historical bridge (v1), frozen for the benchmark.

Runs the SHA-verified historical OOD bridge unchanged except that its EM loop is replaced by
historical_convergence_core.fit_until_stable (penalized objective, barycenter drift, stationarity,
coordinate drift and match-mass stable for 5 updates; at most --max-iterations, default 1600; a
capped fit is reported as such in termination.json).
Source: /home/wangm10/scdebussy-span-test/01_simulation/sim_v2_convergence_bridge.py
sha256 of the source: 913603427efaeefe0dfad7a7bef03bd3ffdce2c905ed5100807cd0548abdd9d1
Copied 2026-10-26; only the core module name and this header changed.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import llvmlite.binding  # noqa: F401
from pathlib import Path
import hashlib
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from historical_convergence_core import fit_until_stable

BASE=Path(__file__).with_name('historical_ood_bridge.py')
raw=BASE.read_bytes()
source=raw.decode()
old='    parser.add_argument("--head-prices")'
assert source.count(old)==1
source=source.replace(old,old+'\n    parser.add_argument("--max-iterations", type=int, default=1600)\n    parser.add_argument("--min-iterations", type=int, default=100)\n    parser.add_argument("--checkpoint-dir", required=True)')
old='    full_patients = tuple(patient for patient in fit_patients if patient != "RU1303")'
assert source.count(old)==1
source=source.replace(old,'''    np.savez_compressed(Path(args.output).with_name("landmarks.npz"),
        cell_ids=np.asarray(data.obs_names,str), genes=np.asarray(genes,str),
        patient_order=np.asarray(all_patients),
        landmark_expression=np.stack([expression[p] for p in all_patients]),
        landmark_matchable=np.stack([matchable[p] for p in all_patients]),
        landmark_coordinates=np.stack([coordinates[p] for p in all_patients]))
'''+old)
start=source.index('    history, stable, converged = [], 0, False')
end=source.index('    for patient in all_patients:',start)
source=source[:start]+'''    from backend import BETA
    from backend import second_diff_operator
    barycenter, posterior, history, converged = fit_until_stable(
        expression, graphs, barycenter, posterior, fit_patients, all_patients,
        local, observed, GAMMA, BETA, second_diff_operator(), expectation,
        barycenter_update_smooth, cell_positions_and_status,
        args.max_iterations, args.checkpoint_dir, args.min_iterations)

'''+source[end:]
exec(compile(source,str(BASE),'exec'),globals())
