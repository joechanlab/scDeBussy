"""Experimental fixed-consensus DTW transitions and terminal deletion sweep.

This is a post-fit Viterbi prototype. It does not change v2 posterior inference or
its conservation identities. A (1,2) step matches one patient landmark to the
mean of two consensus costs; a (2,1) step matches two patient landmarks to one
consensus row and pays both costs. Both pay omega. Interior insertion/deletion
costs and terminal deletion costs are separate.
"""

import argparse
import csv
import json
from pathlib import Path

import anndata as ad
import numpy as np

from summarize_and_plot import score


def align_cost(cost, matchable, consensus, omega, internal_cost=1.0, terminal_cost=1.0):
    n, t = cost.shape
    dp = np.full((n + 1, t + 1), np.inf)
    parent = np.full((n + 1, t + 1, 3), -1, dtype=int)
    dp[0, 0] = 0.0

    def offer(i, j, ii, jj, extra, action):
        proposed = dp[i, j] + extra
        if proposed < dp[ii, jj] - 1e-12:
            dp[ii, jj] = proposed
            parent[ii, jj] = i, j, action

    for i in range(n + 1):
        for j in range(t + 1):
            if not np.isfinite(dp[i, j]):
                continue
            if i < n and j < t and matchable[i]:
                offer(i, j, i + 1, j + 1, cost[i, j], 0)
            if i < n and j + 1 < t and matchable[i]:
                offer(i, j, i + 1, j + 2, .5 * (cost[i, j] + cost[i, j + 1]) + omega, 1)
            if i + 1 < n and j < t and matchable[i] and matchable[i + 1]:
                offer(i, j, i + 2, j + 1, cost[i, j] + cost[i + 1, j] + omega, 2)
            if i < n:
                offer(i, j, i + 1, j, internal_cost, 3)
            if j < t:
                offer(i, j, i, j + 1, internal_cost, 4)

    final_costs = dp + terminal_cost * (t - np.arange(t + 1))[None, :] + internal_cost * (n - np.arange(n + 1))[:, None]
    i, j = np.unravel_index(np.argmin(final_costs), final_costs.shape)
    terminal_local, terminal_consensus = n - i, t - j
    mapped = np.full(n, np.nan)
    counts = np.zeros(5, int)
    while i or j:
        pi, pj, action = parent[i, j]
        if action < 0:
            raise ArithmeticError("DP path has no predecessor")
        counts[action] += 1
        if action == 0:
            mapped[pi] = consensus[pj]
        elif action == 1:
            mapped[pi] = .5 * (consensus[pj] + consensus[pj + 1])
        elif action == 2:
            mapped[pi] = consensus[pj]
            mapped[pi + 1] = consensus[pj]
        i, j = int(pi), int(pj)
    valid = np.isfinite(mapped)
    if not valid.any():
        raise ArithmeticError("DP selected a path with no matches")
    mapped = np.interp(np.arange(n), np.flatnonzero(valid), mapped[valid])
    if np.any(np.diff(mapped) < -1e-12):
        raise ArithmeticError("nonmonotone mapping")
    return mapped, {"matches": int(counts[0]), "warp_12": int(counts[1]), "warp_21": int(counts[2]),
                    "internal_patient_gap": int(counts[3]), "internal_consensus_gap": int(counts[4]),
                    "terminal_patient_gap": int(terminal_local), "terminal_consensus_gap": int(terminal_consensus),
                    "path_cost": float(np.min(final_costs))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adata", type=Path, required=True)
    parser.add_argument("--fit-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--penalty-scale", choices=("fixed", "median"), default="fixed")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    data = ad.read_h5ad(args.adata, backed="r")
    fit = np.load(args.fit_cache, allow_pickle=False)
    if not np.array_equal(fit["cell_ids"], np.asarray(data.obs_names, str)):
        raise ValueError("fit cache does not match input")
    names = fit["patient_order"].tolist()
    patient = data.obs.patient.astype(str).to_numpy()
    local = data.obs.s_local.to_numpy(float)
    offsets = fit["landmark_offsets"]
    rows = []
    for terminal_cost in (1.0, .2, .1):
        for omega in (0.0, .01, .03, .06, .1, .2, .4):
            name = f"dtw_omega_{omega:g}_terminal_{terminal_cost:g}"
            position = np.full(data.n_obs, np.nan)
            details = {}
            maps = []
            for p, label in enumerate(names):
                start, stop = offsets[p:p+2]
                coords = fit["landmark_coordinates"][start:stop]
                matchable = fit["landmark_matchable"][start:stop]
                scale = float(np.median(np.diag(fit["cost_matrices"][p]))) if args.penalty_scale == "median" else 1.0
                mapped, detail = align_cost(fit["cost_matrices"][p], matchable, fit["consensus_coordinates"], omega * scale,
                                            internal_cost=(2 * scale if args.penalty_scale == "median" else 1.0),
                                            terminal_cost=terminal_cost * scale)
                mask = patient == label
                position[mask] = np.interp(local[mask], coords, mapped)
                details[label] = detail
                maps.append(mapped)
            position[~np.isfinite(fit["position"])] = np.nan
            scored = score(data.obs, position, fit["status"], fit["unprojected"], name)
            overall = next(row for row in scored if row["scope"] == "overall")
            overall.update({"omega": omega, "terminal_cost": terminal_cost, "penalty_scale": args.penalty_scale, "iterations": 0,
                            "converged": "fixed_barycenter_viterbi", "objective": args.fit_cache.stem})
            for detail in details.values():
                for key in ("matches", "warp_12", "warp_21", "internal_patient_gap", "internal_consensus_gap", "terminal_patient_gap", "terminal_consensus_gap"):
                    overall[key] = overall.get(key, 0) + detail[key]
            rows.append(overall)
            np.savez_compressed(args.output_dir / f"{name}.npz", cell_ids=fit["cell_ids"], cell_patient=fit["cell_patient"],
                                cell_type=fit["cell_type"], position=position, status=fit["status"],
                                unprojected=fit["unprojected"], patient_order=fit["patient_order"], maps=np.asarray(maps),
                                details=json.dumps(details), config=json.dumps({"name": name, "type": "fixed-consensus Viterbi prototype", "omega": omega, "internal_cost_multiplier": 2.0 if args.penalty_scale == "median" else 1.0, "terminal_cost": terminal_cost, "penalty_scale": args.penalty_scale, "base_fit": str(args.fit_cache)}))
            print(name, f"coverage={overall['coverage']:.3f}", f"purity={overall['purity']:.3f}", f"displacement={overall['mean_abs_displacement']:.3f}", flush=True)
    fields = sorted({key for row in rows for key in row})
    with (args.output_dir / "sweep.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
