"""Isolated runner for the pinned historical RU1303-only gated algorithm.

This file is orchestration only. Graph construction, posterior inference,
costs, smoothing, and cell projection are imported from the frozen historical
worktree named on the command line.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse


MAX_ITER, TOL, PATIENCE, GAMMA = 100, 1e-6, 3, 1.0
CHEAP_PRICE, FULL_PRICE = 0.10, 0.516917977676361
FIT_CONFIG = {
    "patient_key": "patient", "pseudotime_key": "s_local", "n_landmarks": 100,
    "consensus_resolution": 100, "landmark_method": "fixed", "locality_radius": 0.1,
    "beta": 1.0, "mode": "open", "n_min": 5, "epsilon": 1e-6,
    "kappa_i": 3.0, "kappa_d": 3.0, "calibration_quantile": 0.9,
    "calibration_band": 0.25, "max_iter": 200, "tol": 1e-6, "patience": 3,
    "d_min": 0.001, "chunk_size": 1024, "backend": "numba", "precision_mode": "unit",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--landmark-cache", required=True)
    parser.add_argument("--fit-patients", required=True)
    parser.add_argument("--cells-per-window", type=int)
    parser.add_argument("--fit-genes")
    parser.add_argument("--tail-prices")
    parser.add_argument("--head-prices")
    args = parser.parse_args()

    source = Path(args.source_root)
    joint = source / "validation/v21/experiment_round/eight_patient/joint_barycenter"
    pairwise = source / "validation/v21/experiment_round/eight_patient/ru1303_ru581_pairwise"
    sys.path[:0] = [str(source / "src"), str(Path(__file__).resolve().parent)]
    from scdebussy import align  # noqa: PLC0415
    from backend import barycenter_update_smooth, cell_positions_and_status, expectation, match_cost, compile_graph, ETA

    rows = np.load(args.rows)
    data = ad.read_h5ad(args.input)[rows].copy()
    genes = json.loads(Path(args.fit_genes or args.panel).read_text())
    observed = data.obs["patient"].astype(str).to_numpy()
    local = data.obs["s_local"].to_numpy(float)
    fit_patients = tuple(json.loads(args.fit_patients))
    all_patients = tuple(sorted(np.unique(observed)))
    tail_prices = ({patient: (CHEAP_PRICE if patient == "RU1303" else FULL_PRICE)
                    for patient in all_patients} if args.tail_prices is None
                   else {str(key): float(value) for key, value in json.loads(args.tail_prices).items()})
    if set(tail_prices) != set(all_patients):
        raise ValueError("tail prices must contain exactly one value per observed patient")
    head_prices = (None if args.head_prices is None
                   else {str(key): float(value) for key, value in json.loads(args.head_prices).items()})
    if head_prices is not None and set(head_prices) != set(all_patients):
        raise ValueError("head prices must contain exactly one value per observed patient")
    missing = set(fit_patients) - set(all_patients)
    if missing:
        raise ValueError(f"fit patients absent from data: {sorted(missing)}")

    landmark_cache = np.load(args.landmark_cache, allow_pickle=False)
    exact_full_input = args.cells_per_window is None and (
        np.array_equal(np.asarray(data.obs_names, str), landmark_cache["cell_ids"])
        and np.array_equal(np.asarray(genes, str), landmark_cache["genes"].astype(str))
    )
    if exact_full_input:
        names = landmark_cache["patient_order"].astype(str).tolist()
        all_expression = landmark_cache["landmark_expression"].reshape(len(names), 100, len(genes))
        all_matchable = landmark_cache["landmark_matchable"].reshape(len(names), 100)
        expression = {patient: all_expression[names.index(patient)] for patient in all_patients}
        matchable = {patient: all_matchable[names.index(patient)] for patient in all_patients}
        landmark_source = "frozen_full_cohort_cache"
    elif args.cells_per_window is None:
        fitted = align.fit(data[:, genes].copy(), **FIT_CONFIG)
        expression = {patient: fitted.landmarks[fitted.patients.index(patient)].expression for patient in all_patients}
        matchable = {patient: fitted.landmarks[fitted.patients.index(patient)].matchable for patient in all_patients}
        coordinates = {patient: fitted.landmarks[fitted.patients.index(patient)].coordinates for patient in all_patients}
        landmark_source = "recomputed_same_historical_landmark_api"
    else:
        # Deterministic equal-cell windows preserve the original pseudotime
        # coordinate and never split exact ties.  This changes only the input
        # trajectory summary; all graph, cost, truncation, and EM logic below
        # is the frozen historical implementation.
        matrix = data[:, genes].X
        expression, matchable, coordinates = {}, {}, {}
        for patient in all_patients:
            ids = np.flatnonzero(observed == patient)
            order = ids[np.argsort(local[ids], kind="stable")]
            tie_stops = np.r_[np.flatnonzero(np.diff(local[order]) != 0) + 1, len(order)]
            windows, start = [], 0
            for stop in tie_stops:
                if stop - start >= args.cells_per_window:
                    windows.append(order[start:stop]); start = stop
            if start < len(order):
                if windows and len(order) - start < args.cells_per_window / 2:
                    windows[-1] = np.r_[windows[-1], order[start:]]
                else:
                    windows.append(order[start:])
            values = []
            for window in windows:
                block = matrix[window]
                mean = block.mean(axis=0)
                values.append(np.asarray(mean).ravel())
            window_expression = np.stack(values)
            window_coordinates = np.array([np.median(local[window]) for window in windows])
            # The frozen DP graph requires exactly 100 patient rows.  Interpolate
            # the equal-weight window summaries at their original coordinates
            # onto that required lattice; do not stretch either endpoint.
            grid = np.linspace(0, 1, FIT_CONFIG["n_landmarks"])
            expression[patient] = np.column_stack([
                np.interp(grid, window_coordinates, window_expression[:, gene])
                for gene in range(window_expression.shape[1])
            ])
            coordinates[patient] = grid
            matchable[patient] = (grid >= window_coordinates.min()) & (grid <= window_coordinates.max())
        landmark_source = f"equal_cell_windows_{args.cells_per_window}"

    if exact_full_input:
        coordinates = {patient: np.linspace(0, 1, len(expression[patient])) for patient in all_patients}

    full_patients = tuple(patient for patient in fit_patients if patient != "RU1303")
    if not full_patients:
        raise ValueError("historical initialization needs at least one non-RU1303 fit patient")
    consensus_grid = np.linspace(0, 1, FIT_CONFIG["consensus_resolution"])
    seed = np.mean([
        np.column_stack([np.interp(consensus_grid, coordinates[patient], expression[patient][:, gene])
                         for gene in range(expression[patient].shape[1])])
        for patient in full_patients
    ], axis=0)

    def graph(patient):
        compiled, nodes = compile_graph(
            match_cost(expression[patient], seed), tail_prices[patient],
            matchable=matchable[patient],
        )
        if head_prices is None:
            return compiled, nodes
        # The frozen graph prices skipping consensus points 0..j-1 before the
        # first match at j * ETA (start edges, kind 5), plus one shared
        # constant when the first landmark is unmatchable; swap only the rate.
        src, dst, static, slope, kind, mi, mj = compiled
        start = kind == 5
        offset = static[start] - mj[start] * ETA
        if not np.allclose(offset, offset[0]):
            raise RuntimeError("unexpected start-edge pricing in frozen graph")
        static = static.copy()
        static[start] = offset + mj[start] * head_prices[patient]
        return (src, dst, static, slope, kind, mi, mj), nodes

    graphs = {patient: graph(patient) for patient in fit_patients}
    seed_posterior = expectation({p: expression[p] for p in fit_patients}, seed, graphs, fit_patients)
    barycenter, _ = barycenter_update_smooth(
        {p: expression[p] for p in fit_patients}, seed_posterior, seed, fit_patients, GAMMA
    )
    posterior = expectation({p: expression[p] for p in fit_patients}, barycenter, graphs, fit_patients)
    history, stable, converged = [], 0, False
    for iteration in range(1, MAX_ITER + 1):
        before = -sum(posterior[p]["log_z"] for p in fit_patients)
        updated, support = barycenter_update_smooth(
            {p: expression[p] for p in fit_patients}, posterior, barycenter, fit_patients, GAMMA
        )
        following = expectation({p: expression[p] for p in fit_patients}, updated, graphs, fit_patients)
        after = -sum(following[p]["log_z"] for p in fit_patients)
        relative = (before - after) / max(1.0, abs(before))
        history.append({"iteration": iteration, "negative_log_partition": float(after),
                        "relative_improvement": float(relative),
                        "barycenter_max_change": float(np.max(np.abs(updated - barycenter))),
                        "min_consensus_match_support": float(support.min())})
        barycenter, posterior = updated, following
        stable = stable + 1 if relative <= TOL else 0
        if stable >= PATIENCE:
            converged = True
            break

    for patient in all_patients:
        if patient not in posterior:
            patient_graph = graph(patient)
            posterior[patient] = expectation(
                {patient: expression[patient]}, barycenter, {patient: patient_graph}, (patient,)
            )[patient]
    position, status, unprojected = cell_positions_and_status(
        posterior, local, observed, all_patients
    )
    np.savez_compressed(
        args.output, cell_ids=np.asarray(data.obs_names, str), position=position,
        status=status, unprojected=unprojected, final_barycenter=barycenter,
        fit_patients=np.asarray(fit_patients), all_patients=np.asarray(all_patients),
        history=json.dumps(history), converged=converged,
        landmark_source=landmark_source,
        cells_per_window=-1 if args.cells_per_window is None else args.cells_per_window,
        tail_prices=json.dumps(tail_prices, sort_keys=True),
        head_prices=json.dumps(head_prices, sort_keys=True),
        consensus_match_mass=np.stack([
            posterior[patient]["matrix"].sum(axis=0) for patient in all_patients
        ]),
    )


if __name__ == "__main__":
    main()
