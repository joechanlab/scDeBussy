"""Fit one bounded HTAN v2 experiment and cache its projection and diagnostics."""

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.neighbors import NearestNeighbors

from scdebussy import align, tl
from scdebussy.align._barycenter import costs


def patient_metrics(obs, position, status):
    """Metrics on one fixed input ordering; rejected means a nonfinite position."""
    patients = obs["patient"].astype(str).to_numpy()
    labels = obs["cell_type_final2"].astype(str).to_numpy()
    position = np.asarray(position, float)
    finite = np.isfinite(position)
    scores = np.full(len(obs), np.nan)
    if finite.sum() >= 50:
        ix = np.flatnonzero(finite)
        nn = NearestNeighbors(n_neighbors=50, algorithm="kd_tree").fit(position[ix, None])
        _, neighbors = nn.kneighbors(position[ix, None])
        for local, original in enumerate(ix):
            neighbor_ix = ix[neighbors[local]]
            cross = patients[neighbor_ix] != patients[original]
            if cross.any():
                scores[original] = np.mean(labels[neighbor_ix][cross] == labels[original])
            else:
                scores[original] = 0.0
    rows = []
    for patient in sorted(np.unique(patients)):
        mask = patients == patient
        p = position[mask]
        valid = p[np.isfinite(p)]
        occupied = np.zeros(100, bool)
        if len(valid):
            occupied[np.clip((valid * 99).astype(int), 0, 99)] = True
        original = obs.loc[mask, "s_local"].to_numpy(float)
        finite_mask = np.isfinite(p)
        rows.append(
            {
                "method": None,
                "patient": patient,
                "cells": int(mask.sum()),
                "coverage": float(occupied.mean()),
                "purity": float(np.nanmean(scores[mask])),
                "rejection_rate": float((~finite_mask).mean()),
                "unsupported_rate": float((np.asarray(status)[mask] == "unsupported").mean()),
                "mean_unprojected_mass": None,
                "mean_abs_displacement": float(np.mean(np.abs(p[finite_mask] - original[finite_mask]))) if finite_mask.any() else None,
                "max_abs_displacement": float(np.max(np.abs(p[finite_mask] - original[finite_mask]))) if finite_mask.any() else None,
                "position_min": float(valid.min()) if len(valid) else None,
                "position_max": float(valid.max()) if len(valid) else None,
            }
        )
    return rows, scores


def summary(data, position, status, unprojected):
    data.obs["_experiment_position"] = position
    data.uns["_experiment_grid"] = {"expression": np.zeros((100, 1))}
    coverage = tl.compute_patient_coverage(
        data, patient_key="patient", aligned_pseudotime_key="_experiment_position", barycenter_key="_experiment_grid"
    )
    purity = tl.cross_batch_knn_purity(
        data,
        pseudotime_key="_experiment_position",
        label_key="cell_type_final2",
        batch_key="patient",
        n_neighbors=50,
        inplace=False,
    )["global_cross_batch_purity"]
    per_patient, _ = patient_metrics(data.obs, position, status)
    for row in per_patient:
        mask = data.obs["patient"].astype(str).to_numpy() == row["patient"]
        row["mean_unprojected_mass"] = float(np.mean(unprojected[mask]))
    finite = np.isfinite(position)
    raw = data.obs["s_local"].to_numpy(float)
    ru = data.obs["patient"].astype(str).to_numpy() == "RU1303"
    ru_detail = {}
    for label in ("LUAD", "SCLC-A", "SCLC-N"):
        values = np.asarray(position)[ru & (data.obs["cell_type_final2"].astype(str).to_numpy() == label)]
        values = values[np.isfinite(values)]
        ru_detail[label] = {"n": int(len(values)), "median_position": float(np.median(values)) if len(values) else None}
    other_n = np.asarray(position)[~ru & (data.obs["cell_type_final2"].astype(str).to_numpy() == "SCLC-N")]
    other_n = other_n[np.isfinite(other_n)]
    ru_a = np.asarray(position)[ru & (data.obs["cell_type_final2"].astype(str).to_numpy() == "SCLC-A")]
    ru_a = ru_a[np.isfinite(ru_a)]
    if len(other_n) and len(ru_a):
        low, high = np.quantile(other_n, [0.1, 0.9])
        ru_detail["sclc_a_in_other_sclc_n_q10_q90"] = float(np.mean((ru_a >= low) & (ru_a <= high)))
    return {
        "coverage": float(coverage.mean()),
        "purity": float(purity),
        "rejection_rate": float((~finite).mean()),
        "unsupported_rate": float((np.asarray(status) == "unsupported").mean()),
        "mean_unprojected_mass": float(np.mean(unprojected)),
        "mean_abs_displacement": float(np.mean(np.abs(position[finite] - raw[finite]))),
        "max_abs_displacement": float(np.max(np.abs(position[finite] - raw[finite]))),
        "per_patient": per_patient,
        "ru1303": ru_detail,
        "held_out_marker_agreement": {
            "available": False,
            "reason": "All expression genes, including known lineage markers, were supplied to fitting; this cohort has no predeclared held-out marker panel.",
        },
    }


def transformed_data(path, transform):
    data = ad.read_h5ad(path)
    if transform.get("genes_file"):
        genes = json.loads(Path(transform["genes_file"]).read_text())
        if len(genes) != len(set(genes)) or any(g not in data.var_names for g in genes):
            raise ValueError("gene panel contains duplicate or absent genes")
        data = data[:, genes].copy()
    if not transform.get("patient_gene_center"):
        return data
    x = data.X.toarray() if hasattr(data.X, "toarray") else np.asarray(data.X)
    x = np.asarray(x, dtype=np.float32)
    patients = data.obs["patient"].astype(str).to_numpy()
    if transform.get("patient_gene_center"):
        for patient in np.unique(patients):
            mask = patients == patient
            x[mask] -= x[mask].mean(axis=0, dtype=np.float64).astype(np.float32)
    data.X = x
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adata", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    name = cfg["name"]
    cache = args.output_dir / f"{name}.npz"
    report = args.output_dir / f"{name}.json"
    if cache.exists() or report.exists():
        raise FileExistsError(f"refusing to overwrite {cache} or {report}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print(f"[{name}] loading and transforming input", flush=True)
    data = transformed_data(args.adata, cfg["transform"])
    print(f"[{name}] fitting {data.n_obs} cells x {data.n_vars} genes", flush=True)
    result = align.fit(data, **cfg["fit"])
    print(f"[{name}] projecting and scoring", flush=True)
    projection = align.project(result)
    metrics = summary(data, projection.position, projection.status, projection.unprojected)
    metrics.update(
        {
            "name": name,
            "description": cfg["description"],
            "input": str(args.adata),
            "requested_config": cfg,
            "resolved_fit_config": result.config,
            "converged": bool(result.converged),
            "iterations": len(result.history),
            "diagnostics": result.diagnostics,
            "provenance": result.provenance,
            "seconds": time.monotonic() - started,
        }
    )
    report.write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(
        cache,
        cell_ids=np.asarray(data.obs_names, str),
        cell_patient=np.asarray(data.obs["patient"], str),
        cell_type=np.asarray(data.obs["cell_type_final2"], str),
        position=np.asarray(projection.position, float),
        status=np.asarray(projection.status, str),
        unprojected=np.asarray(projection.unprojected, float),
        config=json.dumps(cfg),
        resolved_fit_config=json.dumps(result.config),
        diagnostics=json.dumps(result.diagnostics),
        provenance=json.dumps(result.provenance),
        genes=np.asarray(result.genes, str),
        barycenter=result.barycenter,
        consensus_coordinates=result.consensus_coordinates,
        patient_order=np.asarray(result.patients, str),
        landmark_offsets=np.r_[0, np.cumsum([len(lm.coordinates) for lm in result.landmarks])],
        landmark_coordinates=np.concatenate([lm.coordinates for lm in result.landmarks]),
        landmark_expression=np.concatenate([lm.expression for lm in result.landmarks]),
        landmark_precision=np.concatenate([lm.precision for lm in result.landmarks]),
        landmark_matchable=np.concatenate([lm.matchable for lm in result.landmarks]),
        cost_matrices=np.stack([costs(lm, result.barycenter) for lm in result.landmarks]),
    )
    print(json.dumps({k: metrics[k] for k in ("name", "coverage", "purity", "rejection_rate", "mean_abs_displacement", "converged", "iterations", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
