"""Select label-free, recurrent within-patient pseudotime genes for HTAN."""

import argparse
import csv
import json
from pathlib import Path

import anndata as ad
import numpy as np


def score_genes(data, excluded, n_bins=25, min_cells=20):
    genes = np.asarray(data.var_names, str)
    patients = data.obs.patient.astype(str).to_numpy()
    time = data.obs.s_local.to_numpy(float)
    patient_names = sorted(np.unique(patients))
    profiles = np.full((len(patient_names), n_bins, len(genes)), np.nan, dtype=np.float32)
    means = np.zeros((len(patient_names), len(genes)), dtype=np.float64)
    detected = np.zeros((len(patient_names), len(genes)), dtype=np.float64)
    for p, patient in enumerate(patient_names):
        patient_ix = np.flatnonzero(patients == patient)
        matrix = data.X[patient_ix]
        means[p] = np.asarray(matrix.mean(axis=0)).ravel()
        detected[p] = np.asarray((matrix > 0).mean(axis=0)).ravel()
        bins = np.minimum((time[patient_ix] * n_bins).astype(int), n_bins - 1)
        for b in range(n_bins):
            ix = np.flatnonzero(bins == b)
            if len(ix) >= min_cells:
                profiles[p, b] = np.asarray(matrix[ix].mean(axis=0)).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        within_var = np.nanvar(profiles, axis=1)
    supported = np.isfinite(within_var) & (within_var > .01) & (detected > .03) & (means > .05)
    n_supported = supported.sum(axis=0)
    temporal_signal = np.array([
        np.median(within_var[supported[:, g], g]) if supported[:, g].any() else np.nan
        for g in range(len(genes))
    ])
    between_offset = np.var(means, axis=0)
    corr_sum = np.zeros(len(genes))
    corr_n = np.zeros(len(genes))
    for a in range(len(patient_names)):
        for b in range(a + 1, len(patient_names)):
            common = np.isfinite(profiles[a, :, 0]) & np.isfinite(profiles[b, :, 0])
            if common.sum() < 8:
                continue
            x = profiles[a, common].astype(float)
            y = profiles[b, common].astype(float)
            x -= x.mean(axis=0); y -= y.mean(axis=0)
            denom = np.sqrt((x * x).sum(axis=0) * (y * y).sum(axis=0))
            corr = np.divide((x * y).sum(axis=0), denom, out=np.zeros(len(genes)), where=denom > 1e-10)
            eligible = supported[a] & supported[b]
            corr_sum[eligible] += corr[eligible]
            corr_n[eligible] += 1
    mean_corr = np.divide(corr_sum, corr_n, out=np.zeros(len(genes)), where=corr_n > 0)
    offset_ratio = between_offset / (temporal_signal + 1e-8)
    score = temporal_signal * np.maximum(mean_corr, 0) * np.sqrt(n_supported / len(patient_names)) / (1 + offset_ratio)
    ineligible = (n_supported < 3) | (corr_n < 2) | ~np.isfinite(score) | np.isin(genes, excluded)
    score[ineligible] = -np.inf
    return genes, score, temporal_signal, mean_corr, offset_ratio, n_supported, corr_n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adata", type=Path, required=True)
    parser.add_argument("--held-out", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    data = ad.read_h5ad(args.adata)
    excluded = json.loads(args.held_out.read_text())["genes"]
    output = score_genes(data, excluded)
    genes, score, var, corr, offset, support, pairs = output
    order = np.argsort(-score, kind="stable")
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "gene_scores.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gene", "score", "within_patient_temporal_variance", "mean_pair_correlation", "between_to_within_ratio", "supporting_patients", "correlated_pairs"])
        for i in order:
            writer.writerow([genes[i], score[i], var[i], corr[i], offset[i], support[i], pairs[i]])
    for size in (100, 250, 500, 1000):
        if not np.isfinite(score[order[size - 1]]):
            raise ValueError(f"fewer than {size} eligible genes")
        (args.output_dir / f"top_{size}.json").write_text(json.dumps(genes[order[:size]].tolist(), indent=2) + "\n")
    (args.output_dir / "selection_config.json").write_text(json.dumps({"n_bins": 25, "min_cells_per_bin": 20, "minimum_supporting_patients": 3, "minimum_pair_comparisons": 2, "excluded": excluded, "input": str(args.adata), "uses_cell_type_labels": False}, indent=2) + "\n")
    print("top 20:", genes[order[:20]].tolist())


if __name__ == "__main__":
    main()
