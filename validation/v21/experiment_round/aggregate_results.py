"""Join restricted fits and DTW sweeps into one machine-readable run table."""

import argparse
import csv
import json
from pathlib import Path

import anndata as ad
import numpy as np

from summarize_and_plot import score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adata", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--sweep", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = ad.read_h5ad(args.adata, backed="r")
    rows = []
    raw = score(data.obs, data.obs.s_local.to_numpy(float), np.full(data.n_obs, "input"), np.zeros(data.n_obs), "s_local")[-1]
    raw.update({"run_type": "control", "n_genes": 0, "seed": "none", "input": str(args.adata)})
    rows.append(raw)
    for path in sorted(args.results.glob("*.npz")):
        cache = np.load(path, allow_pickle=False)
        cfg = json.loads(str(cache["config"]))
        if not np.array_equal(cache["cell_ids"], np.asarray(data.obs_names, str)):
            raise ValueError(f"cache order differs: {path}")
        row = score(data.obs, cache["position"], cache["status"], cache["unprojected"], cfg["name"])[-1]
        fit = cfg["fit"]
        row.update({"run_type": "v2_restricted", "n_genes": len(cache["genes"]) if "genes" in cache.files else data.n_vars,
                    "seed": "none", "input": str(args.adata), "cache": str(path),
                    "gene_panel": cfg["transform"].get("genes_file", "all"),
                    "patient_gene_center": cfg["transform"].get("patient_gene_center", False),
                    "precision_mode": fit.get("precision_mode", "estimated"),
                    "contribution_cap_quantile": fit.get("contribution_cap_quantile", ""),
                    "n_landmarks": fit["n_landmarks"], "consensus_resolution": fit["consensus_resolution"],
                    "kappa_i": fit["kappa_i"], "kappa_d": fit["kappa_d"],
                    "iterations": json.loads((path.with_suffix(".json")).read_text())["iterations"],
                    "converged": json.loads((path.with_suffix(".json")).read_text())["converged"],
                    "source_sha256": json.loads(str(cache["provenance"]))["source_sha256"]})
        rows.append(row)
    for sweep in args.sweep:
        for item in csv.DictReader(sweep.open()):
            item["run_type"] = "fixed_consensus_dtw_prototype"
            item["seed"] = "none"
            item["input"] = str(args.adata)
            item["cache"] = str(sweep.parent / f"{item['method']}.npz")
            rows.append(item)
    fields = sorted({key for row in rows for key in row})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(len(rows), "runs written to", args.output)


if __name__ == "__main__":
    main()
