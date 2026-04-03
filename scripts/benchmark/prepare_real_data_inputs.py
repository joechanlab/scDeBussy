#!/usr/bin/env python
"""Prepare per-patient real-data inputs for run_real_data.py.

This script converts the notebook preprocessing workflow into a CLI:
1. Load per-patient h5ad paths from an Excel manifest (Sheet1).
2. Join external cell-type labels by barcode.
3. Keep only tumour cell types.
4. Rescale patient-specific pseudotime into [0, 1] as `s_local`.
5. Write one staged h5ad per patient and a staged manifest file.

The staged manifest is then used by the SLURM array scripts:
    scripts/benchmark/submit_real_data_methods.sh
    scripts/benchmark/run_real_data_methods_array.sh

python scripts/benchmark/prepare_real_data_inputs.py \
  --path-list /home/wangm10/HTA.lung.NE_plasticity_scDeBussy/figures/figure4/Basal/figure_4_adata_path.xlsx \
  --label-csv /data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_All/cell_type_labels.csv \
  --samples RU1083,RU263,RU1444,RU151,RU1303,RU581,RU831,RU1646,RU942 \
  --staging-dir /scratch/chanj3/wangm10/HTAN/preprocessed/ \
  --staged-manifest /scratch/chanj3/wangm10/HTAN/preprocessed/staged_paths.txt \
  --patient-key patient \
  --pseudotime-key s_local \
  --cell-type-key cell_type_final2 \
  --tumor-cell-types "LUAD,LUSC,SCLC-A,SCLC-N,EMT,FOXI1+ NonNE SCLC,NonNE SCLC" \
  --pseudotime-column-field pseudotime_col \
  --clip-low 1 \
  --clip-high 99 \
  --overwrite
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


def _parse_csv_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _make_portable_adata(adata):
    """Drop non-essential AnnData metadata that can break cross-env h5ad reads.

    The staged files are benchmark inputs, not analysis archives. Keep the core
    matrix, obs/var, and an optional counts layer, but strip bulky metadata trees
    in ``.uns`` and related containers that older ``anndata`` versions may fail
    to deserialize.
    """
    keep_layers = {}
    if "counts" in adata.layers:
        keep_layers["counts"] = adata.layers["counts"].copy()

    adata.layers = keep_layers
    adata.uns = {}
    adata.obsm = {}
    adata.varm = {}
    adata.obsp = {}
    adata.varp = {}
    adata.raw = None
    return adata


def main() -> None:
    """Main CLI entry point to prepare h5ad file for real-data benchmarking."""
    parser = argparse.ArgumentParser(
        description="Prepare staged per-patient h5ad files for real-data benchmarking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--path-list", required=True, help="Excel path manifest (Sheet1).")
    parser.add_argument("--label-csv", required=True, help="CSV with cell barcode index and label column.")
    parser.add_argument("--label-column", default="cell_type", help="Column name in --label-csv to use as labels.")
    parser.add_argument("--samples", required=True, help="Comma-separated patient IDs to process.")
    parser.add_argument("--staging-dir", required=True, help="Directory to write staged per-patient h5ad files.")
    parser.add_argument(
        "--staged-manifest",
        default=None,
        help="Output text file listing staged h5ad paths (one per line).",
    )
    parser.add_argument("--patient-key", default="patient", help="obs column for patient labels.")
    parser.add_argument(
        "--pseudotime-key",
        default="s_local",
        help="obs column name written after pseudotime rescaling.",
    )
    parser.add_argument(
        "--cell-type-key",
        default="cell_type_final2",
        help="obs column name written from external labels.",
    )
    parser.add_argument(
        "--tumor-cell-types",
        default="LUAD,LUSC,SCLC-A,SCLC-N,EMT,FOXI1+ NonNE SCLC-N,NonNE SCLC",
        help="Comma-separated cell types to retain before staging.",
    )
    parser.add_argument(
        "--pseudotime-column-field",
        default="pseudotime_col",
        help="Column in Sheet1 that stores each patient's raw pseudotime column name.",
    )
    parser.add_argument("--clip-low", type=float, default=1.0, help="Lower percentile for clipping.")
    parser.add_argument("--clip-high", type=float, default=99.0, help="Upper percentile for clipping.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing staged files. If false, existing files are reused.",
    )

    args = parser.parse_args()

    samples = _parse_csv_list(args.samples)
    if not samples:
        raise ValueError("--samples resolved to an empty list.")

    tumor_cell_types = set(_parse_csv_list(args.tumor_cell_types))
    if not tumor_cell_types:
        raise ValueError("--tumor-cell-types resolved to an empty set.")

    os.makedirs(args.staging_dir, exist_ok=True)

    path_df = pd.read_excel(args.path_list, sheet_name="Sheet1", index_col=0)
    labels_df = pd.read_csv(args.label_csv, index_col=0)
    if args.label_column not in labels_df.columns:
        raise ValueError(f"Label CSV missing column {args.label_column!r}. Available: {labels_df.columns.tolist()}")
    cell_type_labels = labels_df[args.label_column]

    staged_paths: list[str] = []
    skipped: list[str] = []

    for patient in samples:
        if patient not in path_df.index:
            print(f"[{patient}] missing from path manifest; skipping")
            skipped.append(patient)
            continue

        out_path = os.path.join(args.staging_dir, f"{patient}.h5ad")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[{patient}] staged file exists, reusing: {out_path}")
            staged_paths.append(out_path)
            continue

        adata_path = path_df.loc[patient, "path"]
        pseudotime_col = path_df.loc[patient, args.pseudotime_column_field]

        adata = sc.read_h5ad(adata_path)

        shared = adata.obs_names.intersection(cell_type_labels.index)
        if len(shared) == 0:
            print(f"[{patient}] no overlapping barcodes with label CSV; skipping")
            skipped.append(patient)
            continue

        adata = adata[shared].copy()
        adata.obs[args.cell_type_key] = cell_type_labels.loc[shared].values

        is_tumor = adata.obs[args.cell_type_key].isin(tumor_cell_types)
        adata = adata[is_tumor].copy()
        if adata.n_obs == 0:
            print(f"[{patient}] no cells left after tumour filter; skipping")
            skipped.append(patient)
            continue

        if pseudotime_col not in adata.obs.columns:
            print(f"[{patient}] missing pseudotime column {pseudotime_col!r}; skipping")
            skipped.append(patient)
            continue

        raw_pt = pd.to_numeric(adata.obs[pseudotime_col], errors="coerce")
        valid = raw_pt.dropna()
        if valid.empty:
            print(f"[{patient}] pseudotime is all NA after coercion; skipping")
            skipped.append(patient)
            continue

        vmin, vmax = np.percentile(valid, [args.clip_low, args.clip_high])
        if vmax == vmin:
            print(f"[{patient}] clipped pseudotime is constant; skipping")
            skipped.append(patient)
            continue

        adata.obs[args.pseudotime_key] = (np.clip(raw_pt, vmin, vmax) - vmin) / (vmax - vmin)
        adata.obs[args.patient_key] = patient

        adata = _make_portable_adata(adata)

        adata.write_h5ad(out_path)
        staged_paths.append(out_path)
        print(
            f"[{patient}] staged {adata.n_obs} cells -> {out_path} "
            f"| types={dict(adata.obs[args.cell_type_key].value_counts())}"
        )

    manifest_path = (
        args.staged_manifest if args.staged_manifest is not None else os.path.join(args.staging_dir, "staged_paths.txt")
    )
    with open(manifest_path, "w", encoding="utf-8") as fh:
        for p in staged_paths:
            fh.write(str(Path(p).resolve()) + "\n")

    print("\nPreparation complete")
    print(f"  staged patients: {len(staged_paths)}")
    print(f"  skipped patients: {len(skipped)}")
    print(f"  staged manifest: {manifest_path}")
    if skipped:
        print(f"  skipped list: {skipped}")


if __name__ == "__main__":
    main()
