"""Rebuild the HTAN_demo staged manifest with corrected per-patient inputs.

RU151, RU1303, and RU942's staged copies under `preprocessed/` were stale
(dated 2026-04-15) relative to the reprocessed per-patient files listed in
`figure_4_adata_path.xlsx` (dated 2026-07-06), which carry a materially
different `palantir_pseudotime_slalom` for RU151/RU1303 (correlation with the
old staged pseudotime was only ~0.6) and a corrected `new_cell_type` label.
Separately, several patients' staged files (RU1444, RU151, RU1303, RU942) had
LUSC / NonNE SCLC cells mixed in that don't belong to the LUAD -> SCLC-A ->
SCLC-N trajectory this notebook models.

This script writes a corrected copy of all 9 patients (not just the 3 being
swapped) to `preprocessed_v2/`, leaving the original `preprocessed/` and
`staged_paths.txt` untouched:
  - subsets every patient to cell_type_final2 in {LUAD, SCLC-A, SCLC-N}
  - recomputes s_local as min-max normalized palantir_pseudotime_slalom over
    the *retained* cells only (subsetting before renormalizing matters: a
    patient who loses cells at the pseudotime extremes would otherwise end up
    with a truncated s_local range that isn't comparable across patients)
  - for RU151/RU1303/RU942, sources palantir_pseudotime_slalom and cell type
    (via `new_cell_type`, renamed to `cell_type_final2` -- the same-named
    column in the new files is stale and doesn't match) from the updated
    files in figure_4_adata_path.xlsx instead of the old staged copies
  - slims each output to just X, the `counts` layer, and the three obs
    columns the pipeline needs (patient, s_local, cell_type_final2), as plain
    strings rather than pandas Categoricals, to avoid the categorical-dtype
    write issues seen previously with leftover columns like `phenograph`

Verified: old and new source files have identical `.X` and `counts` values
for the same cells (only pseudotime/cell-type annotations differ), and share
the full 26660-gene panel, so mixing old- and new-format patients in one
concat is safe.
"""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

OUT_DIR = Path("/scratch/chanj3/wangm10/HTAN_transformed_combined/preprocessed_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

KEEP_CATEGORIES = {"LUAD", "SCLC-A", "SCLC-N"}

OLD_STAGED_DIR = Path("/scratch/chanj3/wangm10/HTAN_transformed_combined/preprocessed")
OLD_FORMAT_PATIENTS = ["RU1083", "RU263", "RU1444", "RU581", "RU831", "RU1646"]

NEW_SOURCE_PATHS = {
    "RU151": "/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.SCLC_NSCLC.062124/individual.021626/RU151.new_embedding.final_ownct_070626.h5ad",
    "RU1303": "/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.SCLC_NSCLC.062124/individual.021626/RU1303.new_embedding.final_070626.h5ad",
    "RU942": "/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.SCLC_NSCLC.062124/individual.021626/RU942.new_embedding.final_070626.h5ad",
}

written_paths = []


def slim_write(adata, patient, out_path):
    """Keep only what the scDeBussy pipeline needs: X, counts layer, and the three
    essential obs columns as plain strings (avoids categorical-dtype write issues
    seen previously with leftover columns like 'phenograph').
    """
    assert "counts" in adata.layers, f"{patient}: missing 'counts' layer"
    for col in ("patient", "s_local", "cell_type_final2"):
        assert col in adata.obs.columns, f"{patient}: missing obs column {col!r}"
    slim = ad.AnnData(
        X=adata.X,
        obs=pd.DataFrame(
            {
                "patient": adata.obs["patient"].astype(str).to_numpy(),
                "s_local": adata.obs["s_local"].to_numpy(dtype=float),
                "cell_type_final2": adata.obs["cell_type_final2"].astype(str).to_numpy(),
            },
            index=adata.obs_names,
        ),
        var=pd.DataFrame(index=adata.var_names),
    )
    slim.layers["counts"] = adata.layers["counts"]
    slim.write_h5ad(out_path)
    return slim


def subset_and_renormalize(a, patient, cell_type_col):
    """Subset to LUAD/SCLC-A/SCLC-N, then recompute s_local as min-max normalized
    palantir_pseudotime_slalom *over the retained cells only* -- otherwise a patient
    who loses cells at the pseudotime extremes (e.g. off-trajectory LUSC/NonNE SCLC
    cells sitting at one end) ends up with a truncated s_local range that isn't
    comparable to patients whose cells were already pure LUAD/SCLC-A/SCLC-N.
    """
    before = a.n_obs
    a.obs["cell_type_final2"] = a.obs[cell_type_col].astype(str)
    mask = a.obs["cell_type_final2"].isin(KEEP_CATEGORIES)
    a = a[mask].copy()
    pt = a.obs["palantir_pseudotime_slalom"].to_numpy(dtype=float)
    pt_min, pt_max = np.nanmin(pt), np.nanmax(pt)
    a.obs["s_local"] = (pt - pt_min) / (pt_max - pt_min)
    print(f"  {before} -> {a.n_obs} cells after LUAD/SCLC-A/SCLC-N subset")
    print(f"  s_local range after re-normalization: [{a.obs['s_local'].min():.3f}, {a.obs['s_local'].max():.3f}]")
    return a


for p in OLD_FORMAT_PATIENTS:
    print(f"--- {p} (old-format staged file) ---")
    a = ad.read_h5ad(OLD_STAGED_DIR / f"{p}.h5ad")
    a = subset_and_renormalize(a, p, cell_type_col="cell_type_final2")
    out_path = OUT_DIR / f"{p}.h5ad"
    slim_write(a, p, out_path)
    written_paths.append(out_path)
    del a

for p, src in NEW_SOURCE_PATHS.items():
    print(f"--- {p} (new source file, new_cell_type -> cell_type_final2) ---")
    a = ad.read_h5ad(src)
    a = subset_and_renormalize(a, p, cell_type_col="new_cell_type")
    out_path = OUT_DIR / f"{p}.h5ad"
    slim_write(a, p, out_path)
    written_paths.append(out_path)
    del a

manifest_path = OUT_DIR / "staged_paths_v2.txt"
with open(manifest_path, "w") as f:
    for p in written_paths:
        f.write(str(p) + "\n")

print()
print("Manifest written to:", manifest_path)
for p in written_paths:
    print(" ", p)
