"""Refit v1 and v2 on the HTAN cohort and cache both barycenters for plotting.

Mirrors scripts/v2/real_data.py exactly (same config, same input, same keys) so the
cached arrays correspond to the numbers already recorded in validation/v2/real_data.json.
"""

import json
import time
import warnings
from pathlib import Path

import anndata as ad
import numpy as np

from scdebussy import align, tl

H5AD = "/scratch/chanj3/wangm10/HTAN/results/aligned_scdebussy_preprocessed_v3.h5ad"
OUT = Path(
    "/tmp/claude-164119608/-data1-chanj3-wangm10-scDeBussy/c71890ea-55a9-455a-9c01-c8686eb93d37/scratchpad/htan_barycenters.npz"
)
CONFIG = json.loads(Path("/tmp/scdebussy-v2/scripts/v2/real_data_config.json").read_text())

t0 = time.monotonic()
print("loading", H5AD, flush=True)
data = ad.read_h5ad(H5AD)
print(f"  {data.n_obs} cells x {data.n_vars} genes  ({time.monotonic() - t0:.0f}s)", flush=True)

# ---- v1, same options as real_data.py -------------------------------------------------
v1_options = {
    "patient_key": "patient",
    "pseudotime_key": "s_local",
    "n_bins": CONFIG["n_landmarks"],
    "bandwidth": 0.1,
    "gamma": 0.1,
    "max_iter": 20,
    "tol": 1e-4,
    "use_soft_em": True,
    "strict_em": False,
    "verbose": False,
    "key_added": "v1_pos",
    "barycenter_key": "v1_bary",
}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    t = time.monotonic()
    tl.scDeBussy(data, **v1_options)
    print(f"v1 done ({time.monotonic() - t:.0f}s)", flush=True)

    # ---- v2 ---------------------------------------------------------------------------
    t = time.monotonic()
    result = align.fit(data, **CONFIG)
    projection = align.project(result)
    print(f"v2 done ({time.monotonic() - t:.0f}s) converged={result.converged} iters={len(result.history)}", flush=True)

v1_bary = np.asarray(data.uns["v1_bary"]["expression"], dtype=float)
v1_genes = np.asarray(data.uns["v1_bary"].get("genes_used", data.var_names), dtype=str)
v2_bary = np.asarray(result.barycenter, dtype=float)
v2_genes = np.asarray(result.genes, dtype=str)

np.savez_compressed(
    OUT,
    v1_bary=v1_bary,
    v1_genes=v1_genes,
    v2_bary=v2_bary,
    v2_genes=v2_genes,
    v2_identified=np.asarray(result.identified),
    v2_weakly=np.asarray(result.weakly_identified),
    v1_pos=np.asarray(data.obs["v1_pos"], dtype=float),
    v2_pos=np.asarray(projection.position, dtype=float),
    v2_unprojected=np.asarray(projection.unprojected, dtype=float),
    v2_status=np.asarray(projection.status, dtype=str),
    cell_patient=np.asarray(data.obs["patient"], dtype=str),
    cell_type=np.asarray(data.obs["cell_type_final2"], dtype=str),
)
print(f"wrote {OUT}  v1_bary={v1_bary.shape} v2_bary={v2_bary.shape}", flush=True)
print(f"total {time.monotonic() - t0:.0f}s", flush=True)
