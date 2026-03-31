"""Method adapter registry for benchmark alignment methods.

Each adapter receives an AnnData object plus method parameters and must return
an adapter result dict with:
    - evaluation_mode: str ("single_axis" or "pairwise_only")
    - aligned_key: str | None
    - method_params: dict (JSON-serialisable)
    - method_meta: dict (JSON-serialisable)
    - unsupervised_method: dict (JSON-serialisable, optional)

This keeps `run_scenario.py` method-agnostic.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import tempfile
from collections.abc import Callable
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Any

import numpy as np

import scdebussy.tl as tl

MethodFn = Callable[[object, dict], dict]
_METHOD_REGISTRY: dict[str, MethodFn] = {}
_G2G_MODULES: tuple[Any, Any] | None = None


def _safe_minmax_scale(vals: np.ndarray) -> np.ndarray:
    """Scale a 1D array to [0, 1] with stable handling of flat vectors."""
    arr = np.asarray(vals, dtype=float)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    denom = hi - lo
    if denom <= 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / denom


def _to_dense_matrix(X) -> np.ndarray:
    """Return adata.X as dense float ndarray."""
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=float)


def _import_genes2genes_modules() -> tuple[Any, Any]:
    """Lazy import genes2genes modules only when needed."""
    global _G2G_MODULES
    if _G2G_MODULES is not None:
        return _G2G_MODULES

    try:
        from genes2genes import ClusterUtils as g2g_cluster_utils
        from genes2genes import Main as g2g_main
    except Exception as exc:
        raise ImportError(
            "genes2genes is required for the genes2genes benchmark adapter. "
            "Install it in the active Python environment."
        ) from exc

    _G2G_MODULES = (g2g_main, g2g_cluster_utils)
    return _G2G_MODULES


def _ensure_genes2genes_anndata_compat() -> None:
    """Patch anndata view symbols expected by genes2genes when missing.

    genes2genes references anndata internals that moved across anndata
    versions (e.g. SparseCSCView). If absent, create harmless placeholder
    classes so isinstance checks remain valid and non-crashing.
    """
    try:
        import anndata
    except Exception:  # noqa: BLE001
        return

    try:
        views = anndata._core.views  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return

    if not hasattr(views, "SparseCSCView"):
        views.SparseCSCView = type("SparseCSCView", (), {})


@contextmanager
def _maybe_silence_output(verbose: bool):
    """Suppress noisy method logs unless verbose=True."""
    if verbose:
        yield
        return

    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


def _resolve_g2g_gene_list(adata, params: dict) -> list[str]:
    """Resolve a stable, valid gene list for genes2genes."""
    explicit = params.get("gene_list")
    if explicit is not None:
        if not isinstance(explicit, list) or not explicit:
            raise ValueError("method_params.gene_list must be a non-empty list of gene names.")
        explicit_set = {str(g) for g in explicit}
        genes = [str(g) for g in adata.var_names if str(g) in explicit_set]
        if not genes:
            raise ValueError("No method_params.gene_list genes found in adata.var_names.")
        return genes

    gene_list_mode = str(params.get("gene_list_mode", "all_genes"))
    all_genes = [str(g) for g in adata.var_names]
    if not all_genes:
        raise ValueError("adata has no var_names for genes2genes alignment.")

    if gene_list_mode == "all_genes":
        return all_genes

    if gene_list_mode == "hvg":
        if "highly_variable" not in adata.var:
            # Benchmark simulations do not precompute HVGs; derive them on demand.
            X = adata.X
            if hasattr(X, "toarray"):
                X = X.toarray()
            X = np.asarray(X, dtype=float)
            if X.ndim != 2:
                raise ValueError("Expected 2D adata.X when deriving HVGs.")

            n_vars = int(X.shape[1])
            if n_vars == 0:
                raise ValueError("Cannot derive HVGs from an empty gene matrix.")

            n_top = int(params.get("hvg_n_genes", min(2000, n_vars)))
            n_top = max(1, min(n_top, n_vars))

            # Use variance ranking as a lightweight HVG proxy.
            variances = np.var(X, axis=0)
            variances = np.nan_to_num(variances, nan=0.0, posinf=0.0, neginf=0.0)
            top_idx = np.argpartition(-variances, kth=n_top - 1)[:n_top]
            hvg_mask = np.zeros(n_vars, dtype=bool)
            hvg_mask[top_idx] = True
            adata.var["highly_variable"] = hvg_mask

        hvg_mask = adata.var["highly_variable"].to_numpy(dtype=bool)
        if not np.any(hvg_mask):
            raise ValueError("gene_list_mode='hvg' selected zero genes.")
        genes = [str(g) for g, keep in zip(adata.var_names, hvg_mask, strict=True) if keep]
        if not genes:
            raise ValueError("gene_list_mode='hvg' selected zero genes.")
        return genes

    raise ValueError("gene_list_mode must be one of {'all_genes', 'hvg'}.")


def _g2g_tracked_path_to_query_mapping(
    tracked_path,
    *,
    query_interp_points: np.ndarray,
    ref_interp_points: np.ndarray,
    query_cell_pseudotime: np.ndarray,
) -> np.ndarray:
    """Map query-cell pseudotime to reference pseudotime via G2G tracked path."""
    query_interp_points = np.asarray(query_interp_points, dtype=float)
    ref_interp_points = np.asarray(ref_interp_points, dtype=float)
    query_cell_pseudotime = np.asarray(query_cell_pseudotime, dtype=float)

    if query_interp_points.size == 0 or ref_interp_points.size == 0:
        return query_cell_pseudotime.copy()

    path = np.asarray(tracked_path, dtype=float)
    if path.ndim != 2 or path.shape[0] == 0 or path.shape[1] < 2:
        return query_cell_pseudotime.copy()

    # ClusterUtils returns a backward path from [T_len, S_len] to [0, 0].
    path = path[:, :2][::-1, :]
    q_bins = path[:, 0].astype(int)
    r_bins = path[:, 1].astype(int)

    valid = (q_bins > 0) & (r_bins > 0)
    if not np.any(valid):
        return query_cell_pseudotime.copy()

    q_idx = q_bins[valid] - 1
    r_idx = r_bins[valid] - 1
    in_bounds = (q_idx >= 0) & (q_idx < query_interp_points.size) & (r_idx >= 0) & (r_idx < ref_interp_points.size)
    if not np.any(in_bounds):
        return query_cell_pseudotime.copy()

    q_pts = query_interp_points[q_idx[in_bounds]]
    r_pts = ref_interp_points[r_idx[in_bounds]]
    order = np.argsort(q_pts)
    q_pts = q_pts[order]
    r_pts = r_pts[order]

    # Collapse repeated query bins (W states) by averaging mapped ref bins.
    unique_q, inverse = np.unique(q_pts, return_inverse=True)
    mean_r = np.zeros(unique_q.size, dtype=float)
    counts = np.zeros(unique_q.size, dtype=float)
    for idx, ridx in enumerate(inverse):
        mean_r[ridx] += r_pts[idx]
        counts[ridx] += 1.0
    mean_r /= np.maximum(counts, 1.0)
    mean_r = np.maximum.accumulate(mean_r)

    if unique_q.size == 1:
        return np.full_like(query_cell_pseudotime, float(mean_r[0]), dtype=float)

    return np.interp(
        query_cell_pseudotime,
        unique_q,
        mean_r,
        left=float(mean_r[0]),
        right=float(mean_r[-1]),
    )


def _prepare_g2g_adata_slice(adata_slice, pseudotime_key: str):
    """Prepare an AnnData slice for genes2genes compatibility."""
    out = adata_slice.copy()

    # genes2genes expects X views with .todense() support after internal slicing.
    try:
        from scipy.sparse import csr_matrix

        X = out.X
        if hasattr(X, "toarray"):
            dense = np.asarray(X.toarray(), dtype=float)
        else:
            dense = np.asarray(X, dtype=float)
        out.X = csr_matrix(dense)
    except Exception:  # noqa: BLE001
        # Fall back to dense array if scipy sparse conversion is unavailable.
        X = out.X
        if hasattr(X, "toarray"):
            out.X = np.asarray(X.toarray(), dtype=float)
        else:
            out.X = np.asarray(X, dtype=float)

    out.obs["time"] = out.obs[pseudotime_key].to_numpy(dtype=float)
    return out


def _get_patient_ids(adata, patient_key: str) -> list[str]:
    if patient_key not in adata.obs:
        raise ValueError(f"{patient_key!r} is missing from adata.obs")
    return sorted(str(pid) for pid in np.unique(adata.obs[patient_key].to_numpy()))


def _subset_adata_by_patients(adata, patient_key: str, patient_ids: list[str]):
    mask = np.isin(adata.obs[patient_key].to_numpy(dtype=str), np.asarray(patient_ids, dtype=str))
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError(f"No cells found for patient_ids={patient_ids}")
    return adata[indices].copy(), indices


def _select_reference_patient(
    adata, patient_key: str, pseudotime_key: str, reference_policy: str, explicit: str | None
) -> str:
    patient_ids = _get_patient_ids(adata, patient_key)
    if reference_policy == "explicit":
        if explicit is None:
            raise ValueError("reference_patient must be provided when reference_policy='explicit'.")
        if explicit not in patient_ids:
            raise ValueError(f"Unknown reference_patient={explicit!r}. Available: {patient_ids}")
        return explicit

    if reference_policy != "medoid":
        raise ValueError("reference_policy must be 'medoid' or 'explicit'.")

    vals = adata.obs[pseudotime_key].to_numpy(dtype=float)
    patients = adata.obs[patient_key].to_numpy(dtype=str)
    bins = np.linspace(0.0, 1.0, 33)
    hists = {}
    for pid in patient_ids:
        arr = np.clip(vals[patients == pid], 0.0, 1.0)
        hist, _ = np.histogram(arr, bins=bins, density=False)
        hist = hist.astype(float)
        hist /= hist.sum() + 1e-12
        hists[pid] = hist

    best_pid = None
    best_score = float("inf")
    for pid in patient_ids:
        score = float(np.mean([np.mean(np.abs(hists[pid] - hists[qid])) for qid in patient_ids if qid != pid]))
        if score < best_score:
            best_score = score
            best_pid = pid
    return str(best_pid)


def _write_cellalign_bridge_inputs(
    adata,
    *,
    input_dir: str,
    patient_key: str,
    pseudotime_key: str,
    bridge_mode: str,
    source_patient: str | None = None,
    target_patient: str | None = None,
) -> dict:
    """Write standard bridge input files for the cellalign R adapter.

    Files written:
      - expression.csv : rows=cells, cols=genes
      - obs.csv        : cell metadata with cell_id, patient, s_local
      - config.json    : schema and key metadata
    """
    expr_path = os.path.join(input_dir, "expression.csv")
    obs_path = os.path.join(input_dir, "obs.csv")
    config_path = os.path.join(input_dir, "config.json")

    X = _to_dense_matrix(adata.X)
    n_cells, n_genes = X.shape

    # Expression matrix with gene headers for easier ingestion on R side
    gene_names = [str(g) for g in adata.var_names] if getattr(adata, "var_names", None) is not None else []
    if len(gene_names) != n_genes:
        gene_names = [f"gene_{i}" for i in range(n_genes)]

    with open(expr_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cell_id", *gene_names])
        for i in range(n_cells):
            writer.writerow([i, *X[i, :].tolist()])

    if patient_key not in adata.obs:
        raise ValueError(f"cellalign adapter expected {patient_key!r} in adata.obs")
    if pseudotime_key not in adata.obs:
        raise ValueError(f"cellalign adapter expected {pseudotime_key!r} in adata.obs")

    patients = adata.obs[patient_key].to_numpy()
    s_local = adata.obs[pseudotime_key].to_numpy(dtype=float)

    with open(obs_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cell_id", "patient", "s_local"])
        for i in range(n_cells):
            writer.writerow([i, str(patients[i]), float(s_local[i])])

    config = {
        "schema_version": 1,
        "n_cells": int(n_cells),
        "n_genes": int(n_genes),
        "patient_key": patient_key,
        "pseudotime_key": pseudotime_key,
        "bridge_mode": bridge_mode,
        "source_patient": source_patient,
        "target_patient": target_patient,
        "input_files": {
            "expression": "expression.csv",
            "obs": "obs.csv",
        },
        "output_files": {
            "aligned": "aligned.csv",
            "metrics": "metrics.json",
        },
        "aligned_schema": {
            "required_columns": ["cell_id", "aligned_pseudotime"],
            "cell_id_type": "int",
            "aligned_pseudotime_type": "float",
        },
    }
    with open(config_path, "w") as fh:
        json.dump(config, fh, indent=2)

    return {
        "n_cells": int(n_cells),
        "n_genes": int(n_genes),
        "expression_path": expr_path,
        "obs_path": obs_path,
        "config_path": config_path,
    }


def _read_cellalign_bridge_outputs(output_dir: str, n_cells: int) -> tuple[np.ndarray, dict]:
    """Read and validate outputs from the R bridge.

    Required output:
      - aligned.csv with columns: cell_id, aligned_pseudotime
    Optional output:
      - metrics.json (method diagnostics)
    """
    aligned_path = os.path.join(output_dir, "aligned.csv")
    metrics_path = os.path.join(output_dir, "metrics.json")

    if not os.path.exists(aligned_path):
        raise ValueError("cellalign bridge did not produce aligned.csv. Expected columns: cell_id, aligned_pseudotime")

    rows: list[tuple[int, float]] = []
    with open(aligned_path, newline="") as fh:
        reader = csv.DictReader(fh)
        needed = {"cell_id", "aligned_pseudotime"}
        if reader.fieldnames is None or not needed.issubset(set(reader.fieldnames)):
            raise ValueError("aligned.csv schema mismatch. Required header columns: cell_id, aligned_pseudotime")
        for row in reader:
            cid = int(row["cell_id"])
            val = float(row["aligned_pseudotime"])
            rows.append((cid, val))

    if len(rows) != n_cells:
        raise ValueError(f"aligned.csv must have exactly {n_cells} rows, got {len(rows)}.")

    aligned = np.full(n_cells, np.nan, dtype=float)
    seen = set()
    for cid, val in rows:
        if cid < 0 or cid >= n_cells:
            raise ValueError(f"Invalid cell_id in aligned.csv: {cid}")
        if cid in seen:
            raise ValueError(f"Duplicate cell_id in aligned.csv: {cid}")
        seen.add(cid)
        aligned[cid] = val

    if np.any(~np.isfinite(aligned)):
        raise ValueError("aligned.csv contains non-finite aligned_pseudotime values.")

    meta = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as fh:
            loaded = json.load(fh)
        if isinstance(loaded, dict):
            meta = loaded

    return aligned, meta


def _run_cellalign_bridge_call(
    adata,
    *,
    patient_key: str,
    pseudotime_key: str,
    bridge_script_abs: str,
    rscript_bin: str,
    timeout_s: int,
    num_pts: int,
    win_sz: float,
    dist_method: str,
    mode: str,
    source_patient: str | None = None,
    target_patient: str | None = None,
) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"cellalign_{mode}_") as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        io_meta = _write_cellalign_bridge_inputs(
            adata,
            input_dir=input_dir,
            patient_key=patient_key,
            pseudotime_key=pseudotime_key,
            bridge_mode=mode,
            source_patient=source_patient,
            target_patient=target_patient,
        )

        cmd = [
            rscript_bin,
            bridge_script_abs,
            "--input_dir",
            input_dir,
            "--output_dir",
            output_dir,
            "--mode",
            mode,
            "--num_pts",
            str(num_pts),
            "--win_sz",
            str(win_sz),
            "--dist_method",
            dist_method,
        ]
        if source_patient is not None:
            cmd.extend(["--source_patient", source_patient])
        if target_patient is not None:
            cmd.extend(["--target_patient", target_patient])

        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"cellalign bridge failed. Command: {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )

        aligned, bridge_metrics = _read_cellalign_bridge_outputs(output_dir, io_meta["n_cells"])
        return {
            "aligned": aligned,
            "bridge_metrics": bridge_metrics,
            "bridge_stdout_tail": proc.stdout[-2000:],
            "io_meta": io_meta,
            "mode": mode,
            "num_pts": num_pts,
            "win_sz": win_sz,
            "dist_method": dist_method,
            "source_patient": source_patient,
            "target_patient": target_patient,
        }


def _make_pair_label(source_patient: str, target_patient: str) -> str:
    return f"{source_patient}__to__{target_patient}"


def _pairwise_summary(pair_records: list[dict]) -> dict:
    n_pairs = len(pair_records)
    successful = sum(int(rec.get("status") == "ok") for rec in pair_records)
    return {
        "n_pairs_attempted": int(n_pairs),
        "n_pairs_succeeded": int(successful),
        "pair_success_rate": float(successful / n_pairs) if n_pairs else 0.0,
    }


def register_method(name: str):
    """Decorator to register a benchmark method adapter by name."""

    def _decorator(func: MethodFn) -> MethodFn:
        if name in _METHOD_REGISTRY:
            raise ValueError(f"Method {name!r} is already registered.")
        _METHOD_REGISTRY[name] = func
        return func

    return _decorator


def available_methods() -> list[str]:
    """Return sorted method names available in this process."""
    return sorted(_METHOD_REGISTRY)


def run_method(method: str, adata, method_params: dict | None = None) -> dict:
    """Run a registered method and return its adapter result."""
    if method not in _METHOD_REGISTRY:
        raise ValueError(f"Unknown method: {method!r}. Available: {available_methods()}")

    params = dict(method_params or {})
    result = _METHOD_REGISTRY[method](adata, params)

    if not isinstance(result, dict):
        raise TypeError("Adapter must return a dict.")
    evaluation_mode = result.get("evaluation_mode", "single_axis")
    if evaluation_mode not in {"single_axis", "pairwise_only"}:
        raise ValueError(f"Unknown evaluation_mode returned by adapter: {evaluation_mode!r}")

    aligned_key = result.get("aligned_key")
    if evaluation_mode == "single_axis":
        if not aligned_key:
            raise ValueError("Single-axis adapter result missing required key 'aligned_key'.")
        if aligned_key not in adata.obs:
            raise ValueError(f"Adapter reported aligned_key={aligned_key!r} but the column is missing from adata.obs.")

        vals = adata.obs[aligned_key].to_numpy(dtype=float)
        if np.any(~np.isfinite(vals)):
            raise ValueError(f"Aligned pseudotime in {aligned_key!r} contains non-finite values.")

    out = {
        "evaluation_mode": evaluation_mode,
        "aligned_key": aligned_key,
        "method_params": result.get("method_params", params),
        "method_meta": result.get("method_meta", {}),
        "unsupervised_method": result.get("unsupervised_method", {}),
    }
    return out


@register_method("scdebussy")
def run_scdebussy(adata, method_params: dict) -> dict:
    """Run scDeBussy and return adapter metadata."""
    params = dict(method_params)
    tl.scDeBussy(adata, **params)

    aligned_key = params.get("key_added", "aligned_pseudotime")
    barycenter_key = params.get("barycenter_key", "barycenter")
    barycenter_meta = adata.uns.get(barycenter_key, {}) if hasattr(adata, "uns") else {}
    em_convergence = barycenter_meta.get("em_convergence", {}) if isinstance(barycenter_meta, dict) else {}

    unsupervised_method = {}
    if isinstance(em_convergence, dict):
        if "final_iteration" in em_convergence:
            unsupervised_method["em_final_iteration"] = em_convergence["final_iteration"]
        if "converged" in em_convergence:
            unsupervised_method["em_converged"] = bool(em_convergence["converged"])
        if "final_cost" in em_convergence:
            unsupervised_method["em_final_cost"] = em_convergence["final_cost"]

    return {
        "evaluation_mode": "single_axis",
        "aligned_key": aligned_key,
        "method_params": params,
        "method_meta": {
            "barycenter_key": barycenter_key,
            "backend": "python",
            "em_convergence": em_convergence,
        },
        "unsupervised_method": unsupervised_method,
    }


@register_method("identity")
def run_identity(adata, method_params: dict) -> dict:
    """Baseline adapter: copy local pseudotime to an output key.

    Useful as a smoke-test method and baseline comparator within the same
    method framework.
    """
    params = dict(method_params)
    input_key = params.get("pseudotime_key", "s_local")
    key_added = params.get("key_added", "aligned_pseudotime_identity")

    if input_key not in adata.obs:
        raise ValueError(f"identity method expected {input_key!r} in adata.obs")

    adata.obs[key_added] = adata.obs[input_key].to_numpy(dtype=float)
    return {
        "evaluation_mode": "single_axis",
        "aligned_key": key_added,
        "method_params": params,
        "method_meta": {
            "source_key": input_key,
            "backend": "python",
        },
        "unsupervised_method": {},
    }


def _cellalign_common_params(method_params: dict) -> dict:
    params = dict(method_params)
    patient_key = params.get("patient_key", "patient")
    pseudotime_key = params.get("pseudotime_key", "s_local")
    bridge_script = params.get("bridge_script", "scripts/benchmark/cellalign_bridge.R")
    rscript_bin = params.get("rscript_bin", "Rscript")
    timeout_s = int(params.get("timeout_s", 3600))
    dry_run = bool(params.get("dry_run", False))
    num_pts = int(params.get("num_pts", 200))
    win_sz = float(params.get("win_sz", 0.1))
    dist_method = str(params.get("dist_method", "Euclidean"))
    bridge_script_abs = os.path.abspath(bridge_script)

    if num_pts < 3:
        raise ValueError("cellalign num_pts must be at least 3.")
    if not np.isfinite(win_sz) or win_sz <= 0:
        raise ValueError("cellalign win_sz must be a positive finite number.")
    if dist_method not in {"Euclidean", "Correlation"}:
        raise ValueError("cellalign dist_method must be one of {'Euclidean', 'Correlation'}.")

    return {
        "params": params,
        "patient_key": patient_key,
        "pseudotime_key": pseudotime_key,
        "rscript_bin": rscript_bin,
        "timeout_s": timeout_s,
        "dry_run": dry_run,
        "num_pts": num_pts,
        "win_sz": win_sz,
        "dist_method": dist_method,
        "bridge_script_abs": bridge_script_abs,
    }


def _genes2genes_common_params(method_params: dict) -> dict:
    params = dict(method_params)
    patient_key = params.get("patient_key", "patient")
    pseudotime_key = params.get("pseudotime_key", "s_local")
    n_bins = int(params.get("n_bins", 50))
    verbose = bool(params.get("verbose", False))
    concurrent = bool(params.get("concurrent", False))
    n_processes = params.get("n_processes")
    if n_processes is not None:
        n_processes = int(n_processes)
        if n_processes <= 0:
            raise ValueError("n_processes must be a positive integer when provided.")

    return {
        "params": params,
        "patient_key": patient_key,
        "pseudotime_key": pseudotime_key,
        "n_bins": n_bins,
        "verbose": verbose,
        "concurrent": concurrent,
        "n_processes": n_processes,
    }


@register_method("genes2genes_fixed_reference")
def run_genes2genes_fixed_reference(adata, method_params: dict) -> dict:
    """Run Genes2Genes against a fixed reference and emit one axis per cell."""
    params = dict(method_params)
    patient_key = params.get("patient_key", "patient")
    pseudotime_key = params.get("pseudotime_key", "s_local")
    key_added = params.get("key_added", "aligned_pseudotime_genes2genes_fixed_reference")
    reference_policy = params.get("reference_policy", "medoid")
    reference_patient = params.get("reference_patient")

    n_bins = int(params.get("n_bins", 50))
    verbose = bool(params.get("verbose", False))
    concurrent = bool(params.get("concurrent", False))
    n_processes = params.get("n_processes")
    if n_processes is not None:
        n_processes = int(n_processes)
        if n_processes <= 0:
            raise ValueError("n_processes must be a positive integer when provided.")

    if patient_key not in adata.obs:
        raise ValueError(f"genes2genes method expected {patient_key!r} in adata.obs")
    if pseudotime_key not in adata.obs:
        raise ValueError(f"genes2genes method expected {pseudotime_key!r} in adata.obs")

    gene_list = _resolve_g2g_gene_list(adata, params)
    g2g_main, g2g_cluster_utils = _import_genes2genes_modules()
    _ensure_genes2genes_anndata_compat()

    reference_patient = _select_reference_patient(
        adata,
        patient_key=patient_key,
        pseudotime_key=pseudotime_key,
        reference_policy=reference_policy,
        explicit=reference_patient,
    )
    patient_ids = _get_patient_ids(adata, patient_key)
    patients = adata.obs[patient_key].to_numpy(dtype=str)

    aligned_global = np.full(adata.n_obs, np.nan, dtype=float)
    ref_mask_global = patients == reference_patient
    aligned_global[ref_mask_global] = adata.obs[pseudotime_key][ref_mask_global].to_numpy(dtype=float)

    pair_records = []
    for pid in patient_ids:
        if pid == reference_patient:
            continue

        rec: dict[str, object] = {"pair": _make_pair_label(pid, reference_patient), "status": "ok"}
        try:
            src_idx = np.flatnonzero(patients == pid)
            ref_idx = np.flatnonzero(ref_mask_global)
            if src_idx.size == 0 or ref_idx.size == 0:
                raise ValueError("Source or reference subset is empty.")

            adata_src = _prepare_g2g_adata_slice(adata[src_idx], pseudotime_key)
            adata_ref = _prepare_g2g_adata_slice(adata[ref_idx], pseudotime_key)

            with _maybe_silence_output(verbose):
                aligner = g2g_main.RefQueryAligner(adata_ref, adata_src, gene_list, n_bins)
                aligner.align_all_pairs(concurrent=concurrent, n_processes=n_processes)
                _, tracked_path = g2g_cluster_utils.get_cluster_average_alignments(aligner, gene_list)

            mapped = _g2g_tracked_path_to_query_mapping(
                tracked_path,
                query_interp_points=np.asarray(aligner.TrajInt_Q.interpolation_points, dtype=float),
                ref_interp_points=np.asarray(aligner.TrajInt_R.interpolation_points, dtype=float),
                query_cell_pseudotime=adata_src.obs[pseudotime_key].to_numpy(dtype=float),
            )
            aligned_global[src_idx] = mapped
            rec.update(
                {
                    "n_cells_source": int(src_idx.size),
                    "n_cells_reference": int(ref_idx.size),
                }
            )
        except Exception as exc:  # noqa: BLE001
            rec.update({"status": "error", "error": str(exc)})
        pair_records.append(rec)

    missing = ~np.isfinite(aligned_global)
    if np.any(missing):
        aligned_global[missing] = adata.obs[pseudotime_key].to_numpy(dtype=float)[missing]

    adata.obs[key_added] = _safe_minmax_scale(aligned_global)
    summary = _pairwise_summary(pair_records)

    gene_list_mode = str(params.get("gene_list_mode", "all_genes"))
    return {
        "evaluation_mode": "single_axis",
        "aligned_key": key_added,
        "method_params": params,
        "method_meta": {
            "backend": "python",
            "library": "genes2genes",
            "reference_policy": reference_policy,
            "reference_patient": reference_patient,
            "n_bins": int(n_bins),
            "gene_list_mode": gene_list_mode,
            "n_genes_aligned": int(len(gene_list)),
            "verbose": bool(verbose),
            "concurrent": bool(concurrent),
            "n_processes": n_processes,
            "pair_records": pair_records,
            **summary,
        },
        "unsupervised_method": {
            "fraction_failed_pairs": 1.0 - summary["pair_success_rate"],
        },
    }


@register_method("genes2genes_pairwise")
def run_genes2genes_pairwise(adata, method_params: dict) -> dict:
    """Run native pairwise Genes2Genes across all patient pairs."""
    common = _genes2genes_common_params(method_params)
    params = common["params"]
    patient_key = common["patient_key"]
    pseudotime_key = common["pseudotime_key"]
    n_bins = common["n_bins"]
    verbose = common["verbose"]
    concurrent = common["concurrent"]
    n_processes = common["n_processes"]

    if patient_key not in adata.obs:
        raise ValueError(f"genes2genes method expected {patient_key!r} in adata.obs")
    if pseudotime_key not in adata.obs:
        raise ValueError(f"genes2genes method expected {pseudotime_key!r} in adata.obs")

    gene_list = _resolve_g2g_gene_list(adata, params)
    g2g_main, g2g_cluster_utils = _import_genes2genes_modules()
    _ensure_genes2genes_anndata_compat()

    patient_ids = _get_patient_ids(adata, patient_key)
    patients = adata.obs[patient_key].to_numpy(dtype=str)

    pair_records = []
    for i, source_patient in enumerate(patient_ids):
        for target_patient in patient_ids[i + 1 :]:
            rec: dict[str, object] = {
                "pair": _make_pair_label(source_patient, target_patient),
                "status": "ok",
            }
            try:
                src_idx = np.flatnonzero(patients == source_patient)
                tgt_idx = np.flatnonzero(patients == target_patient)
                if src_idx.size == 0 or tgt_idx.size == 0:
                    raise ValueError("Source or target subset is empty.")

                adata_src = _prepare_g2g_adata_slice(adata[src_idx], pseudotime_key)
                adata_tgt = _prepare_g2g_adata_slice(adata[tgt_idx], pseudotime_key)

                with _maybe_silence_output(verbose):
                    aligner = g2g_main.RefQueryAligner(adata_tgt, adata_src, gene_list, n_bins)
                    aligner.align_all_pairs(concurrent=concurrent, n_processes=n_processes)
                    _, _ = g2g_cluster_utils.get_cluster_average_alignments(aligner, gene_list)

                rec.update(
                    {
                        "n_cells_source": int(src_idx.size),
                        "n_cells_target": int(tgt_idx.size),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                rec.update({"status": "error", "error": str(exc)})
            pair_records.append(rec)

    summary = _pairwise_summary(pair_records)
    gene_list_mode = str(params.get("gene_list_mode", "all_genes"))
    return {
        "evaluation_mode": "pairwise_only",
        "aligned_key": None,
        "method_params": params,
        "method_meta": {
            "backend": "python",
            "library": "genes2genes",
            "n_bins": int(n_bins),
            "gene_list_mode": gene_list_mode,
            "n_genes_aligned": int(len(gene_list)),
            "verbose": bool(verbose),
            "concurrent": bool(concurrent),
            "n_processes": n_processes,
            "pair_records": pair_records,
            **summary,
        },
        "unsupervised_method": {
            "fraction_failed_pairs": 1.0 - summary["pair_success_rate"],
        },
    }


@register_method("genes2genes_consensus")
def run_genes2genes_consensus(adata, method_params: dict) -> dict:
    """Run all-pairs Genes2Genes and derive one consensus pseudotime per cell."""
    common = _genes2genes_common_params(method_params)
    params = common["params"]
    patient_key = common["patient_key"]
    pseudotime_key = common["pseudotime_key"]
    n_bins = common["n_bins"]
    verbose = common["verbose"]
    concurrent = common["concurrent"]
    n_processes = common["n_processes"]
    key_added = params.get("key_added", "aligned_pseudotime_genes2genes_consensus")

    if patient_key not in adata.obs:
        raise ValueError(f"genes2genes method expected {patient_key!r} in adata.obs")
    if pseudotime_key not in adata.obs:
        raise ValueError(f"genes2genes method expected {pseudotime_key!r} in adata.obs")

    gene_list = _resolve_g2g_gene_list(adata, params)
    g2g_main, g2g_cluster_utils = _import_genes2genes_modules()
    _ensure_genes2genes_anndata_compat()

    patient_ids = _get_patient_ids(adata, patient_key)
    patients = adata.obs[patient_key].to_numpy(dtype=str)
    base_vals = adata.obs[pseudotime_key].to_numpy(dtype=float)

    accum = np.zeros(adata.n_obs, dtype=float)
    counts = np.zeros(adata.n_obs, dtype=float)
    pair_records = []

    for i, source_patient in enumerate(patient_ids):
        for target_patient in patient_ids[i + 1 :]:
            rec: dict[str, object] = {
                "pair": _make_pair_label(source_patient, target_patient),
                "status": "ok",
            }
            try:
                src_idx = np.flatnonzero(patients == source_patient)
                tgt_idx = np.flatnonzero(patients == target_patient)
                if src_idx.size == 0 or tgt_idx.size == 0:
                    raise ValueError("Source or target subset is empty.")

                adata_src = _prepare_g2g_adata_slice(adata[src_idx], pseudotime_key)
                adata_tgt = _prepare_g2g_adata_slice(adata[tgt_idx], pseudotime_key)

                with _maybe_silence_output(verbose):
                    # For this pair, map source -> target axis.
                    aligner = g2g_main.RefQueryAligner(adata_tgt, adata_src, gene_list, n_bins)
                    aligner.align_all_pairs(concurrent=concurrent, n_processes=n_processes)
                    _, tracked_path = g2g_cluster_utils.get_cluster_average_alignments(aligner, gene_list)

                mapped_src = _g2g_tracked_path_to_query_mapping(
                    tracked_path,
                    query_interp_points=np.asarray(aligner.TrajInt_Q.interpolation_points, dtype=float),
                    ref_interp_points=np.asarray(aligner.TrajInt_R.interpolation_points, dtype=float),
                    query_cell_pseudotime=adata_src.obs[pseudotime_key].to_numpy(dtype=float),
                )

                # Pair-context consensus contribution:
                # source contributes mapped values; target contributes its local axis.
                accum[src_idx] += mapped_src
                counts[src_idx] += 1.0
                accum[tgt_idx] += adata_tgt.obs[pseudotime_key].to_numpy(dtype=float)
                counts[tgt_idx] += 1.0

                rec.update(
                    {
                        "n_cells_source": int(src_idx.size),
                        "n_cells_target": int(tgt_idx.size),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                rec.update({"status": "error", "error": str(exc)})
            pair_records.append(rec)

    missing = counts == 0
    if np.any(missing):
        accum[missing] = base_vals[missing]
        counts[missing] = 1.0

    consensus = accum / counts
    adata.obs[key_added] = _safe_minmax_scale(consensus)

    summary = _pairwise_summary(pair_records)
    gene_list_mode = str(params.get("gene_list_mode", "all_genes"))
    return {
        "evaluation_mode": "single_axis",
        "aligned_key": key_added,
        "method_params": params,
        "method_meta": {
            "backend": "python",
            "library": "genes2genes",
            "consensus_strategy": "mean_over_pairwise_contexts",
            "n_bins": int(n_bins),
            "gene_list_mode": gene_list_mode,
            "n_genes_aligned": int(len(gene_list)),
            "verbose": bool(verbose),
            "concurrent": bool(concurrent),
            "n_processes": n_processes,
            "pair_records": pair_records,
            **summary,
        },
        "unsupervised_method": {
            "fraction_failed_pairs": 1.0 - summary["pair_success_rate"],
            "mean_pair_count_per_cell": float(np.mean(counts)),
            "min_pair_count_per_cell": float(np.min(counts)),
        },
    }


@register_method("cellalign")
@register_method("cellalign_fixed_reference")
def run_cellalign_fixed_reference(adata, method_params: dict) -> dict:
    """Run CellAlign against a fixed common reference and emit one axis per cell."""
    common = _cellalign_common_params(method_params)
    params = common["params"]
    patient_key = common["patient_key"]
    pseudotime_key = common["pseudotime_key"]
    rscript_bin = common["rscript_bin"]
    timeout_s = common["timeout_s"]
    dry_run = common["dry_run"]
    num_pts = common["num_pts"]
    win_sz = common["win_sz"]
    dist_method = common["dist_method"]
    bridge_script_abs = common["bridge_script_abs"]
    key_added = params.get("key_added", "aligned_pseudotime_cellalign_fixed_reference")
    reference_policy = params.get("reference_policy", "medoid")
    reference_patient = params.get("reference_patient")

    if not os.path.exists(bridge_script_abs):
        raise FileNotFoundError(
            f"cellalign bridge script not found: {bridge_script_abs}. Create it or pass method_params.bridge_script."
        )

    reference_patient = _select_reference_patient(
        adata,
        patient_key=patient_key,
        pseudotime_key=pseudotime_key,
        reference_policy=reference_policy,
        explicit=reference_patient,
    )
    patient_ids = _get_patient_ids(adata, patient_key)
    aligned_global = np.full(adata.n_obs, np.nan, dtype=float)
    patients = adata.obs[patient_key].to_numpy(dtype=str)
    aligned_global[patients == reference_patient] = adata.obs[pseudotime_key][patients == reference_patient].to_numpy(
        dtype=float
    )

    pair_records = []
    for pid in patient_ids:
        if pid == reference_patient:
            continue
        pair_adata, orig_idx = _subset_adata_by_patients(adata, patient_key, [pid, reference_patient])
        if dry_run:
            _ = _run_cellalign_bridge_call(
                pair_adata,
                patient_key=patient_key,
                pseudotime_key=pseudotime_key,
                bridge_script_abs=bridge_script_abs,
                rscript_bin=rscript_bin,
                timeout_s=timeout_s,
                num_pts=num_pts,
                win_sz=win_sz,
                dist_method=dist_method,
                mode="pairwise_fixed_reference",
                source_patient=pid,
                target_patient=reference_patient,
            )
            raise RuntimeError(
                "cellalign_fixed_reference dry_run=True: pairwise bridge schema written successfully. "
                "Set dry_run=False to execute full pipeline."
            )

        rec: dict[str, object] = {"pair": _make_pair_label(pid, reference_patient), "status": "ok"}
        try:
            bridge = _run_cellalign_bridge_call(
                pair_adata,
                patient_key=patient_key,
                pseudotime_key=pseudotime_key,
                bridge_script_abs=bridge_script_abs,
                rscript_bin=rscript_bin,
                timeout_s=timeout_s,
                num_pts=num_pts,
                win_sz=win_sz,
                dist_method=dist_method,
                mode="pairwise_fixed_reference",
                source_patient=pid,
                target_patient=reference_patient,
            )
            pair_patients = pair_adata.obs[patient_key].to_numpy(dtype=str)
            source_mask = pair_patients == pid
            aligned_global[orig_idx[source_mask]] = bridge["aligned"][source_mask]
            rec.update(
                {
                    "n_cells_source": int(source_mask.sum()),
                    "bridge_metrics": bridge["bridge_metrics"],
                }
            )
        except Exception as exc:  # noqa: BLE001
            rec.update({"status": "error", "error": str(exc)})
        pair_records.append(rec)

    failed_pairs = [r for r in pair_records if r.get("status") == "error"]
    if np.any(~np.isfinite(aligned_global)):
        if failed_pairs:
            details = "\n".join(f"  pair={r['pair']}: {r['error']}" for r in failed_pairs)
            raise ValueError(
                f"cellalign_fixed_reference failed to assign aligned values to all cells. "
                f"{len(failed_pairs)} pair(s) errored:\n{details}"
            )
        raise ValueError(
            "cellalign_fixed_reference failed to assign aligned values to all cells (no pair errors recorded)."
        )

    adata.obs[key_added] = _safe_minmax_scale(aligned_global)
    summary = _pairwise_summary(pair_records)

    return {
        "evaluation_mode": "single_axis",
        "aligned_key": key_added,
        "method_params": params,
        "method_meta": {
            "backend": "Rscript",
            "bridge_script": bridge_script_abs,
            "reference_policy": reference_policy,
            "reference_patient": reference_patient,
            "num_pts": num_pts,
            "win_sz": win_sz,
            "dist_method": dist_method,
            "pair_records": pair_records,
            **summary,
        },
        "unsupervised_method": {
            "fraction_failed_pairs": 1.0 - summary["pair_success_rate"],
        },
    }


@register_method("cellalign_pairwise")
def run_cellalign_pairwise(adata, method_params: dict) -> dict:
    """Run native pairwise CellAlign across all patient pairs without forcing one global axis."""
    common = _cellalign_common_params(method_params)
    params = common["params"]
    patient_key = common["patient_key"]
    pseudotime_key = common["pseudotime_key"]
    rscript_bin = common["rscript_bin"]
    timeout_s = common["timeout_s"]
    dry_run = common["dry_run"]
    num_pts = common["num_pts"]
    win_sz = common["win_sz"]
    dist_method = common["dist_method"]
    bridge_script_abs = common["bridge_script_abs"]

    if not os.path.exists(bridge_script_abs):
        raise FileNotFoundError(
            f"cellalign bridge script not found: {bridge_script_abs}. Create it or pass method_params.bridge_script."
        )

    patient_ids = _get_patient_ids(adata, patient_key)
    pair_records = []
    for i, source_patient in enumerate(patient_ids):
        for target_patient in patient_ids[i + 1 :]:
            pair_adata, _ = _subset_adata_by_patients(adata, patient_key, [source_patient, target_patient])
            if dry_run:
                _ = _run_cellalign_bridge_call(
                    pair_adata,
                    patient_key=patient_key,
                    pseudotime_key=pseudotime_key,
                    bridge_script_abs=bridge_script_abs,
                    rscript_bin=rscript_bin,
                    timeout_s=timeout_s,
                    num_pts=num_pts,
                    win_sz=win_sz,
                    dist_method=dist_method,
                    mode="pairwise_native",
                    source_patient=source_patient,
                    target_patient=target_patient,
                )
                raise RuntimeError(
                    "cellalign_pairwise dry_run=True: pairwise bridge schema written successfully. "
                    "Set dry_run=False to execute full pairwise pipeline."
                )

            rec: dict[str, object] = {"pair": _make_pair_label(source_patient, target_patient), "status": "ok"}
            try:
                bridge = _run_cellalign_bridge_call(
                    pair_adata,
                    patient_key=patient_key,
                    pseudotime_key=pseudotime_key,
                    bridge_script_abs=bridge_script_abs,
                    rscript_bin=rscript_bin,
                    timeout_s=timeout_s,
                    num_pts=num_pts,
                    win_sz=win_sz,
                    dist_method=dist_method,
                    mode="pairwise_native",
                    source_patient=source_patient,
                    target_patient=target_patient,
                )
                rec.update(
                    {
                        "n_cells": int(pair_adata.n_obs),
                        "bridge_metrics": bridge["bridge_metrics"],
                    }
                )
            except Exception as exc:  # noqa: BLE001
                rec.update({"status": "error", "error": str(exc)})
            pair_records.append(rec)

    summary = _pairwise_summary(pair_records)
    return {
        "evaluation_mode": "pairwise_only",
        "aligned_key": None,
        "method_params": params,
        "method_meta": {
            "backend": "Rscript",
            "bridge_script": bridge_script_abs,
            "num_pts": num_pts,
            "win_sz": win_sz,
            "dist_method": dist_method,
            "pair_records": pair_records,
            **summary,
        },
        "unsupervised_method": {
            "fraction_failed_pairs": 1.0 - summary["pair_success_rate"],
        },
    }


@register_method("cellalign_consensus")
def run_cellalign_consensus(adata, method_params: dict) -> dict:
    """Run all-pairs CellAlign and derive one consensus pseudotime per cell.

    Current skeleton consensus averages each cell's aligned values across all
    pairwise runs in which it appears. This preserves the adapter contract and
    can later be replaced by a stricter monotone-map synchronization solver.
    """
    common = _cellalign_common_params(method_params)
    params = common["params"]
    patient_key = common["patient_key"]
    pseudotime_key = common["pseudotime_key"]
    rscript_bin = common["rscript_bin"]
    timeout_s = common["timeout_s"]
    dry_run = common["dry_run"]
    num_pts = common["num_pts"]
    win_sz = common["win_sz"]
    dist_method = common["dist_method"]
    bridge_script_abs = common["bridge_script_abs"]
    key_added = params.get("key_added", "aligned_pseudotime_cellalign_consensus")

    if not os.path.exists(bridge_script_abs):
        raise FileNotFoundError(
            f"cellalign bridge script not found: {bridge_script_abs}. Create it or pass method_params.bridge_script."
        )

    patient_ids = _get_patient_ids(adata, patient_key)
    accum = np.zeros(adata.n_obs, dtype=float)
    counts = np.zeros(adata.n_obs, dtype=float)
    pair_records = []

    for i, source_patient in enumerate(patient_ids):
        for target_patient in patient_ids[i + 1 :]:
            pair_adata, orig_idx = _subset_adata_by_patients(adata, patient_key, [source_patient, target_patient])
            if dry_run:
                _ = _run_cellalign_bridge_call(
                    pair_adata,
                    patient_key=patient_key,
                    pseudotime_key=pseudotime_key,
                    bridge_script_abs=bridge_script_abs,
                    rscript_bin=rscript_bin,
                    timeout_s=timeout_s,
                    num_pts=num_pts,
                    win_sz=win_sz,
                    dist_method=dist_method,
                    mode="pairwise_consensus",
                    source_patient=source_patient,
                    target_patient=target_patient,
                )
                raise RuntimeError(
                    "cellalign_consensus dry_run=True: pairwise bridge schema written successfully. "
                    "Set dry_run=False to execute full consensus pipeline."
                )

            rec: dict[str, object] = {"pair": _make_pair_label(source_patient, target_patient), "status": "ok"}
            try:
                bridge = _run_cellalign_bridge_call(
                    pair_adata,
                    patient_key=patient_key,
                    pseudotime_key=pseudotime_key,
                    bridge_script_abs=bridge_script_abs,
                    rscript_bin=rscript_bin,
                    timeout_s=timeout_s,
                    num_pts=num_pts,
                    win_sz=win_sz,
                    dist_method=dist_method,
                    mode="pairwise_consensus",
                    source_patient=source_patient,
                    target_patient=target_patient,
                )
                accum[orig_idx] += bridge["aligned"]
                counts[orig_idx] += 1.0
                rec.update(
                    {
                        "n_cells": int(pair_adata.n_obs),
                        "bridge_metrics": bridge["bridge_metrics"],
                    }
                )
            except Exception as exc:  # noqa: BLE001
                rec.update({"status": "error", "error": str(exc)})
            pair_records.append(rec)

    missing = counts == 0
    if np.any(missing):
        accum[missing] = adata.obs[pseudotime_key].to_numpy(dtype=float)[missing]
        counts[missing] = 1.0

    consensus = accum / counts
    adata.obs[key_added] = _safe_minmax_scale(consensus)

    summary = _pairwise_summary(pair_records)
    return {
        "evaluation_mode": "single_axis",
        "aligned_key": key_added,
        "method_params": params,
        "method_meta": {
            "backend": "Rscript",
            "bridge_script": bridge_script_abs,
            "consensus_strategy": "mean_over_pairwise_contexts",
            "num_pts": num_pts,
            "win_sz": win_sz,
            "dist_method": dist_method,
            "pair_records": pair_records,
            **summary,
        },
        "unsupervised_method": {
            "fraction_failed_pairs": 1.0 - summary["pair_success_rate"],
            "mean_pair_count_per_cell": float(np.mean(counts)),
            "min_pair_count_per_cell": float(np.min(counts)),
        },
    }
