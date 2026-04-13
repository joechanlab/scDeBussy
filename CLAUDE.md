# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

**scDeBussy** is a Python package for pseudotime alignment of single-cell RNA-seq data across multiple patients/batches. It uses EM-based dynamic time warping (DTW) barycenter averaging to map per-patient local pseudotimes onto a shared consensus timeline, producing `adata.obs["aligned_pseudotime"]`.

## Commands

### Install (development)
```bash
pip install -e ".[dev,test]"
pre-commit install
```

### Run tests
```bash
uvx hatch test --cover --python 3.10   # full suite with coverage
pytest tests/test_align.py             # single test file
pytest tests/test_align.py::test_name  # single test
```

### Lint / format
```bash
pre-commit run --all-files
ruff check --fix src/
ruff format src/
```

### Build docs
```bash
hatch run docs:build
```

### Build package
```bash
uv build
uvx twine check --strict dist/*.whl
```

## Architecture

### Module layout
```
src/scdebussy/
├── __init__.py          # Public API surface and tuning helpers
├── tuning.py            # Optuna-based hyperparameter optimization
├── tl/                  # Tools (algorithms)
│   ├── _alignment.py    # Core: smooth_patient_trajectory(), scDeBussy class
│   ├── _data_prep.py    # load_from_manifest(), filter_* utilities
│   ├── _metrics.py      # cross_batch_knn_purity(), purity sweep
│   ├── _synthetic_dataset.py  # simulate_LF_MOGP() for benchmarking
│   └── _trend_analysis.py     # Gene trend features, GSEA-style enrichment
└── pl/                  # Plots (visualizations)
    ├── _alignment.py    # plot_barycenter_boundaries(), plot_em_convergence()
    └── _metrics.py      # cross_batch_purity_sweep(), performance plots
```

### Core data flow

1. **Input**: Per-patient `AnnData` with local pseudotime in `.obs`
2. **`smooth_patient_trajectory()`** — Gaussian kernel regression onto a uniform `n_bins`-point grid, producing a `[n_bins × n_genes]` trajectory per patient
3. **`scDeBussy.fit()`** — EM loop:
   - *E-step*: asymmetric soft-DTW aligns each patient trajectory to the current barycenter (Sakoe-Chiba band, open ends)
   - *M-step*: soft-DTW barycenter averaging (tslearn) updates the consensus `B(τ)`
4. **`scDeBussy.transform()`** — maps raw cells to aligned pseudotime via DTW path interpolation + isotonic regression (monotonicity)
5. **Outputs** stored in `adata.obs["aligned_pseudotime"]` and `adata.uns["barycenter"]` (expression, warp paths, convergence diagnostics)

### Conventions (follow the scanpy/scverse pattern)
- Results go into `adata.obs`, `adata.uns`, or `adata.obsm` — not as return values
- New public functions must be added to the relevant `__init__.py` and its `__all__` list
- Type annotations required for all public functions
- NumPy-style docstrings (Parameters, Returns, Examples sections)
- Line length: 120 characters; docstring style: NumPy; linter: ruff

### Key configuration
- `pyproject.toml`: all build, dependency, ruff, pytest, coverage, and hatch settings
- `pytest` importmode is `importlib` (allows duplicate filenames across test dirs)
- CI runs Python 3.10 and 3.12; sets `MPLBACKEND=agg` and `SCIPY_ARRAY_API=1`
