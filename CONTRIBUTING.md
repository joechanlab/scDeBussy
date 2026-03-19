# Contributing to scDeBussy

Thank you for your interest in contributing! Please follow the guidelines below to keep the codebase consistent and maintainable.

---

## Getting Started

```bash
git clone https://github.com/joechanlab/scDeBussy.git
cd scDeBussy
pip install -e ".[dev,test]"
pre-commit install
```

Python ≥ 3.10 is required.

---

## Project Structure

- `src/scdebussy/tl/` — algorithms (tools layer)
- `src/scdebussy/pl/` — visualizations (plot layer)
- `tests/` — pytest suite
- `docs/` — Sphinx + MyST documentation

---

## Development Workflow

1. **Branch** off `main` with a descriptive name (`feat/my-feature`, `fix/issue-42`).
2. **Write tests** in `tests/` for any new functionality. Tests use `pytest`; run them with:
   ```bash
   pytest tests/
   ```
3. **Lint & format** with Ruff (configured in `pyproject.toml`, line length 120):
   ```bash
   pre-commit run --all-files
   ```
   Pre-commit will also run Ruff automatically on each commit once installed.
4. **Open a pull request** against `main`. Include a clear description of what changed and why.

---

## Code Conventions

- Follow the **scanpy/scverse ecosystem pattern**: algorithms live in `tl`, plots in `pl`, and both operate on `AnnData` objects.
- Store algorithm outputs in `adata.obs`, `adata.uns`, or `adata.obsm` — avoid returning large arrays outside the `AnnData`.
- Use type annotations for all public function signatures.
- Keep functions focused; prefer small, testable units over monolithic methods.

---

## Documentation

- Public functions must have NumPy-style docstrings with `Parameters`, `Returns`, and a short `Examples` section.
- If you add a new public function, add it to `docs/api.md` and the appropriate `__init__.py` `__all__` list.
- Build the docs locally with:
  ```bash
  hatch run docs:build
  ```

---

## Reporting Issues

Please open a GitHub issue with:
- A minimal reproducible example.
- The output of `session_info()` (from the `session-info2` package).
- The full traceback if applicable.
