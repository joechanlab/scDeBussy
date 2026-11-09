# Archived simulation backend replay

Status: explicit historical compatibility source, not the public alignment default.

The simulation slide results use posterior inference and reference updates from historical commit `0f920137b9b7e30c5127a6afe7c4ee31fd950821`, plus the strict monitor recorded in the paper repository. Calling a newer normal alignment API does not reproduce those runs.

The capsule in `src/scdebussy/_reproduction/sim_v2_20261026` contains the required generic historical `align` source and generic simulator/selector helpers. Its backend consolidates graph/reference/projection function bodies extracted from verified historical validation scripts, without carrying patient datasets, cohort drivers or result files. The paper repository owns scenarios, benchmark orchestration and figures. The source manifest records original commits and source hashes; `portable_hashes.json` protects the archived Python files at replay time.

Load this source through the paper repository's isolated `reproduce_selected_strict.py` entry point with an explicit package checkout. It does not replace or monkey-patch a user's normally imported current scDeBussy package. In the isolated replay process, the explicitly archived namespace intentionally takes precedence over editable installs.

Validation on saved schematic seed0 inputs reproduced both selected-strict and TT coordinates exactly, at 402 and317 updates respectively. Extracted graph arrays, posterior curves/matrices, smooth reference updates and projection also matched the original implementations exactly. This does not establish global optimality, fresh input-generation identity or historical comparator-environment identity. The archived interface requires a source checkout; wheel-only replay is not certified.

The replay driver checks archive hashes instead of requiring the surrounding checkout HEAD to equal the historical commit. This is an intentional provenance/transport change: a modern checkout contains the explicit immutable historical capsule. Current public defaults remain separate.
