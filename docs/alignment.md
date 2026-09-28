# Canonical alignment API

`scdebussy.alignment` provides reusable affine-gap pseudotime alignment.

The public API is `fit`, `project`, `summarize`, and `compare`. It uses local pseudotime coordinates, explicit matchability masks, affine gaps, support-aware projection, and patient-level permutation inference.

The module does not include joint posterior warps, adaptive terminal pricing, coverage gates, HTAN feature selection, or cohort-specific validation.
