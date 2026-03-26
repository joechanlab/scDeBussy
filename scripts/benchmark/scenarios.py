"""Scenario registry for the scDeBussy benchmark suite.

Each scenario is a dict with:
  name          : unique identifier used as subdirectory name
  description   : human-readable description
  n_seeds       : number of random seeds to run (0 … n_seeds-1)
  sim_kwargs    : keyword arguments forwarded to scdebussy.tl.simulate_LF_MOGP
  scdebussy_params : keyword arguments forwarded to scdebussy.tl.scDeBussy
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Common defaults shared across all scenarios
# ---------------------------------------------------------------------------
_STANDARD_MIX = ["early", "full", "late", "bimodal", "full"]

_COMMON_SIM = {
    "K": 100,
    "M": 4,
    "time_grid": None,  # filled in at runtime (np.linspace)
    "factor_kernels": [
        {"lengthscale": 0.40, "variance": 1.0},
        {"lengthscale": 0.40, "variance": 1.0},
        {"lengthscale": 0.10, "variance": 0.5},
        {"lengthscale": 1.00, "variance": 0.1},
    ],
    "deviation_kernel": {"lengthscale": 0.25, "variance": 0.15},
    "eps": 0.0,
    "warp_mode": "none",
    "warp_strength": 0.0,
    "warp_types": None,
    "noise_settings": None,
}

_DEFAULT_SCDEBUSSY = {
    "patient_key": "patient",
    "pseudotime_key": "s_local",
    "key_added": "aligned_pseudotime",
    "barycenter_key": "barycenter",
    "n_bins": 100,
    "bandwidth": 0.1,
    "gamma": 0.1,
    "max_iter": 20,
    "tol": 1e-4,
    "dtw_dist_method": "cosine",
    "dtw_step_pattern": "asymmetric",
    "dtw_open_begin": True,
    "dtw_open_end": True,
    "dtw_window_fraction": 0.2,
    "barycenter_init_iter": 5,
    "barycenter_update_iter": 1,
    "verbose": False,
}

# ---------------------------------------------------------------------------
# Helper to build a scenario dict cleanly
# ---------------------------------------------------------------------------


def _scenario(
    name,
    description,
    *,
    P,
    patient_groups,
    lambda_cells,
    sigma_noise,
    warp_mode="none",
    warp_strength=0.0,
    warp_types=None,
    noise_settings=None,
    n_seeds=10,
    scdebussy_overrides=None,
):
    sim = {
        **_COMMON_SIM,
        "P": P,
        "patient_groups": patient_groups,
        "lambda_cells": lambda_cells,
        "sigma_noise": sigma_noise,
        "warp_mode": warp_mode,
        "warp_strength": warp_strength,
        "warp_types": warp_types,
        "noise_settings": noise_settings,
    }
    scd = {**_DEFAULT_SCDEBUSSY, **(scdebussy_overrides or {})}
    return {
        "name": name,
        "description": description,
        "n_seeds": n_seeds,
        "sim_kwargs": sim,
        "scdebussy_params": scd,
    }


# ---------------------------------------------------------------------------
# 12 Scenario definitions
# ---------------------------------------------------------------------------

SCENARIOS: list[dict] = [
    # 1. Ideal case — no warp, low noise, all patients cover full trajectory
    _scenario(
        "clean_baseline",
        "No warping, low Gaussian noise, all patients cover full pseudotime range.",
        P=5,
        patient_groups=["full"] * 5,
        lambda_cells=300,
        sigma_noise=0.05,
    ),
    # 2. Easy monotone warping
    _scenario(
        "warp_easy",
        "Mild random-monotone warping (strength 0.2), moderate Gaussian noise.",
        P=5,
        patient_groups=_STANDARD_MIX,
        lambda_cells=300,
        sigma_noise=0.10,
        warp_mode="random_monotone",
        warp_strength=0.2,
    ),
    # 3. Medium monotone warping
    _scenario(
        "warp_medium",
        "Medium random-monotone warping (strength 0.5), moderate noise.",
        P=5,
        patient_groups=_STANDARD_MIX,
        lambda_cells=300,
        sigma_noise=0.20,
        warp_mode="random_monotone",
        warp_strength=0.5,
    ),
    # 4. Hard monotone warping
    _scenario(
        "warp_hard_monotone",
        "Strong random-monotone warping (strength 0.8), higher noise.",
        P=5,
        patient_groups=_STANDARD_MIX,
        lambda_cells=300,
        sigma_noise=0.30,
        warp_mode="random_monotone",
        warp_strength=0.8,
    ),
    # 5. Mixed sigmoid warps (matching external benchmark setup)
    _scenario(
        "warp_sigmoid",
        "Mixed per-patient sigmoid and nonlinear warps at moderate strength.",
        P=5,
        patient_groups=_STANDARD_MIX,
        lambda_cells=300,
        sigma_noise=0.15,
        warp_mode="mixed",
        warp_strength=0.5,
        warp_types=["sigmoid", "sigmoid", "sigmoid", "sigmoid", "nonlinear"],
    ),
    # 6. Mixed nonlinear warps (complementary directions)
    _scenario(
        "warp_nonlinear",
        "Mixed nonlinear, nonlinear_inverse, and sigmoid warps at medium-high strength.",
        P=5,
        patient_groups=_STANDARD_MIX,
        lambda_cells=300,
        sigma_noise=0.20,
        warp_mode="mixed",
        warp_strength=0.6,
        warp_types=["nonlinear", "nonlinear_inverse", "nonlinear_inverse", "sigmoid", "sigmoid"],
    ),
    # 7. Heavy heteroscedastic noise with dropout and outliers, no warp
    _scenario(
        "noise_heavy",
        "No warping. Heteroscedastic Gaussian noise σ=0.40, 15% dropout, 8% outliers.",
        P=5,
        patient_groups=["full"] * 5,
        lambda_cells=300,
        sigma_noise=0.40,
        noise_settings={
            "model": "gaussian",
            "heteroscedastic": True,
            "hetero_base": 0.75,
            "hetero_amp": 1.8,
            "outlier_frac": 0.08,
            "dropout_frac": 0.15,
            "patient_offset_scale": 0.10,
        },
    ),
    # 8. Heavy-tailed (Student-t) noise, no warp
    _scenario(
        "noise_student_t",
        "No warping. Heavy-tailed Student-t noise (df=3, σ=0.30).",
        P=5,
        patient_groups=["full"] * 5,
        lambda_cells=300,
        sigma_noise=0.30,
        noise_settings={
            "model": "student_t",
            "df": 3,
        },
    ),
    # 9. Partial coverage — some patients only see early or late pseudotime
    _scenario(
        "partial_coverage",
        "Random-monotone warping on partially-covered patient groups (early/late bias).",
        P=5,
        patient_groups=["early", "late", "late", "early", "full"],
        lambda_cells=300,
        sigma_noise=0.20,
        warp_mode="random_monotone",
        warp_strength=0.4,
    ),
    # 10. Severe cell-count imbalance between patients
    _scenario(
        "cell_imbalance",
        "Random-monotone warping with extreme cell-count imbalance per patient.",
        P=5,
        patient_groups=_STANDARD_MIX,
        lambda_cells=[50, 500, 1000, 50, 300],  # list triggers per-patient Poisson rates
        sigma_noise=0.20,
        warp_mode="random_monotone",
        warp_strength=0.4,
    ),
    # 11. Many patients (P=10)
    _scenario(
        "high_p",
        "10 patients with cycling group pattern, random-monotone warping.",
        P=10,
        patient_groups=[_STANDARD_MIX[i % len(_STANDARD_MIX)] for i in range(10)],
        lambda_cells=200,
        sigma_noise=0.20,
        warp_mode="random_monotone",
        warp_strength=0.4,
    ),
    # 12. Combined challenge — strong mixed warps + heavy noise
    _scenario(
        "combined_challenge",
        "Strong mixed sigmoid+nonlinear warps, heteroscedastic noise with dropout, "
        "partial coverage, and cell imbalance.",
        P=5,
        patient_groups=["early", "late", "bimodal", "full", "early"],
        lambda_cells=[100, 500, 300, 200, 400],
        sigma_noise=0.40,
        warp_mode="mixed",
        warp_strength=0.75,
        warp_types=["sigmoid", "sigmoid", "nonlinear", "nonlinear_inverse", "nonlinear"],
        noise_settings={
            "model": "gaussian",
            "heteroscedastic": True,
            "hetero_base": 0.75,
            "hetero_amp": 1.8,
            "outlier_frac": 0.07,
            "dropout_frac": 0.12,
            "patient_offset_scale": 0.12,
        },
    ),
]

# Quick lookup by name
SCENARIO_REGISTRY: dict[str, dict] = {s["name"]: s for s in SCENARIOS}


# Full manifest: list of (scenario_name, seed) tuples in job-array order
def build_manifest() -> list[tuple[str, int]]:
    """Return all (scenario_name, seed) pairs in deterministic order."""
    manifest = []
    for sc in SCENARIOS:
        for seed in range(sc["n_seeds"]):
            manifest.append((sc["name"], seed))
    return manifest
