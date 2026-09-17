"""Independent generative fixtures for recovery and deliberate misspecification."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    """Explicit simulator interventions; none are fitted-model parameters."""

    seed: int = 0
    n_patients: int = 8
    cells_per_patient: int = 200
    n_genes: int = 12
    noise: float = 0.08
    warp: float = 0.0
    occupancy: float = 0.0
    expression: float = 0.0
    gap_depth: float = 0.0
    gap_start: float = 0.4
    gap_end: float = 0.6
    ramp_width: float = 0.03
    noise_model: str = "gaussian"
    reversal: bool = False
    multimodal: bool = False
    gene_panel: str = "all"
    design: str = "exchangeable"


def _expression(z, genes):
    frequency = 1 + np.arange(genes) % 3
    phase = np.arange(genes) * np.pi / max(genes, 1)
    return np.sin(np.asarray(z)[:, None] * frequency * np.pi + phase) + np.asarray(z)[:, None]


def simulate(config=None, **kwargs):
    """Generate AnnData with latent truth, patient conditions, blocks, and pairs.

    Expression and observed pseudotime are parallel functions of latent state.
    Density and censoring act on sampling of that state. Gene panels are applied
    after generation, preserving the truth of the retained genes.
    """
    import anndata as ad
    import pandas as pd

    if config is not None and kwargs:
        raise ValueError("pass a SimulationConfig or keyword settings, not both")
    cfg = SimulationConfig(**kwargs) if config is None else config
    if cfg.n_patients < 4 or cfg.n_patients % 2 or cfg.cells_per_patient < 2 or cfg.n_genes < 2:
        raise ValueError("use an even number of >=4 patients, >=2 cells, and >=2 genes")
    if cfg.noise < 0 or not 0 <= cfg.gap_depth <= 1 or not 0 <= cfg.gap_start < cfg.gap_end <= 1:
        raise ValueError("invalid noise or gap configuration")
    if cfg.ramp_width <= 0 or cfg.design not in {"exchangeable", "blocked", "paired"}:
        raise ValueError("positive ramp_width and a valid design are required")
    if cfg.noise_model not in {"gaussian", "heavy_tailed", "correlated", "heteroskedastic"}:
        raise ValueError("unknown noise_model")
    if cfg.gene_panel not in {
        "all",
        "random",
        "high_variance",
        "low_variance",
        "duplicated",
        "nuisance",
        "informative",
    }:
        raise ValueError("unknown gene_panel")
    rng = np.random.default_rng(cfg.seed)
    arrays, metadata = [], []
    for p in range(cfg.n_patients):
        condition = p % 2
        accepted, total = [], 0
        while total < cfg.cells_per_patient:
            shape = max(0.1, 1 + condition * cfg.occupancy)
            z = rng.beta(shape, 1.0, cfg.cells_per_patient * 2)
            ingress = np.clip((z - cfg.gap_start) / cfg.ramp_width, 0.0, 1.0)
            egress = np.clip((cfg.gap_end - z) / cfg.ramp_width, 0.0, 1.0)
            keep = rng.random(len(z)) < 1 - condition * cfg.gap_depth * ingress * egress
            accepted.append(z[keep])
            total += int(keep.sum())
        z = np.sort(np.concatenate(accepted)[: cfg.cells_per_patient])
        tau = z ** np.exp(condition * cfg.warp)
        if cfg.reversal and condition:
            tau = np.clip(z + 0.15 * np.sin(4 * np.pi * z), 0.0, 1.0)
        expression = _expression(z, cfg.n_genes)
        affected = np.arange(cfg.n_genes) < max(1, cfg.n_genes // 3)
        expression[:, affected] += condition * cfg.expression * np.exp(-(((z[:, None] - 0.5) / 0.2) ** 2))
        # Patient-level random expression offsets preserve patient as the unit.
        offset_rng = np.random.default_rng(cfg.seed + 100_003 + (p // 2 if cfg.design != "exchangeable" else p))
        expression += offset_rng.normal(0, cfg.noise / 2, cfg.n_genes)
        noise = rng.normal(size=expression.shape)
        if cfg.noise_model == "heavy_tailed":
            noise = rng.standard_t(3, size=expression.shape) / np.sqrt(3)
        elif cfg.noise_model == "correlated":
            noise = np.sqrt(0.3) * noise + np.sqrt(0.7) * rng.normal(size=(len(z), 1))
        elif cfg.noise_model == "heteroskedastic":
            noise *= np.linspace(0.2, 3.0, cfg.n_genes)[None, :] * (0.5 + z[:, None])
        expression += cfg.noise * noise
        if cfg.multimodal:
            expression += rng.choice([-1.0, 1.0], size=(len(z), 1)) * 0.4 * np.exp(-(((z[:, None] - 0.5) / 0.1) ** 2))
        arrays.append(expression)
        metadata.append(
            pd.DataFrame(
                {
                    "patient": f"p{p}",
                    "condition": condition,
                    "s_local": tau,
                    "z_true": z,
                    "pair": p // 2,
                    "block": p // 2,
                },
                index=[f"p{p}_c{i}" for i in range(len(z))],
            )
        )
    x = np.concatenate(arrays)
    genes = np.arange(cfg.n_genes)
    if cfg.gene_panel == "random":
        genes = np.sort(rng.choice(genes, max(2, cfg.n_genes // 2), replace=False))
    elif cfg.gene_panel in {"high_variance", "low_variance"}:
        ordering = np.argsort(x.var(0))
        genes = (
            ordering[-max(2, cfg.n_genes // 2) :]
            if cfg.gene_panel == "high_variance"
            else ordering[: max(2, cfg.n_genes // 2)]
        )
    elif cfg.gene_panel == "informative":
        genes = genes[: max(2, cfg.n_genes // 3)]
    x = x[:, genes]
    var = pd.DataFrame(
        {"source_gene": genes, "expression_affected": genes < max(1, cfg.n_genes // 3)}, index=[f"g{i}" for i in genes]
    )
    if cfg.gene_panel == "duplicated":
        x = np.concatenate([x, x], axis=1)
        duplicate = var.copy()
        duplicate.index = [f"{g}_copy" for g in duplicate.index]
        var = pd.concat([var, duplicate])
    elif cfg.gene_panel == "nuisance":
        nuisance = pd.DataFrame(
            {"source_gene": -1, "expression_affected": False}, index=[f"nuisance{i}" for i in range(cfg.n_genes)]
        )
        x = np.concatenate([x, rng.normal(size=(len(x), cfg.n_genes))], axis=1)
        var = pd.concat([var, nuisance])
    result = ad.AnnData(x, obs=pd.concat(metadata), var=var)
    result.uns["simulation_v2"] = asdict(cfg)
    result.uns["simulation_v2"]["causal_structure"] = "z -> expression; z -> observed pseudotime; density selects z"
    return result


def nonidentifiable_pair(*, seed=0, shift=0.1):
    """Identical observations admit either a shifted expression or time origin.

    In interpretation A, x=f(z+shift), tau=z. In B, x=f(z'),
    tau=z'-shift, with z'=z+shift. The observation law is identical.
    """
    if not 0 < shift < 0.25:
        raise ValueError("shift must lie in (0, .25)")
    a = simulate(seed=seed, noise=0.0)
    z = 0.25 + 0.5 * a.obs["z_true"].to_numpy()
    a.obs["z_true"] = a.obs["s_local"] = z
    a.X = _expression(z + shift, a.n_vars)
    b = a.copy()
    b.obs["z_true"] = z + shift
    a.uns["interpretation"] = "expression shift"
    b.uns["interpretation"] = "observed-time shift and translated latent sampling support"
    return a, b
