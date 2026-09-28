"""Public result containers for the experimental linear alignment API."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .posterior import Posterior


class InsufficientSupportError(ValueError):
    """No matchable landmark is available to initialize a consensus."""


class DegenerateCalibrationError(ValueError):
    """Automatic gap calibration requires explicit positive extension costs."""


@dataclass(frozen=True)
class Landmarks:
    """Patient expression estimates; variance is an approximate mean variance."""

    patient: str
    coordinates: np.ndarray
    expression: np.ndarray
    variance: np.ndarray
    precision: np.ndarray
    slope: np.ndarray
    n_eff: np.ndarray
    local_count: np.ndarray
    matchable: np.ndarray
    support_intervals: np.ndarray | None = None


@dataclass(frozen=True)
class ProjectionResult:
    """Cell positions always accompanied by support and uncertainty."""

    cell_ids: np.ndarray
    patients: np.ndarray
    pseudotime: np.ndarray
    probability: np.ndarray
    conditional_probability: np.ndarray
    log_retained: np.ndarray
    unprojected: np.ndarray
    position: np.ndarray
    entropy: np.ndarray
    status: np.ndarray
    reversals: dict


@dataclass(frozen=True)
class ComparisonResult:
    """Observed statistics and patient-permutation inference by family."""

    kinetic: pd.DataFrame
    occupancy: pd.DataFrame
    expression: pd.DataFrame
    groups: tuple
    config: dict


@dataclass
class AlignmentResult:
    """A fitted model and complete provenance; fitting never mutates AnnData."""

    patients: tuple[str, ...]
    genes: tuple[str, ...]
    barycenter: np.ndarray
    landmarks: tuple[Landmarks, ...]
    posteriors: tuple[Posterior, ...]
    s0: np.ndarray
    s1: np.ndarray
    log_s0: np.ndarray
    identified: np.ndarray
    weakly_identified: np.ndarray
    config: dict
    history: list[dict]
    converged: bool
    cell_ids: np.ndarray
    cell_patients: np.ndarray
    cell_pseudotime: np.ndarray
    provenance: dict
    diagnostics: dict = field(default_factory=dict)
    consensus_coordinates: np.ndarray | None = None

    def __post_init__(self):
        if self.consensus_coordinates is None:
            # Explicit compatibility for results constructed under the v2.0 contract.
            self.consensus_coordinates = np.arange(len(self.barycenter)) / len(self.barycenter)
            self.provenance.setdefault("coordinate_convention", "half-open-rank-v2.0")
        self.consensus_coordinates = np.asarray(self.consensus_coordinates, dtype=float).copy()
        axis = self.consensus_coordinates
        if (
            axis.shape != (len(self.barycenter),)
            or not np.isfinite(axis).all()
            or np.any(np.diff(axis) <= 0)
            or np.any((axis < 0) | (axis > 1))
        ):
            raise ValueError("consensus_coordinates must be strictly increasing in [0,1] and match barycenter rows")

    def write(self, adata, *, namespace="scdebussy_alignment"):
        """Explicitly write v2 outputs without replacing any v1 fields."""
        from .projection import project

        if not namespace.startswith("scdebussy_alignment"):
            raise ValueError("namespace must start with 'scdebussy_alignment'")
        projection = project(self, adata)
        for name in ("position", "unprojected", "entropy", "status", "log_retained"):
            adata.obs[f"{namespace}_{name}"] = getattr(projection, name)
        adata.obsm[f"{namespace}_probability"] = projection.probability
        adata.uns[namespace] = {
            "experimental": True,
            "patients": np.asarray(self.patients),
            "genes": np.asarray(self.genes),
            "barycenter": self.barycenter.copy(),
            "consensus_coordinates": self.consensus_coordinates.copy(),
            "s0": self.s0.copy(),
            "s1": self.s1.copy(),
            "log_s0": self.log_s0.copy(),
            "identified": self.identified.copy(),
            "weakly_identified": self.weakly_identified.copy(),
            "config": dict(self.config),
            "provenance": dict(self.provenance),
            "converged": self.converged,
            "history": {key: np.asarray([row[key] for row in self.history]) for key in self.history[0]},
            "landmarks": {
                str(k): {
                    "patient": lm.patient,
                    "support_intervals": lm.support_intervals if lm.support_intervals is not None else np.empty((0, 2)),
                    **{
                        name: getattr(lm, name).copy()
                        for name in (
                            "coordinates",
                            "expression",
                            "variance",
                            "precision",
                            "slope",
                            "n_eff",
                            "local_count",
                            "matchable",
                        )
                    },
                }
                for k, lm in enumerate(self.landmarks)
            },
            "diagnostics": {
                str(k): {
                    "patient": patient,
                    **self.diagnostics[patient],
                    "projection_reversals": projection.reversals[patient],
                }
                for k, patient in enumerate(self.patients)
            },
            "posterior": {
                str(k): {
                    "patient": self.patients[k],
                    "log_a": p.log_a.copy(),
                    "log_z": p.log_z,
                    "insertion": p.insertion.copy(),
                    "deletion": p.deletion.copy(),
                    "match_count_probability": p.count_probability.copy(),
                    "entropy": p.entropy,
                    "map_edges": p.map_edges.copy(),
                    "map_energy": p.map_energy,
                }
                for k, p in enumerate(self.posteriors)
            },
        }
        return adata
