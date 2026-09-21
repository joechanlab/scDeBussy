"""Experimental affine-gap Gibbs alignment; v1 remains the recommended API.

No Torch dependency is imported by this namespace. Inputs must already be
normalized; gap probabilities conflate unmatched material and resampling.
"""

from ._api import fit
from ._compare import compare
from ._dp import JointPosterior, Posterior, check_joint_invariants, infer, infer_joint
from ._model import LatticeModel, NoValidPathError, enumerate_joint_paths, enumerate_paths
from ._project import PatientQuantities, project, summarize
from ._simulate import SimulationConfig, nonidentifiable_pair, simulate
from ._types import (
    AlignmentResult,
    ComparisonResult,
    DegenerateCalibrationError,
    InsufficientSupportError,
    Landmarks,
    ProjectionResult,
)

__all__ = [
    "fit",
    "project",
    "compare",
    "summarize",
    "infer",
    "infer_joint",
    "check_joint_invariants",
    "enumerate_paths",
    "enumerate_joint_paths",
    "LatticeModel",
    "Posterior",
    "JointPosterior",
    "AlignmentResult",
    "ProjectionResult",
    "ComparisonResult",
    "PatientQuantities",
    "Landmarks",
    "NoValidPathError",
    "InsufficientSupportError",
    "DegenerateCalibrationError",
    "SimulationConfig",
    "simulate",
    "nonidentifiable_pair",
]
