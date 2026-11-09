from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

__all__ = [
    "align",
    "v3",
    "pl",
    "tl",
    "TuningResult",
    "is_method_tunable",
    "tunable_methods",
    "tune_method",
    "tune_scdebussy",
    "tune_cellalign_consensus",
    "tune_cellalign_fixed_reference",
    "tune_genes2genes_consensus",
    "tune_genes2genes_fixed_reference",
]

try:
    __version__ = version("scDeBussy")
except PackageNotFoundError:
    __version__ = "0.0.1+source"


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = (
        import_module(f".{name}", __name__)
        if name in {"align", "pl", "tl", "v3"}
        else getattr(import_module(".tuning", __name__), name)
    )
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
