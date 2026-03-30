import pytest

from scdebussy.tuning import is_method_tunable, tunable_methods, tune_method


def test_tunable_methods_contains_expected_adapters():
    methods = tunable_methods()

    assert "scdebussy" in methods
    assert "cellalign_consensus" in methods
    assert "cellalign_fixed_reference" in methods
    assert "genes2genes_consensus" in methods
    assert "genes2genes_fixed_reference" in methods


def test_is_method_tunable_reports_expected_values():
    assert is_method_tunable("scdebussy")
    assert is_method_tunable("cellalign_consensus")
    assert not is_method_tunable("identity")
    assert not is_method_tunable("genes2genes_pairwise")


def test_tune_method_rejects_unknown_method(sample_adata):
    with pytest.raises(ValueError, match="is not tunable"):
        tune_method("identity", sample_adata, n_trials=2)


def test_tune_method_rejects_unsupported_objective(sample_adata):
    with pytest.raises(ValueError, match="Only objective='unsupervised'"):
        tune_method("scdebussy", sample_adata, objective="supervised", n_trials=2)
