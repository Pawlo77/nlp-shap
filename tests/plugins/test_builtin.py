"""Tests for built-in plugin registration."""

from nlp_shap.estimation.complementary import ComplementaryEstimator
from nlp_shap.estimation.exact import ExactEstimator
from nlp_shap.estimation.monte_carlo import MonteCarloEstimator
from nlp_shap.estimation.neyman import NeymanEstimator
from nlp_shap.estimation.normalizers import IdentityNormalizer, MinMaxNormalizer
from nlp_shap.masking.partitions import TokenPartitioner
from nlp_shap.masking.policies import DeletePolicy, NeutralPolicy, PadPolicy
from nlp_shap.pipeline.config import ExplainConfig
from nlp_shap.plugins import PluginGroup, PluginRegistry, register_builtin_plugins
from nlp_shap.value.tfidf import TfIdfCosineValue


def test_builtin_plugins_resolve_from_config_keys() -> None:
    """ExplainConfig player and absence keys resolve through the registry."""
    registry = PluginRegistry()
    register_builtin_plugins(registry)
    registry.load_entry_points(PluginGroup.PARTITIONS)
    registry.load_entry_points(PluginGroup.ABSENCE_POLICIES)

    config = ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "stub"},
        "explanation": {"players": "tokens", "absence_policy": "delete"},
    })

    partitioner = registry.resolve(PluginGroup.PARTITIONS, config.explanation.players)
    policy = registry.resolve(
        PluginGroup.ABSENCE_POLICIES,
        config.explanation.absence_policy,
    )

    assert isinstance(partitioner, TokenPartitioner)
    assert isinstance(policy, DeletePolicy)


def test_builtin_registers_all_absence_policies() -> None:
    """Built-in registration exposes delete, pad, and neutral policies."""
    registry = PluginRegistry()
    register_builtin_plugins(registry)

    assert registry.names(PluginGroup.ABSENCE_POLICIES) == ("delete", "neutral", "pad")
    assert isinstance(registry.resolve(PluginGroup.ABSENCE_POLICIES, "pad"), PadPolicy)
    assert isinstance(
        registry.resolve(PluginGroup.ABSENCE_POLICIES, "neutral"),
        NeutralPolicy,
    )


def test_builtin_registers_exact_estimator() -> None:
    """Built-in registration exposes the exact coalition enumerator."""
    registry = PluginRegistry()
    register_builtin_plugins(registry)
    registry.load_entry_points(PluginGroup.ESTIMATORS)

    estimator = registry.resolve(PluginGroup.ESTIMATORS, "exact")
    assert isinstance(estimator, ExactEstimator)
    assert estimator.name == "exact"


def test_builtin_registers_approximate_estimators() -> None:
    """Built-in registration exposes MC, complementary, and Neyman estimators."""
    registry = PluginRegistry()
    register_builtin_plugins(registry)
    registry.load_entry_points(PluginGroup.ESTIMATORS)

    assert registry.names(PluginGroup.ESTIMATORS) == (
        "complementary",
        "exact",
        "mc",
        "neyman_cc",
    )
    assert isinstance(
        registry.resolve(PluginGroup.ESTIMATORS, "mc"), MonteCarloEstimator
    )
    assert isinstance(
        registry.resolve(PluginGroup.ESTIMATORS, "complementary"),
        ComplementaryEstimator,
    )
    assert isinstance(
        registry.resolve(PluginGroup.ESTIMATORS, "neyman_cc"),
        NeymanEstimator,
    )


def test_builtin_registers_value_functions_and_normalizers() -> None:
    """Built-in registration exposes value functions and normalizers."""
    registry = PluginRegistry()
    register_builtin_plugins(registry)
    registry.load_entry_points(PluginGroup.VALUE_FNS)
    registry.load_entry_points(PluginGroup.NORMALIZERS)

    config = ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "stub"},
        "explanation": {"value_fn": "tfidf_cosine", "normalizer": "min_max"},
    })

    value_fn = registry.resolve(
        PluginGroup.VALUE_FNS,
        config.explanation.value_fn,
    )
    normalizer = registry.resolve(
        PluginGroup.NORMALIZERS,
        config.explanation.normalizer,
    )

    assert isinstance(value_fn, TfIdfCosineValue)
    assert isinstance(normalizer, MinMaxNormalizer)
    assert registry.names(PluginGroup.VALUE_FNS) == (
        "embedding_cosine",
        "embedding_euclidean",
        "logprob",
        "tfidf_cosine",
    )
    assert registry.names(PluginGroup.NORMALIZERS) == (
        "abs_sum",
        "identity",
        "min_max",
        "power_shift",
    )
    assert isinstance(
        registry.resolve(PluginGroup.NORMALIZERS, "identity"),
        IdentityNormalizer,
    )
