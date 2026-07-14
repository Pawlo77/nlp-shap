"""Tests for plugin registry entry-point loading."""

from nlp_shap.backends.mock import MockBackend
from nlp_shap.estimation.estimands.banzhaf import BanzhafAggregator
from nlp_shap.estimation.estimands.shapley import ShapleyAggregator
from nlp_shap.plugins import PluginGroup, PluginRegistry


def test_registry_loads_estimand_entry_points() -> None:
    """Packaging entry points resolve to concrete estimand aggregators."""
    registry = PluginRegistry()
    registry.load_entry_points(PluginGroup.ESTIMANDS)

    shapley = registry.resolve(PluginGroup.ESTIMANDS, "shapley")
    banzhaf = registry.resolve(PluginGroup.ESTIMANDS, "banzhaf")

    assert isinstance(shapley, ShapleyAggregator)
    assert isinstance(banzhaf, BanzhafAggregator)
    assert registry.names(PluginGroup.ESTIMANDS) == ("banzhaf", "shapley")


def test_registry_loads_value_fn_and_normalizer_entry_points() -> None:
    """Packaging entry points resolve to value functions and normalizers."""
    registry = PluginRegistry()
    registry.load_entry_points(PluginGroup.VALUE_FNS)
    registry.load_entry_points(PluginGroup.NORMALIZERS)

    value_fn = registry.resolve(PluginGroup.VALUE_FNS, "tfidf_cosine")
    normalizer = registry.resolve(PluginGroup.NORMALIZERS, "identity")

    assert value_fn.name == "tfidf_cosine"
    assert normalizer.name == "identity"
    assert "logprob" in registry.names(PluginGroup.VALUE_FNS)
    assert "min_max" in registry.names(PluginGroup.NORMALIZERS)


def test_registry_loads_backend_entry_points() -> None:
    """Packaging entry points resolve to backend constructors."""
    registry = PluginRegistry()
    registry.load_entry_points(PluginGroup.BACKENDS)

    backend_type = registry.resolve_type(PluginGroup.BACKENDS, "mock")
    backend = backend_type("stub")
    assert isinstance(backend, MockBackend)
    assert backend.model_id == "stub"


def test_registry_register_and_resolve_round_trip() -> None:
    """Manual registration resolves to the registered factory."""
    registry = PluginRegistry()
    sentinel = object()
    registry.register(PluginGroup.VALUE_FNS, "stub", lambda: sentinel)
    assert registry.resolve(PluginGroup.VALUE_FNS, "stub") is sentinel


def test_registry_resolve_unknown_plugin_raises() -> None:
    """Unknown plugin names raise a lookup error."""
    registry = PluginRegistry()
    try:
        registry.resolve(PluginGroup.BACKENDS, "missing")
    except LookupError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("expected LookupError")
