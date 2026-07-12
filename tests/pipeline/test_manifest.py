"""Tests for run archive manifest schema stubs."""

import pytest

from nlp_shap.domain.estimands import Estimand
from nlp_shap.pipeline.manifest import RunManifest, RunManifestPayload, parse_manifest


def test_run_manifest_requires_estimand() -> None:
    """Manifest dataclass always includes estimand."""
    manifest = RunManifest(estimand=Estimand.BANZHAF, run_id="run-1")
    assert manifest.estimand is Estimand.BANZHAF
    assert manifest.run_id == "run-1"


def test_parse_manifest_accepts_valid_payload() -> None:
    """Typed manifest payloads round-trip estimand wire values."""
    payload: RunManifestPayload = {"estimand": "shapley", "run_id": "run-42"}
    manifest = parse_manifest(payload)
    assert manifest.estimand is Estimand.SHAPLEY
    assert manifest.run_id == "run-42"


def test_run_manifest_to_dict_includes_estimand() -> None:
    """Serialized manifests always expose estimand."""
    manifest = RunManifest(estimand=Estimand.SHAPLEY, run_id="run-7")
    payload = manifest.to_dict()
    assert payload["estimand"] == "shapley"
    assert payload["run_id"] == "run-7"
