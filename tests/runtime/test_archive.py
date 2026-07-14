"""Tests for run archive persistence."""

import json
from pathlib import Path

import pytest

from nlp_shap.domain.estimands import Estimand
from nlp_shap.masking.codec import MaskCodec
from nlp_shap.pipeline.manifest import RunManifest, parse_manifest
from nlp_shap.runtime.archive import CoalitionRecordDraft, RunArchive


def test_run_archive_round_trips_records(
    tmp_path: Path,
    run_manifest: RunManifest,
    sample_snapshot,
) -> None:
    """One hundred coalition rows round-trip through SQLite and blob storage."""
    root = tmp_path / "run-1"
    packed = MaskCodec.pack((True, False, True))

    with RunArchive.open(root, run_manifest, flush_every=25) as archive:
        for index in range(100):
            archive.append(
                CoalitionRecordDraft(
                    snapshot_id=sample_snapshot.snapshot_id,
                    coalition_key=f"key-{index % 10}",
                    mask=packed,
                    absence_policy="delete",
                    model_id="mock",
                    generation_text=f"generation-{index}",
                    utility=float(index),
                    elapsed_ms=float(index),
                    cache_hit=index % 2 == 0,
                )
            )

    manifest_payload = parse_manifest(
        json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    )
    assert manifest_payload.estimand is Estimand.SHAPLEY
    assert manifest_payload.run_id == "runtime-test"

    with RunArchive.open(root, run_manifest) as archive:
        records = list(archive.history_lazy())

    assert len(records) == 100
    assert records[0].generation_text == "generation-0"
    assert records[-1].generation_text == "generation-99"
    assert records[10].utility == 10.0


def test_history_lazy_reads_blobs_incrementally(
    tmp_path: Path,
    run_manifest: RunManifest,
    sample_snapshot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lazy history loads one blob per iterated record instead of bulk reads."""
    root = tmp_path / "run-2"
    packed = MaskCodec.pack((True, True))

    with RunArchive.open(root, run_manifest) as archive:
        for index in range(100):
            archive.append(
                CoalitionRecordDraft(
                    snapshot_id=sample_snapshot.snapshot_id,
                    coalition_key=f"key-{index}",
                    mask=packed,
                    absence_policy="pad",
                    model_id="mock",
                    generation_text=f"text-{index}",
                    utility=1.0,
                    elapsed_ms=1.0,
                    cache_hit=False,
                )
            )

    read_count = 0
    original_read_text = Path.read_text

    def counting_read_text(self: Path, *args: object, **kwargs: object) -> str:
        nonlocal read_count
        if self.parent.name == "blobs":
            read_count += 1
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read_text)

    with RunArchive.open(root, run_manifest) as archive:
        for index, record in enumerate(archive.history_lazy()):
            assert record.generation_text == f"text-{index}"
            if index == 49:
                break

    assert read_count == 50
