"""Micro-benchmarks for archive persistence."""

import time
from pathlib import Path

import pytest

from nlp_shap.domain.estimands import Estimand
from nlp_shap.masking.codec import MaskCodec
from nlp_shap.pipeline.manifest import RunManifest
from nlp_shap.runtime.archive import CoalitionRecordDraft, RunArchive

from ._regression import BaselineStore

_MANIFEST = RunManifest(estimand=Estimand.SHAPLEY, run_id="bench-archive")
_PACKED = MaskCodec.pack((True, False, True, False))


@pytest.mark.bench
def test_archive_round_trip_10k(
    tmp_path: Path,
    baseline_store: BaselineStore,
) -> None:
    """Appending and lazy-reading 10k coalition rows stays within budget."""
    root = tmp_path / "bench-run"
    start = time.perf_counter()
    with RunArchive.open(root, _MANIFEST, flush_every=250) as archive:
        for index in range(10_000):
            archive.append(
                CoalitionRecordDraft(
                    snapshot_id="bench-snapshot",
                    coalition_key=f"key-{index % 128}",
                    mask=_PACKED,
                    absence_policy="pad",
                    model_id="mock",
                    generation_text=f"generation-{index}",
                    utility=float(index % 100),
                    elapsed_ms=1.0,
                    cache_hit=index % 2 == 0,
                )
            )

    with RunArchive.load(root) as archive:
        count = sum(1 for _ in archive.history_lazy())

    elapsed = time.perf_counter() - start
    assert count == 10_000
    baseline_store.check("archive_round_trip_10k", elapsed, ceiling_s=8.0)
