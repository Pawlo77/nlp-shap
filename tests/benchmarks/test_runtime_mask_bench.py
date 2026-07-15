"""Micro-benchmarks for mask pack and unpack throughput."""

import time

import pytest

from nlp_shap.masking.codec import MaskCodec

from ._regression import BaselineStore


@pytest.mark.bench
def test_mask_pack_unpack_10k(baseline_store: BaselineStore) -> None:
    """Packing and unpacking 10k masks stays within budget."""
    masks = [
        tuple(bit == 1 for bit in format(index % 256, "08b")) for index in range(256)
    ]
    start = time.perf_counter()
    for index in range(10_000):
        packed = MaskCodec.pack(masks[index % len(masks)])
        restored = MaskCodec.unpack(packed)
        assert len(restored) == len(masks[index % len(masks)])
    elapsed = time.perf_counter() - start
    baseline_store.check("mask_pack_unpack_10k", elapsed, ceiling_s=0.25)
