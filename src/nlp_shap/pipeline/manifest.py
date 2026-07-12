"""Run archive manifest schema stubs."""

from dataclasses import dataclass
from typing import TypedDict

from ..domain.estimands import Estimand, EstimandWire, estimand_to_wire


class RunManifestPayload(TypedDict):
    """JSON-compatible manifest payload persisted with a run archive."""

    estimand: EstimandWire
    """Estimand label recorded for every attribution output."""

    run_id: str
    """Stable identifier for the explain run."""


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Top-level metadata persisted alongside a run archive."""

    estimand: Estimand
    """Estimand label recorded for every attribution output."""

    run_id: str
    """Stable identifier for the explain run."""

    def to_dict(self) -> RunManifestPayload:
        """Serialize manifest fields to JSON-compatible primitives."""
        return {
            "estimand": estimand_to_wire(self.estimand),
            "run_id": self.run_id,
        }


def parse_manifest(payload: RunManifestPayload) -> RunManifest:
    """Parse a typed manifest payload into a run manifest."""
    return RunManifest(
        estimand=Estimand(payload["estimand"]),
        run_id=payload["run_id"],
    )
