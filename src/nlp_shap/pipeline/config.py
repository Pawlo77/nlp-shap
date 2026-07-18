"""Explain pipeline configuration schema."""

from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ..domain.enums import EmbeddingMode
from ..domain.estimands import Estimand


class BackendConfig(BaseModel):
    """Backend selection and connection parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str
    """Registered backend plugin identifier."""

    model_id: str
    """Model name or repository id passed to the backend."""

    api_host: str | None = None
    """Optional API host for HTTP-compatible backends."""

    hub_search: str | None = None
    """Optional LM Studio repository search term; defaults to ``model_id``."""

    quantization: str = "Q4_K_M"
    """GGUF quantization to download when the model is not already local."""

    auto_download: bool = True
    """Download missing models through the LM Studio repository API."""

    serialize_generate: bool = True
    """Serialize backend ``generate`` calls (needed for MLX LM Studio)."""


class GenerationConfig(BaseModel):
    """Generation parameters for the base and coalition evaluations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_new_tokens: int = 128
    """Maximum tokens generated per coalition evaluation."""

    temperature: float = 0.0
    """Sampling temperature for model generation."""

    top_k: int = 1
    """Top-k sampling cutoff for model generation."""

    precompute_base: bool = True
    """Whether to generate the grand-coalition reference before explaining."""


class BudgetConfig(BaseModel):
    """Estimator budget controls replacing standard/limited split configs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    """Fraction of the full coalition budget used by the estimator."""


class ArchiveConfig(BaseModel):
    """Run archive persistence options."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = "./runs/{run_id}"
    """Filesystem path template for persisted run archives."""

    flush_every: int = Field(default=50, ge=1)
    """Number of coalition records between archive flushes."""


class DedupConfig(BaseModel):
    """Coalition deduplication options."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: Literal["auto", "on", "off"] = "auto"
    """Deduplication mode for repeated coalition evaluations."""


class KvCacheConfig(BaseModel):
    """Prefix cache options for supported backends."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    """Whether prefix/KV cache reuse is enabled when supported."""


class NeymanConfig(BaseModel):
    """Neyman-CC estimator controls used when ``estimator`` is ``neyman_cc``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    use_standard_method: bool = False
    """Whether phase-one sampling uses unstructured random coalitions."""

    initial_fraction: float | None = None
    """Fraction of the total budget used during phase-one initialization."""

    initial_num_samples: int | None = None
    """Explicit phase-one pair budget overriding ``initial_fraction``."""


class ExplanationConfig(BaseModel):
    """Explanation algorithm and runtime controls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    estimand: Estimand = Estimand.SHAPLEY
    """Cooperative-game value targeted by aggregation."""

    estimator: str = "neyman_cc"
    """Registered estimator plugin used to sample coalitions."""

    value_fn: str = "tfidf_cosine"
    """Registered value-function plugin used to score outputs."""

    normalizer: str = "identity"
    """Registered normalizer applied after estimand aggregation."""

    players: str = "tokens"
    """Registered partition plugin that defines explainability players."""

    absence_policy: str = "delete"
    """Registered absence policy used to render masked snapshots."""

    budget: BudgetConfig = BudgetConfig()
    """Estimator budget controls."""

    include_minimal_masks: bool = False
    """Whether minimal coalitions are included in estimator sampling."""

    neyman: NeymanConfig = NeymanConfig()
    """Neyman-CC controls applied when the Neyman estimator is selected."""

    max_inflight: int = Field(default=2, ge=1)
    """Maximum concurrent coalition evaluations."""

    archive: ArchiveConfig = ArchiveConfig()
    """Run archive persistence settings."""

    dedup: DedupConfig = DedupConfig()
    """Coalition deduplication settings."""

    kv_cache: KvCacheConfig = KvCacheConfig()
    """Prefix cache settings for supported backends."""

    embedding_mode: EmbeddingMode = EmbeddingMode.STATIC
    """Embedding mode used by embedding-based value functions."""

    seed: int = 42
    """Random seed for reproducible coalition sampling."""


class ExplainConfig(BaseModel):
    """Top-level explain pipeline configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: BackendConfig
    """Backend selection and connection parameters."""

    generation: GenerationConfig = GenerationConfig()
    """Generation parameters for base and coalition evaluations."""

    explanation: ExplanationConfig = ExplanationConfig()
    """Explanation algorithm and runtime controls."""


def explain_config_from_yaml(text: str) -> ExplainConfig:
    """Parse YAML text into a validated :class:`ExplainConfig`."""
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        msg = "explain config root must be a mapping"
        raise TypeError(msg)
    return ExplainConfig.model_validate(data)


def explain_config_to_yaml(config: ExplainConfig) -> str:
    """Serialize a config to YAML with stable key ordering within sections."""
    payload = config.model_dump(mode="json")
    dumped: str = yaml.safe_dump(payload, sort_keys=False)
    return dumped
