"""Hugging Face causal language model backend (text-only)."""

import asyncio
from typing import Any

from ...backends.mock.generation import generation_record_from_snapshot
from ...domain.conversation import ConversationSnapshot
from ...domain.generation import GenerationRecord
from ...errors import BackendUnavailableError
from ...pipeline.config import BackendConfig
from ...runtime.kv_cache import PrefixCacheManager
from .chat import render_prompt, snapshot_has_audio, snapshot_to_chat_messages
from .embeddings import contextual_embedding, static_embedding


class TransformersTextBackend:
    """Generate text with a local Hugging Face causal language model."""

    def __init__(self, config: BackendConfig) -> None:
        self._config = config
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._device: Any | None = None
        self._prefix_cache: PrefixCacheManager | None = None
        self._kv_cache_hits = 0
        self._init_lock = asyncio.Lock()

    @property
    def model_id(self) -> str:
        """Return the configured Hugging Face model identifier."""
        return self._config.model_id

    @property
    def kv_cache_hits(self) -> int:
        """Return the number of prefix-cache hits during this backend lifetime."""
        return self._kv_cache_hits

    def set_kv_cache_enabled(self, enabled: bool) -> None:
        """Enable or disable prefix-cache reuse for coalition generation."""
        if enabled:
            if self._prefix_cache is None:
                self._prefix_cache = PrefixCacheManager()
        else:
            self._prefix_cache = None
            self._kv_cache_hits = 0

    async def generate(
        self,
        snapshot: ConversationSnapshot,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> GenerationRecord:
        """Generate assistant text for ``snapshot`` with optional prefix caching."""
        if snapshot_has_audio(snapshot):
            msg = "audio snapshots are not supported by the transformers text backend"
            raise ValueError(msg)
        await self._ensure_loaded()
        return await asyncio.to_thread(
            self._generate_sync,
            snapshot,
            max_new_tokens,
            temperature,
            top_k,
        )

    async def aclose(self) -> None:
        """Release loaded model references."""
        self._tokenizer = None
        self._model = None
        self._device = None
        if self._prefix_cache is not None:
            self._prefix_cache.reset()

    async def _ensure_loaded(self) -> None:
        async with self._init_lock:
            if self._model is not None and self._tokenizer is not None:
                return
            try:
                torch = _import_torch()
                transformers = _import_transformers()
            except ImportError as exc:
                msg = "transformers backend requires the transformers optional extra"
                raise BackendUnavailableError(msg) from exc
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            try:
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    self._config.model_id,
                )
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    self._config.model_id,
                )
            except Exception as exc:
                msg = f"failed to load transformers model {self._config.model_id!r}"
                raise BackendUnavailableError(msg) from exc
            model.to(device)
            model.eval()
            _ensure_pad_token(tokenizer)
            self._tokenizer = tokenizer
            self._model = model
            self._device = device

    def _generate_sync(
        self,
        snapshot: ConversationSnapshot,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> GenerationRecord:
        tokenizer = self._tokenizer
        model = self._model
        if tokenizer is None or model is None:
            msg = "transformers backend is not loaded"
            raise BackendUnavailableError(msg)
        messages = snapshot_to_chat_messages(snapshot)
        prompt = render_prompt(tokenizer, messages)
        encoded = tokenizer.encode(prompt, add_special_tokens=True)
        token_ids = [int(token_id) for token_id in encoded]
        self._record_prefix_cache_hit(token_ids)
        torch = _import_torch()
        generate_ids = token_ids
        if not generate_ids:
            msg = "transformers backend received an empty prompt"
            raise BackendUnavailableError(msg)
        input_ids = torch.tensor([generate_ids], dtype=torch.long, device=self._device)
        attention_mask = torch.ones_like(input_ids)
        do_sample = temperature > 0.0
        generate_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
            if top_k > 0:
                generate_kwargs["top_k"] = top_k
        with torch.no_grad():
            if self._prefix_cache is not None and token_ids:
                prompt_tensor = torch.tensor(
                    [token_ids],
                    dtype=torch.long,
                    device=self._device,
                )
                for length in range(1, len(token_ids) + 1):
                    prefix_out = model(
                        prompt_tensor[:, :length],
                        use_cache=True,
                    )
                    self._prefix_cache.store(
                        token_ids[:length],
                        prefix_out.past_key_values,
                    )
            generated = model.generate(**generate_kwargs)
        prompt_len = input_ids.shape[1]
        new_token_ids = [
            int(token_id)
            for token_id in generated[0, prompt_len:].detach().cpu().tolist()
        ]
        text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        record = generation_record_from_snapshot(text, snapshot)
        embedding = static_embedding(model, new_token_ids)
        contextual = contextual_embedding(model, new_token_ids)
        return GenerationRecord(
            text=record.text,
            text_token_rows=record.text_token_rows,
            embedding=embedding,
            contextual_embedding=contextual,
        )

    def _record_prefix_cache_hit(self, token_ids: list[int]) -> None:
        if self._prefix_cache is None or not token_ids:
            return
        _, matched_len = self._prefix_cache.get_prefix(token_ids)
        if matched_len > 0:
            if matched_len >= len(token_ids):
                matched_len = len(token_ids) - 1
            if matched_len > 0:
                self._prefix_cache.hits += 1
                self._kv_cache_hits += 1
                return
        self._prefix_cache.misses += 1


def _ensure_pad_token(tokenizer: Any) -> None:
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        msg = "torch is required for the transformers backend extra"
        raise ImportError(msg) from exc
    return torch


def _import_transformers() -> Any:
    try:
        import transformers
    except ImportError as exc:
        msg = "transformers is required for the transformers backend extra"
        raise ImportError(msg) from exc
    return transformers
