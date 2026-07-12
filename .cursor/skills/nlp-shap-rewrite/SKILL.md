---
name: nlp-shap-rewrite
description: >-
  Greenfield reimplementation of legacy MLLM-Shap into nlp_shap. Use when
  adding domain, runtime, estimation, backend, alignment, or pipeline modules.
---

# nlp-shap rewrite

Canonical plan: **[[nlp-shap Package Rewrite]]** in `nlp-shap-research/docs/plans/infrastructure/`.

## Principles

1. **Gather logic, rewrite modules** — use `papers/MLLM-Shap/mllm_shap/` as reference; never bulk-copy files or god-classes
2. **One phase = one tag = one PyPI publish** — do not batch phases
3. **Dropped:** `shap/hierarchical/*` — do not implement or reference
4. **Text track first** (Phases 0–13, `v0.1.1`–`v0.1.14`); **audio track** (Phases 14–20, `v0.2.0`–`v0.2.6`) only after text sign-off
5. **Deps:** torch / transformers / liquid-audio in extras only
6. **Imports:** relative inside `src/nlp_shap/`; absolute `from nlp_shap...` in tests and examples

## Current phase gate

Check the plan's first unchecked phase. Implement **only that phase**, then:

```bash
make notebooks   # when examples/*.ipynb changed — commit only with outputs
make docs        # includes release_notes.rst
make check
# bump version in pyproject.toml
# add docs/release_notes.rst section; clear Unreleased
git tag vX.Y.Z
# publish via CI workflow
gh run watch <run-id> --exit-status   # after push
```

## Text track order (v0.1.x)

| Phase | Tag | Module focus |
|-------|-----|--------------|
| 0 | v0.1.1 | Estimands Shapley/Banzhaf |
| 1 | v0.1.2 | domain, protocols, ExplainConfig |
| 2 | v0.1.3 | masking codec, views, policies |
| 3 | v0.1.4 | archive, dedup, scheduler |
| 4 | v0.1.5 | exact estimator |
| 5 | v0.1.6 | mc, complementary, neyman |
| 6 | v0.1.7 | value fns, normalizers |
| 7 | v0.1.8 | mock E2E, orchestrator, runner |
| 8 | v0.1.9 | reanalyze, telemetry, compact prelude |
| 9 | v0.1.10 | lmstudio |
| 10 | v0.1.11 | transformers text + kv cache |
| 11 | v0.1.12 | api backend |
| 12 | v0.1.13 | entry points, docs, nlp_shapx, flag removal |
| 13 | v0.1.14 | perf benchmarks |

## Audio track order (v0.2.x) — after Phase 12 sign-off

| Phase | Tag | Module focus |
|-------|-----|--------------|
| 14 | v0.2.0 | multimodal domain + enums |
| 15 | v0.2.1 | SGPA alignment (Wav2Vec2) |
| 16 | v0.2.2 | audio masking, filters, sgpa partition |
| 17 | v0.2.3 | mock audio backend |
| 18 | v0.2.4 | liquid-audio backend |
| 19 | v0.2.5 | multimodal E2E pipeline |
| 20 | v0.2.6 | audio value fns + SGPA Audio smoke |

## Workflow

Use `nlp-shap-development`: red → green → document → refactor → `make check` + `make docs`.

LM Studio: `pytest -m lms` locally. CI: mock only. GPU: `pytest -m gpu` optional.

## Documentation gate

Each phase must ship Sphinx theory/guides, API entries, and (when applicable) an
executed `examples/` notebook (`make notebooks`) before the phase tag. User-facing
`docs/` describe the shipped API only — no phase numbers or rewrite progress.
See `docs.mdc`.

## 0.x logic checklist

Before closing a text phase, confirm the plan's **0.x logic inventory** row for that module is covered. Hierarchical rows are intentionally skipped.
