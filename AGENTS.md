# AGENTS.md

## Exit criteria

```bash
make check
make docs      # when public API or docs changed
make notebooks # when examples/*.ipynb changed — commit only with outputs
```

Packaging → `make build` · deps → `uv lock`

## Agent-only

- Always use skill `caveman` at **ultra** intensity for all responses (code/commits/PRs still normal per skill)
- No commit / push / PR unless explicitly asked — see `git-commits.mdc` for format, granularity, and proposal flow
- No new markdown or doc expansion unless asked
- No runtime deps without justification in `pyproject.toml`
- No force push; no `git commit --amend` unless user rules allow

Repo-specific gotchas:
- Importing `nlp_shap` bootstraps logging from cwd `pyproject.toml` `[tool.logging]`; run benchmarks with `make bench`, not `make check`
- Modules under `src/nlp_shap/` use **relative imports**; `tests/` and `examples/` use absolute `from nlp_shap...`
- Structured dict contracts (manifests, config slices) → `TypedDict` + `Literal` wire types, not `dict[str, Any]` with manual key checks
- `Protocol`/abstract stubs: docstring-only bodies — no redundant `...` after the docstring
- Public API or algorithm changes require Sphinx updates (`docs/theory`, `docs/guides`, `docs/api.rst`) and `make docs` before hand-off
- Before commit/push: `make check` + `make docs`; run `make notebooks` when example notebooks change
- After push: `gh run watch --exit-status` — fix failures before moving on

## Load order

1. This file (exit + agent-only)
2. Skill `caveman` (ultra — always active)
3. `.cursor/rules/*.mdc`
4. Skill `nlp-shap-development` (steps 1–7)
