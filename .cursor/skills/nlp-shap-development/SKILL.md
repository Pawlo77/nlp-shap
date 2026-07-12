---
name: nlp-shap-development
description: >-
  Ordered implementation workflow for nlp-shap. Use for any code, test,
  packaging, hook, or CI change.
---

# nlp-shap development

Follow rules in `.cursor/rules/*.mdc`. Agent exit criteria in `AGENTS.md`.

## 1. Read

- Task scope only
- Existing code in `src/nlp_shap/` and `tests/` for conventions; `examples/` for published notebook style
- MLLM-Shap port → minimum slice, not bulk copy
- `docs/theory/` and `docs/guides/` for prior theory and citation style

## 2. Red

- Failing test first (`test_*.py`, mirror `src/` paths)
- Bug → regression test before fix
- Test behavior, not trivia

## 3. Green

- Smallest typed implementation that passes
- Reuse before new abstractions; no placeholder packages
- Relative imports inside `src/nlp_shap/`; absolute `from nlp_shap...` in tests and examples
- Structured payloads → `TypedDict` with `Literal` wire fields; validate only at external boundaries

## 4. Document

Ship Sphinx docs with every public API / algorithm change (see `docs.mdc`):

- `docs/theory/<topic>.rst` — definitions, formulas, original paper links
- `docs/guides/<topic>.rst` — runnable usage aligned with public API
- `docs/api.rst` — `automodule` for new public modules
- `docs/index.rst` toctree + `make docs` clean build
- Notebook in `examples/` when users need an end-to-end story; update `examples/README.md` and `docs/examples.rst`

## 5. Refactor

- Simplify while green; no unrelated cleanup

## 6. Verify

```bash
make notebooks   # when examples/*.ipynb changed — commit only with outputs
make check
make docs
```

Packaging → `make build` · deps → `uv lock`

New connector → `tests/benchmarks/` bench tests + `make bench` (see `connector-benchmarks` rule)

## 7. Hand off

Stop unless user asks: commit, push, PR.

After push: `gh run watch --exit-status` until remote workflows pass.

New Makefile target → `name: ## help` + `.PHONY`

## Pitfalls

`pre-commit` → prek · `pip install` → pyproject + `uv lock` · implement before test → back to step 2 · `from nlp_shap...` inside `src/nlp_shap/` → use relative imports · `dict[str, Any]` payload guards → use `TypedDict` · `...` after docstring in Protocol stubs → omit · public `feat` without `docs/` → incomplete hand-off · push without `make check` → remote Prek Checks fail · notebooks without outputs → do not commit
