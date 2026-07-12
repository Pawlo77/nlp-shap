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

## 2. Red

- Failing test first (`test_*.py`, mirror `src/` paths)
- Bug → regression test before fix
- Test behavior, not trivia

## 3. Green

- Smallest typed implementation that passes
- Reuse before new abstractions; no placeholder packages
- Relative imports inside `src/nlp_shap/`; absolute `from nlp_shap...` in tests and examples
- Structured payloads → `TypedDict` with `Literal` wire fields; validate only at external boundaries

## 4. Refactor

- Simplify while green; no unrelated cleanup

## 5. Verify

```bash
make check
```

Packaging → `make build` · docs → `make docs` · deps → `uv lock`

New connector → `tests/benchmarks/` bench tests + `make bench` (see `connector-benchmarks` rule)

Entirely new public workflow → minimal Jupyter notebook in `examples/` + catalog line in `examples/README.md` (see `examples.mdc`)

## 6. Hand off

Stop unless user asks: commit, push, PR, new markdown, doc expansion.

New Makefile target → `name: ## help` + `.PHONY`

## Pitfalls

`pre-commit` → prek · `pip install` → pyproject + `uv lock` · implement before test → back to step 2 · `from nlp_shap...` inside `src/nlp_shap/` → use relative imports · `dict[str, Any]` payload guards → use `TypedDict` · `...` after docstring in Protocol stubs → omit
