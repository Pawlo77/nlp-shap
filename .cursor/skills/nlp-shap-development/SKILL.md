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
- Existing code in `src/nlp_shap/` and `tests/` for conventions
- MLLM-Shap port → minimum slice, not bulk copy

## 2. Red

- Failing test first (`test_*.py`, mirror `src/` paths)
- Bug → regression test before fix
- Test behavior, not trivia

## 3. Green

- Smallest typed implementation that passes
- Reuse before new abstractions; no placeholder packages

## 4. Refactor

- Simplify while green; no unrelated cleanup

## 5. Verify

```bash
make check
```

Packaging → `make build` · docs → `make docs` · deps → `uv lock`

New connector → `tests/benchmarks/` bench tests + `make bench` (see `connector-benchmarks` rule)

## 6. Hand off

Stop unless user asks: commit, push, PR, new markdown, doc expansion.

New Makefile target → `name: ## help` + `.PHONY`

## Pitfalls

`pre-commit` → prek · `pip install` → pyproject + `uv lock` · implement before test → back to step 2
