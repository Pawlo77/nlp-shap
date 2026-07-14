---
name: nlp-shap-development
description: >-
  Ordered implementation workflow for nlp-shap. Use for any code, test,
  packaging, hook, or CI change.
---

# nlp-shap development

Follow rules in `.cursor/rules/*.mdc`. Minimal exit in `AGENTS.md` (`make check`).

## Repo conventions

- Importing `nlp_shap` bootstraps logging from cwd `pyproject.toml` `[tool.logging]`; benchmarks → `make bench`, not `make check`
- Relative imports inside `src/nlp_shap/`; absolute `from nlp_shap...` in `tests/` and `examples/`
- Structured payloads → `TypedDict` + `Literal` wire fields; validate only at external boundaries (`docs.mdc`, `python-quality.mdc`); **every class field gets a one-line docstring** (`python-quality.mdc`)
- `Protocol` stubs: docstring-only bodies — no redundant `...` (`python-quality.mdc`)
- No `from __future__ import annotations` unless required — use `Self` or quoted forward refs (`python-quality.mdc`)
- User-facing `docs/` and `examples/`: shipped API only — no phases or rewrite progress (`docs.mdc`)
- MLLM-Shap port → minimum slice from `papers/MLLM-Shap/mllm_shap/`, not bulk copy

## 1. Read

- Task scope only
- Existing code in `src/nlp_shap/` and `tests/`; `examples/` for notebook style
- Rewrite work → also load skill `nlp-shap-rewrite` and the plan in `nlp-shap-research`

## 2. Red

- Failing test first (`test_*.py`, mirror `src/` paths)
- Bug → regression test before fix
- Test behavior, not trivia

## 3. Green

- Smallest typed implementation that passes
- Reuse before new abstractions; no placeholder packages

## 4. Document

Ship Sphinx docs with every public API / algorithm change (see `docs.mdc`):

- `docs/theory/<topic>.rst` — definitions, formulas, paper links
- `docs/guides/<topic>.rst` — runnable usage
- `docs/api.rst` — `automodule` for new public modules
- `docs/release_notes.rst` — user-facing bullets per version; `Unreleased` until tag
- `docs/index.rst` toctree
- Notebook in `examples/` when users need an end-to-end story; catalog in `examples/README.md` and `docs/examples.rst`
- Example notebooks must cover the **full public usability scope** of the feature they present, with **all imports in the first code cell**, and **short dual-audience section commentary** (technical behavior + business motivation). Omit stubs with no runnable behaviour. Catalog every notebook in `examples/README.md`, `docs/examples.rst`, and `README.md`. See `examples.mdc`.

## 5. Refactor

- Simplify while green; no unrelated cleanup

## 6. Validate

After each task, **test and analyze your own work** before hand-off:

- Re-read the diff for scope creep, style drift, and rule violations (bare `*`, docstrings, imports, docs scope)
- No `from __future__ import annotations` unless required (`python-quality.mdc`)
- Run `make check`; run `make docs` when API or docs changed; run `make notebooks` when example notebooks changed
- Confirm tests cover real behavior and meaningful edge cases — not only happy paths or trivia
- Cross-check consistency with adjacent modules, plugin registration, examples catalog, and release notes
- **Performance review** — analyze hot paths for time/memory scaling; prefer streaming, batching, and lowest-resource correct implementations (see `performance.mdc`)
- Fix every issue found during validation; do not hand off known defects

```bash
make notebooks   # when examples/*.ipynb changed — commit only with outputs
make check
make docs
```

Packaging → `make build` · deps → `uv lock`

New connector → `tests/benchmarks/` + `make bench` (`connector-benchmarks.mdc`)

## 7. Hand off

Stop unless user asks: commit, push, PR (`git-commits.mdc`).

Before push: `make check` green; notebooks executed if changed; `make docs` if docs changed.

After push: `gh run watch <run-id> --exit-status` for each workflow on that commit — fix failures before moving on.

New Makefile target → `name: ## help` + `.PHONY`

## Pitfalls

`pre-commit` → prek · `pip install` → pyproject + `uv lock` · implement before test → back to step 2 · push without `make check` → remote Prek Checks fail · notebooks without outputs → do not commit · public `feat` without `docs/` → incomplete hand-off · phase/roadmap text in `docs/` → forbidden · partial example notebooks that omit shipped APIs → incomplete hand-off
