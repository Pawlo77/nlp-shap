# AGENTS.md

## Exit criteria

```bash
make check
```

Full verify, docs, notebooks, and release steps → skill `nlp-shap-development`.

## Agent-only

- Always use skill `caveman` at **ultra** intensity for all responses (code/commits/PRs still normal per skill)
- No commit / push / PR unless explicitly asked — see `git-commits.mdc`
- No new markdown or doc expansion unless asked
- No runtime deps without justification in `pyproject.toml`
- No force push; no `git commit --amend` unless user rules allow

Keep this file **general**. Implementation and workflow detail live in **skills** and `.cursor/rules/*.mdc` — not here.

## Load order

1. This file (exit + agent-only)
2. Skill `caveman` (ultra — always active)
3. `.cursor/rules/*.mdc`
4. Skill `nlp-shap-development`
5. Skill `nlp-shap-rewrite` when doing package rewrite work
