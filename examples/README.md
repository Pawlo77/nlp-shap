<div align="center">
  <h1>Examples</h1>
  <p><strong>End-to-end Jupyter notebooks for nlp-shap explainability workflows.</strong></p>
</div>

## Setup

Install the package, then open a notebook with Jupyter or VS Code.

From PyPI:

```bash
pip install nlp-shap
```

From a local clone:

```bash
git clone https://github.com/Pawlo77/nlp-shap
cd nlp-shap
make install
```

Use the project `.venv` as the notebook kernel when working from source.

## Notebooks

- `estimands_toy_game.ipynb` — Shapley vs Banzhaf on a three-player majority game; shows estimand divergence and `ExplainResult` / manifest labelling. CPU-only; no optional extras.
