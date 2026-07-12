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

- `estimands_toy_game.ipynb` — Full 0.1.1 estimand walkthrough: Shapley/Banzhaf aggregators, coalition weights, majority vs additive games, labelled `ExplainResult`, manifest wire format, and plugin entry points. CPU-only; no optional extras.
- `masking_views.ipynb` — Full 0.1.3 masking walkthrough: token partition, delete/pad/neutral policies (including custom parameters), `MaskedSnapshot`/`MaskBuilder`, `MaskCodec`, `MaskSpace`, plugin registry, and validation guards. CPU-only; no optional extras.
- `runtime_core.ipynb` — Full 0.1.4 runtime walkthrough: `RunArchive`, coalition dedup keys, `HotResultStore`, `InferenceScheduler`, and scheduler metrics. CPU-only; no optional extras.
- `exact_estimation.ipynb` — Full 0.1.5 exact estimation walkthrough: `ExactEstimator`, coalition enumeration, estimand delegation, budget guard, and plugin resolution. CPU-only; no optional extras.
