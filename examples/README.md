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

- `estimands_toy_game.ipynb` — Estimand walkthrough: Shapley/Banzhaf aggregators, coalition weights, majority vs additive games, labelled `ExplainResult`, manifest wire format, and plugin entry points. CPU-only; no optional extras.
- `masking_views.ipynb` — Masking walkthrough: token partition, delete/pad/neutral policies (including custom parameters), `MaskedSnapshot`/`MaskBuilder`, `MaskCodec`, `MaskSpace`, plugin registry, and validation guards. CPU-only; no optional extras.
- `runtime_core.ipynb` — Runtime walkthrough: `RunArchive`, coalition dedup keys, `HotResultStore`, `InferenceScheduler`, and scheduler metrics. CPU-only; no optional extras.
- `exact_estimation.ipynb` — Exact estimation walkthrough: `ExactEstimator`, coalition enumeration, estimand delegation, budget guard, and plugin resolution. CPU-only; no optional extras.
- `estimator_comparison.ipynb` — Eight-player benchmark (≤20 evals) plus five-player fraction sweep 0.1–0.5; MC bias analysis; multi-seed L1 vs exact Shapley. CPU-only.
- `attribution_viz.ipynb` — Mock exact run with `token_text` and `token_bar` renderers; CPU-only with `[viz]`.
- `text_explain_e2e.ipynb` — End-to-end refund-policy prompt explained via LM Studio and transformers on the same `model_id`, with attribution charts. Requires `[lmstudio]`, `[transformers]`, `[viz]`; LM Studio optional at runtime.
