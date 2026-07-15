<div align="center">
  <h1>📦 nlp-shap</h1>
  <p><strong>Multimodal explainability for NLP and multimodal models based on Shapley-style cooperative game theory.</strong></p>
  <p>
    <a href="https://pypi.org/project/nlp-shap/"><img src="https://img.shields.io/pypi/v/nlp-shap.svg" alt="PyPI"></a>
    <a href="https://pypi.org/project/nlp-shap/"><img src="https://img.shields.io/pypi/pyversions/nlp-shap.svg" alt="Python"></a>
    <a href="https://pawlo77.github.io/nlp-shap/"><img src="https://img.shields.io/badge/docs-GitHub%20Pages-blue" alt="Documentation"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
  </p>
</div>

## Installation

```bash
pip install nlp-shap
```

Optional backends and visualization:

```bash
pip install "nlp-shap[transformers]"   # Hugging Face text backend
pip install "nlp-shap[lmstudio]"       # LM Studio SDK
pip install "nlp-shap[api]"            # OpenAI-compatible HTTP API
pip install "nlp-shap[viz]"            # matplotlib token charts
```

The core package does not install PyTorch. See the [extending guide](https://pawlo77.github.io/nlp-shap/guides/extending.html) for plugin entry points.

## Quickstart

```python
from nlp_shap import ExplainConfig, ExplainRunner
from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
from nlp_shap.domain.enums import Role

snapshot = ConversationSnapshot.from_turns((
    Turn(messages=(Message(role=Role.USER, text="refund my order"),)),
))
config = ExplainConfig.model_validate({
    "backend": {"kind": "mock", "model_id": "stub"},
    "explanation": {"estimator": "exact", "value_fn": "tfidf_cosine"},
})
output = ExplainRunner(config).explain_sync(snapshot)
print(output.result.values)
```

From source:

```bash
git clone https://github.com/Pawlo77/nlp-shap
cd nlp-shap
make install
```

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

## Documentation

Full documentation: **[pawlo77.github.io/nlp-shap](https://pawlo77.github.io/nlp-shap/)**

- [Getting started](https://pawlo77.github.io/nlp-shap/guides/getting_started.html)
- [User guide — estimands](https://pawlo77.github.io/nlp-shap/guides/estimands.html)
- [User guide — masking](https://pawlo77.github.io/nlp-shap/guides/masking.html)
- [User guide — runtime](https://pawlo77.github.io/nlp-shap/guides/runtime.html)
- [User guide — exact estimation](https://pawlo77.github.io/nlp-shap/guides/exact.html)
- [User guide — approximate estimation](https://pawlo77.github.io/nlp-shap/guides/approximation.html)
- [User guide — visualization](https://pawlo77.github.io/nlp-shap/guides/visualization.html)
- [Examples (notebooks)](https://pawlo77.github.io/nlp-shap/examples.html)
- [API reference](https://pawlo77.github.io/nlp-shap/api.html)

## Examples

Runnable **Jupyter notebooks** in [`examples/`](examples/):

| Notebook | What it demonstrates |
|----------|----------------------|
| [`estimands_toy_game.ipynb`](examples/estimands_toy_game.ipynb) | Shapley vs Banzhaf aggregators, coalition weights, labelled results, manifests, plugin entry points |
| [`masking_views.ipynb`](examples/masking_views.ipynb) | Token partitioning, absence policies, masked views, mask codec, mask space, plugin registry |
| [`runtime_core.ipynb`](examples/runtime_core.ipynb) | Run archive, coalition dedup, hot LRU cache, async scheduler with bounded concurrency |
| [`exact_estimation.ipynb`](examples/exact_estimation.ipynb) | Exact coalition enumeration, estimand delegation, budget guard, plugin resolution |
| [`estimator_comparison.ipynb`](examples/estimator_comparison.ipynb) | Eight-player cap≤20 benchmark + five-player fraction sweep 0.1–0.5, MC bias analysis |
| [`attribution_viz.ipynb`](examples/attribution_viz.ipynb) | Per-token Shapley visualization with `token_text` and `token_bar` renderers |
| [`text_explain_e2e.ipynb`](examples/text_explain_e2e.ipynb) | Full explain pipeline on a refund FAQ prompt — LM Studio + transformers + charts |

Setup and details: [`examples/README.md`](examples/README.md). Embedded walkthroughs: [docs/examples](https://pawlo77.github.io/nlp-shap/examples.html).

## Development

```bash
make install
make check
```

Run `make help` for other targets.

## License

Apache License 2.0. See [LICENSE](LICENSE).

Copyright 2026 Paweł Pozorski.
