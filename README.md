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
- [Examples (notebooks)](https://pawlo77.github.io/nlp-shap/examples.html)
- [API reference](https://pawlo77.github.io/nlp-shap/api.html)

## Examples

Runnable **Jupyter notebooks** in [`examples/`](examples/):

| Notebook | What it demonstrates |
|----------|----------------------|
| [`estimands_toy_game.ipynb`](examples/estimands_toy_game.ipynb) | Shapley vs Banzhaf aggregators, coalition weights, labelled results, manifests, plugin entry points |
| [`masking_views.ipynb`](examples/masking_views.ipynb) | Token partitioning, absence policies, masked views, mask codec, mask space, plugin registry |
| [`runtime_core.ipynb`](examples/runtime_core.ipynb) | Run archive, coalition dedup, hot LRU cache, async scheduler with bounded concurrency |

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
