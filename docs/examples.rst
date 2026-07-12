Examples
========

The ``examples/`` directory contains **Jupyter notebooks** with runnable
``nlp-shap`` workflows. They complement guides, theory pages, and the API
reference.

Setup
-----

Install the package:

.. code-block:: bash

   pip install nlp-shap

From a local clone:

.. code-block:: bash

   make install

Use the project ``.venv`` as the notebook kernel when working from source.

Toy-game walkthrough
--------------------

The notebook below is rendered inline. It compares Shapley and Banzhaf
aggregators on a three-player majority game and shows why estimand labelling
matters.

.. toctree::
   :maxdepth: 1

   notebooks/estimands_toy_game

Gallery
-------

**estimands_toy_game.ipynb**
   Compare Shapley and Banzhaf aggregators on a three-player majority game.
   Demonstrates why estimand labelling matters and how to use
   :class:`~nlp_shap.estimation.estimands.shapley.ShapleyAggregator` and
   :class:`~nlp_shap.estimation.estimands.banzhaf.BanzhafAggregator` directly.
   No optional dependencies beyond ``nlp-shap``.

Source on GitHub: `examples/estimands_toy_game.ipynb
<https://github.com/Pawlo77/nlp-shap/blob/main/examples/estimands_toy_game.ipynb>`_

See also `examples/README.md
<https://github.com/Pawlo77/nlp-shap/blob/main/examples/README.md>`_.
