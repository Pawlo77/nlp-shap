Examples
========

The ``examples/`` directory contains **Jupyter notebooks** with runnable
``nlp-shap`` workflows. They complement theory pages and the API reference.

Setup
-----

Install the package:

.. code-block:: bash

   pip install nlp-shap

From a local clone:

.. code-block:: bash

   make install

Use the project ``.venv`` as the notebook kernel when working from source.

Gallery
-------

**estimands_toy_game.ipynb**
   Compare Shapley and Banzhaf aggregators on a three-player majority game.
   Demonstrates why estimand labelling matters and how to use
   :class:`~nlp_shap.estimation.estimands.shapley.ShapleyAggregator` and
   :class:`~nlp_shap.estimation.estimands.banzhaf.BanzhafAggregator` directly.
   No optional dependencies beyond ``nlp-shap``.

Source: `examples/estimands_toy_game.ipynb
<https://github.com/Pawlo77/nlp-shap/blob/main/examples/estimands_toy_game.ipynb>`_

See also `examples/README.md
<https://github.com/Pawlo77/nlp-shap/blob/main/examples/README.md>`_.
