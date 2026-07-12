nlp-shap
========

Multimodal explainability for NLP and multimodal models based on **Shapley-style
cooperative game theory**. The library separates *what you measure* (estimands),
*how you sample coalitions* (estimators), and *how you score outputs* (value
functions).

.. note::

   **Release 0.1.1 (Phase 0)** ships estimand aggregators (Shapley, Banzhaf),
   result/manifest labelling, and theory documentation. Full explain pipelines,
   masking, and backends follow in later phases — see the
   `rewrite plan <https://github.com/Pawlo77/nlp-shap-research/blob/main/docs/plans/infrastructure/nlp-shap%20Package%20Rewrite.md>`_.

Installation
------------

.. code-block:: bash

   pip install nlp-shap

From source:

.. code-block:: bash

   git clone https://github.com/Pawlo77/nlp-shap
   cd nlp-shap
   make install

Development
-----------

.. code-block:: bash

   make install
   make check
   make docs

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Theory

   theory/cooperative_games
   theory/estimands

.. toctree::
   :maxdepth: 2
   :caption: Guides

   guides/estimands

.. toctree::
   :maxdepth: 2
   :caption: Reference

   examples
   api

Examples
--------

Runnable Jupyter notebooks live in ``examples/``. See :doc:`examples`.
