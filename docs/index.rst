nlp-shap
========

Multimodal explainability for NLP and multimodal models based on **Shapley-style
cooperative game theory**. The library separates *what you measure* (estimands),
*how you sample coalitions* (estimators), and *how you score outputs* (value
functions).

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Getting started
      :link: guides/getting_started
      :link-type: doc

      Install ``nlp-shap`` and run your first Shapley vs Banzhaf comparison in
      a few lines of Python.

   .. grid-item-card:: User guide
      :link: guides/estimands
      :link-type: doc

      Work with estimand aggregators, explain results, and run-archive manifests.

   .. grid-item-card:: Theory
      :link: theory/cooperative_games
      :link-type: doc

      Cooperative games, Shapley values, Banzhaf indices, and when they diverge.

   .. grid-item-card:: API reference
      :link: api
      :link-type: doc

      Typed public modules, protocols, and re-exported symbols.

   .. grid-item-card:: Examples
      :link: examples
      :link-type: doc

      Runnable Jupyter notebooks, including an embedded toy-game walkthrough.

   .. grid-item-card:: Release notes
      :link: release_notes
      :link-type: doc

      Per-version changelog for published releases.

Quick install
-------------

.. code-block:: bash

   pip install nlp-shap

Requires Python 3.12. See :doc:`guides/getting_started` for a full walkthrough.

.. toctree::
   :hidden:
   :caption: Guides
   :maxdepth: 2

   guides/getting_started
   guides/estimands

.. toctree::
   :hidden:
   :caption: Theory
   :maxdepth: 2

   theory/cooperative_games
   theory/estimands

.. toctree::
   :hidden:
   :caption: Reference
   :maxdepth: 2

   release_notes
   examples
   api
