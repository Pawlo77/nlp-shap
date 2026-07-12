Cooperative games
=================

``nlp-shap`` explains model behaviour by treating input units (tokens, segments,
etc.) as **players** in a cooperative game. Each coalition :math:`S \subseteq N`
of players yields a **payoff** :math:`v(S)` from a characteristic function
:math:`v` defined by a value function and model inference.

Characteristic function
-----------------------

A cooperative game is given by:

- a finite player set :math:`N = \{1, \dots, n\}`
- a function :math:`v : 2^N \to \mathbb{R}` with :math:`v(\emptyset)` as the
  empty-coalition reference

In explainability, :math:`v(S)` is the utility of the model output when only
the players in :math:`S` are **present** (others are masked according to an
absence policy). The grand coalition :math:`v(N)` is the unmasked input.

Marginal contributions
----------------------

For player :math:`i` and coalition :math:`S \subseteq N \setminus \{i\}` the
marginal contribution is:

.. math::

   \Delta_i(S) = v(S \cup \{i\}) - v(S)

Attribution methods differ in how they **weight** and **aggregate** marginal
contributions across coalitions — see :doc:`estimands`.

Estimand aggregators
--------------------

The :class:`~nlp_shap.estimation.estimands.shapley.ShapleyAggregator` and
:class:`~nlp_shap.estimation.estimands.banzhaf.BanzhafAggregator` classes
aggregate precomputed coalition payoffs. You supply boolean coalition masks and
aligned values :math:`v(S)`.

References
----------

- Shapley, L. S. (1953). *A Value for n-person Games*. In *Contributions to the
  Theory of Games II*, Annals of Mathematics Studies 28, Princeton University
  Press. `Project Euclid <https://projecteuclid.org/journals/contributions-to-the-theory-of-games/volume-2/issue-none/A-Value-for-n-person-Games/10.1215/sm/1175702444.full>`__
- Owen, G. (1972). *Multilinear extensions of games*. Management Science.
  `DOI:10.1287/mnsc.18.5.64 <https://doi.org/10.1287/mnsc.18.5.64>`__
