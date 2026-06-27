Branch Simulation
=================

This page describes the core algorithmic loop in monoprop: how a single unitary
gate is applied to a Majorana operator, how the operator grows as a tree of
branches, and how that algebraic structure is compiled into a replayable graph.

Single-gate application
------------------------

The central transformation is the conjugation of an operator
:math:`O = \sum_\alpha h_\alpha \hat{M}_\alpha` by a unitary gate
:math:`U(\theta) = e^{i\theta G}`, where the generator :math:`G = \hat{M}_\gamma`
is a Hermitian Majorana monomial:

.. math::

   U(\theta)^\dagger\, O\, U(\theta)
   = \cos(2\theta)\, O + i\sin(2\theta)\,[G,\, O].

The factor of 2 in the trig argument arises from :math:`G^2 = I` (every
Majorana monomial is involutory), which gives a doubled-frequency expectation value
landscape :math:`f(\theta) = A\cos(2\theta) + B\sin(2\theta) + C`. The gate
acts independently on each term in the operator:

- **Cosine branch**: if :math:`[G, \hat{M}_\alpha] = 0` (overlap even), the
  term is scaled by :math:`\cos(2\theta)`.
- **Sine branch**: if :math:`\{G, \hat{M}_\alpha\} = 0` (overlap odd), a new
  term :math:`\hat{M}_{\alpha \oplus \gamma}` is created scaled by
  :math:`i\sin(2\theta)` times a multiplicative phase from the Majorana algebra
  (see :doc:`/concepts/algebra` for the exact phase formula).

Because each anticommuting pair maps to *exactly one* new monomial, a gate
application produces at most one new term per existing term.

Cycles
------

When the operator already contains both :math:`\hat{M}_\alpha` and
:math:`\hat{M}_{\alpha \oplus \gamma}`, the two terms are coupled: each is the
sine-branch target of the other. This is called a *cycle*. The update is a
symplectic rotation in the two-dimensional subspace spanned by the pair:

.. math::

   \begin{pmatrix} h_\alpha' \\ h_{\alpha\oplus\gamma}' \end{pmatrix}
   =
   \begin{pmatrix}
     \cos(2\theta)        & -\phi\sin(2\theta) \\
     \phi\sin(2\theta)    &  \cos(2\theta)
   \end{pmatrix}
   \begin{pmatrix} h_\alpha \\ h_{\alpha\oplus\gamma} \end{pmatrix},

where :math:`\phi \in \{+1,-1\}` is the combined multiplicative phase. Terms
not involved in any cycle receive only a cosine scaling.

The branch tree
---------------

A sequence of :math:`L` gates applied to an initial operator builds a *tree of
branches*. At each step, each existing term either stays (cosine branch) or
creates one new term (sine branch). In the worst case the number of active
terms doubles at every gate, growing as :math:`2^L`. The structural cutoff (see
:doc:`/concepts/truncation`) prunes sine branches whose new monomial fails the
length or pairing-distance limit, keeping the tree tractable.

Graph compilation
-----------------

Rather than applying each gate to the coefficient vector immediately, monoprop
*compiles* the gate into a layer stored in the evolution graph. Compilation
runs the Majorana algebra once and classifies every term as one of:

- **Cosine-only**: commutes with the generator; receives only a scaling by
  :math:`\cos(2\theta)` at replay time.
- **Local cycle**: both the source and sine-branch target belong to the same
  MPI rank; updated together at replay time.
- **Cross-rank cycle**: source and target belong to different ranks; resolved
  via an MPI exchange at replay time.

One layer is appended to the graph per gate application. The algebraic
structure is computed once; subsequent evaluations replay the stored layer
without re-running Majorana algebra. For the internal storage format, see
:doc:`/internals/graph`.

Graph replay
------------

Evaluating the operator at a concrete parameter vector
:math:`(\theta_1,\ldots,\theta_L)` replays each layer in order:

1. **Cosine pass**: scale all cosine-only coefficients by :math:`\cos(2\theta_l)`.
2. **Local cycle pass**: apply the symplectic rotation to each local
   :math:`(\text{src}, \text{tgt})` pair.
3. **Cross-rank exchange**: exchange coefficient values with remote ranks and
   complete the cross-rank rotations.

Replay cost is :math:`\mathcal{O}(|\text{graph}|)` per call and does not
depend on the number of prior compilation calls. For the kernel-level details,
see :doc:`/internals/graph_evolution`.

Compile-once, replay-many
--------------------------

The separation between compilation and replay is the key architectural property
of monoprop. Variational optimisation proceeds in two phases:

1. **Compilation** (once, during circuit construction): run Majorana algebra,
   resolve cross-rank terms, store graph layers.
2. **Replay** (repeated, during optimisation): evaluate at each trial parameter
   vector using the stored graph; no Majorana algebra is re-run.

Each optimisation iteration costs :math:`\mathcal{O}(|\text{graph}|)`,
independent of the compilation cost. For ADAPT-VQE inner loops where the same
graph is evaluated hundreds of times per parameter step, this separation
provides a decisive performance advantage.

For the three execution modes that govern when graph construction and coefficient
work happen relative to each other, see :doc:`/concepts/simulation_modes`.
