Truncation Strategies
=====================

Without truncation, applying :math:`L` gates can produce up to :math:`2^L`
active Majorana terms — exponential growth that quickly becomes intractable.
monoprop provides four independent mechanisms to control this growth, each
targeting a different aspect of the approximation.

Structural cutoff
-----------------

The structural cutoff is the primary truncation: any Majorana monomial produced
during a gate application that fails the cutoff predicate is discarded before
being inserted into the operator. Two cutoff *types* are available.

``LengthPairingDistance`` (default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Discards monomials whose **length plus pairing distance** from the reference
Slater determinant exceeds the cutoff value.

The **length** of a monomial is the number of distinct Majorana modes it
involves. A length-2 monomial corresponds to a one-body term; a length-4
monomial to a two-body term.

The **pairing distance** from a reference Fock state :math:`|\Phi\rangle`
counts how many Majorana indices in the monomial fall outside the paired
structure of the reference state. For a monomial with support :math:`\alpha`
and reference occupied orbitals :math:`J`, each mode pair
:math:`\{\gamma_{2j}, \gamma_{2j+1}\}` for :math:`j \in J` accounts for two
paired indices; the pairing distance is the number of indices in :math:`\alpha`
that are not matched into such pairs.

The sum (length + pairing distance) is a physically motivated measure of how
far the monomial departs from the reference in both degree and occupation. For
quantum chemistry Hamiltonians near the Hartree-Fock ground state, this cutoff
retains chemically relevant correlations while discarding high-rank excitations.
A cutoff value of :math:`2\ell` keeps all monomials satisfying
:math:`|M| + \mathrm{pdist}(M, \Phi) \le 2\ell`.

``Mode``
~~~~~~~~~

Discards monomials whose **mode span** (the range from the smallest to the
largest Majorana index in the support) exceeds the cutoff value. This is useful
when the orbital ordering encodes spatial locality: a mode cutoff then enforces
a maximum interaction range in systems with a spatial structure.

Updating at runtime
~~~~~~~~~~~~~~~~~~~~

Both the cutoff value and cutoff type can be changed at any time without
rebuilding the graph:

.. code-block:: python

   sim.update_cutoff(8)
   sim.update_cutoff_type("Mode")

Raising the cutoff allows subsequent gate applications to grow the operator
further. Lowering it restricts future growth but does not retroactively remove
terms already present in the graph.

Coefficient tolerance filtering
--------------------------------

Two absolute tolerance thresholds filter terms by coefficient magnitude,
independently of the structural cutoff.

``lower_atol``
~~~~~~~~~~~~~~~

Discards any new term whose coefficient magnitude falls below ``lower_atol``
during gate application:

.. math::

   \bigl|i\sin(2\theta) \cdot \phi \cdot h_\alpha\bigr| < \text{lower\_atol}
   \;\Rightarrow\; \text{discard the sine branch.}

This is the primary tool for suppressing numerically negligible branches in
long ADAPT-VQE loops. Because the filtering depends on the current parameter
value, it is most effective during Mode 2 evolution (graph with live coefficient
truncation) where actual parameter values are supplied at compilation time.
At :math:`\theta = 0` all sine branches vanish regardless of ``lower_atol``.

``upper_atol``
~~~~~~~~~~~~~~~

Discards any term whose coefficient magnitude exceeds ``upper_atol``. This hard
upper bound is useful for numerical stability in special regimes but is disabled
by default.

Both thresholds can be updated at runtime:

.. code-block:: python

   sim.update_lower_atol(1e-9)
   sim.update_upper_atol(1e-3)

Exact paring (masked execution plan)
------------------------------------

Exact paring is a post-hoc, read-only filter applied at functional-evaluation
time rather than at graph-construction time. Passing a ``pare_threshold`` to
``expectation_value_functional`` or ``expectation_value_and_gradient_functional`` causes monoprop to
build a masked execution plan that omits graph edges whose contribution to the
expectation value is provably below the threshold:

.. code-block:: python

   pared_fn = sim.expectation_value_functional(param_inds, gen_coeffs, pare_threshold=1e-7)

The paring algorithm walks the graph backward from the inner-product end,
collecting only terms that can contribute above the threshold. The original
graph is never modified, so the same compiled graph can be evaluated with or
without paring, and at different thresholds, without recompiling. See
:doc:`/internals/masked_execution_plan` for the construction algorithm.

Accuracy and performance tradeoffs
-------------------------------------

The four knobs act at different stages and combine multiplicatively — paring
acts only on terms that survived the structural cutoff and coefficient filtering:

.. list-table::
   :header-rows: 1

   * - Knob
     - When applied
     - Accuracy effect
     - Cost effect
   * - Structural cutoff value
     - Gate compilation
     - Higher → more accurate
     - Higher → more terms in graph
   * - Structural cutoff type
     - Gate compilation
     - ``LengthPairingDistance`` more selective for chemistry
     - Both are :math:`\mathcal{O}(1)` per monomial check
   * - ``lower_atol``
     - Mode 2 compilation
     - Higher → less accurate
     - Higher → fewer terms (smaller operator)
   * - Exact paring threshold
     - Functional construction
     - Lower threshold → more accurate
     - Lower threshold → larger plan (slower replay)

A common strategy for ADAPT-VQE is to use a moderate ``lower_atol`` during the
outer loop to keep the operator compact, then apply a tight ``pare_threshold``
for the final expectation value evaluation to eliminate low-weight edges and accelerate
convergence checking.
