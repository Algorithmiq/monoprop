Expectation values and gradients
=================================

Once the circuit has been propagated with
:meth:`~monoprop.MajoranaPropagator.build_graph`, the stored graph can be
replayed at any parameter vector without re-running the Majorana algebra. The graph
owns the gate information (which parameter drives each layer and its generator
coefficient), so evaluation takes **only the parameter values**. This is a very
powerful feature for example in variational workflows, where the same circuit is
evaluated repeatedly at many parameter values.

The parameter values are given as a plain sequence of floats (a list or numpy array) in
parameter-index order (``values[i]`` is the angle for gates mapped to index ``i`` by the
circuit's ``parameter_mapping``), or as the :class:`~monoprop.Circuit` itself (its
:attr:`~monoprop.Circuit.parameters` are used). To evaluate two independently-authored
circuit halves together, compose them with ``+`` and build the combined circuit in a single
:meth:`~monoprop.MajoranaPropagator.build_graph` call.

Expectation value and gradient
-------------------------------

The simplest path is to evaluate directly:

.. code-block:: python

   expval = sim.expectation_value(parameters)
   expval, grad = sim.expectation_value_and_gradient(parameters)  # single backward pass

The gradient is returned in parameter-index order (``grad[i]`` is the derivative with
respect to the angle at index ``i``), and ``expectation_value_and_gradient`` computes
both quantities in one backward pass over the graph.

Reusable functionals
--------------------

When the same graph is evaluated at many parameter values, build a functional once
and call it repeatedly. ``expectation_value_functional`` and
``expectation_value_and_gradient_functional`` return callables that accept a
parameter vector:

.. code-block:: python

   expval_fn = sim.expectation_value_functional()
   expval = expval_fn(parameters)

   expval_grad_fn = sim.expectation_value_and_gradient_functional()
   expval, grad = expval_grad_fn(parameters)

Tying parameters on a built graph
---------------------------------

The graph owns the parameter mapping, so which angle drives each gate can be re-wired
*after* building — without rebuilding the graph — through the
:attr:`~monoprop.MajoranaPropagator.parameter_mapping` setter. This is a cheap relabel,
useful to tie or untie angles between optimisation stages. The mapping may be given at
either granularity and must be contiguous ``0..n-1``:

.. code-block:: python

   # Per gate (length sim.n_gates): tie every gate to a single shared angle.
   sim.parameter_mapping = [0] * sim.n_gates
   expval = sim.expectation_value([theta])          # one angle drives the whole circuit

   # Back to one distinct angle per gate.
   sim.parameter_mapping = list(range(sim.n_gates))

A per-gate mapping (length :attr:`~monoprop.MajoranaPropagator.n_gates`) is the same
granularity as the authoring circuit's ``parameter_mapping``, so a circuit's mapping is
directly reusable here; a per-layer mapping (length
:attr:`~monoprop.MajoranaPropagator.graph_layers`, finer when a gate bundles several
monomials) is also accepted. Functionals snapshot the mapping when created, so rebuild a
functional to pick up a new one.

Paring
------

Both functionals accept an optional ``pare_threshold`` — an optional speed-up: terms
whose contribution to the expectation value falls below the threshold are *pared*
away, so they no longer have to be tracked through the graph during replay. The
stored graph itself is unchanged, only the replay skips the negligible terms, so this
can dramatically speed up replay cost for sparse graphs (at the expense of some memory
and accuracy):

.. code-block:: python

   pared_expval_fn = sim.expectation_value_functional(pare_threshold=1e-7)

Partial contraction
-------------------

Where paring skips terms during replay, ``contract_partially`` permanently folds
a chosen set of gates *into* the operator, shrinking the graph that remains to be
replayed. The gates are contracted into the initial operator (Heisenberg picture)
or into the reference state (Schrödinger picture), and by default the simulator's
internal graph is updated in place:

.. code-block:: python

   # Fold the graph evaluated at these parameters into the operator, in place.
   sim.contract_partially(parameters)

This is useful when a prefix of the circuit is fixed so their contribution can be baked in
once instead of being replayed on every evaluation. Subsequent functionals only
need to cover the remaining, shorter graph.

Pass ``inplace=False`` to leave the stored graph untouched and only return the
contracted operator coefficients, so the same graph can be reused with different
parameters:

.. code-block:: python

   coeffs = sim.contract_partially(parameters, inplace=False)

To read the fully evolved operator as a dictionary keyed by Majorana indices —
without modifying the simulator — use ``evolved_operator``.
