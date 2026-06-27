Expectation value and Gradient Evaluation
=========================================

This page describes how monoprop computes expectation values and parameter
gradients from a compiled evolution graph, and covers the candidate functional mechanism.

Expectation value
-----------------

The target quantity is the expectation value

.. math::

   E(\theta_1,\ldots,\theta_L)
   = \mathrm{Tr}\!\left[\varrho\, H(\theta_1,\ldots,\theta_L)\right].

In the Heisenberg picture,
:math:`H(\theta_1,\ldots,\theta_L) = U_L^\dagger \cdots U_1^\dagger H U_1 \cdots U_L`.
In the Majorana basis the trace reduces to a real dot product because the basis
elements are orthogonal under the trace:

.. math::

   E = \sum_\alpha s_\alpha\, h_\alpha(\theta_1,\ldots,\theta_L),

where :math:`s_\alpha` is the coefficient of :math:`\hat{M}_\alpha` in
:math:`\varrho` and :math:`h_\alpha` is the corresponding coefficient in the
evolved Hamiltonian.

Graph replay for expectation value
----------------------------------

Given a compiled graph, evaluation replays every graph layer in order, applying
cosine scaling and symplectic cycle rotations to the coefficient vector. At the
end, the rank-local dot product of the state and evolved-Hamiltonian coefficient
vectors is summed across all ranks to form the global expectation value scalar.

The expectation value functional is created by ``expectation_value_functional(param_inds,
gen_coeffs)``:

.. code-block:: python

   expval_fn = sim.expectation_value_functional(param_inds, gen_coeffs)
   expval = expval_fn(parameters)

The functional captures a snapshot of the state and Hamiltonian coefficients at
construction time, so later simulator mutations do not affect its output.

Gradient evaluation
--------------------

The full parameter gradient :math:`(\partial E/\partial \theta_1, \ldots,
\partial E/\partial \theta_L)` is computed analytically in a single backward
pass over the graph — no finite differences, no extra forward passes.

Analytic derivative formula
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For layer :math:`k` with scaling :math:`g_k = \text{gen\_coeffs}[\sigma(k)]`,
the chain-rule contribution is

.. math::

   \frac{\partial E}{\partial \theta_k}
   = -2g_k\sin(2g_k\theta_k)\cdot C_k
     + 2g_k\cos(2g_k\theta_k)\cdot S_k,

where :math:`C_k` and :math:`S_k` are the cosine and sine gradient
contributions accumulated at layer :math:`k` during the simultaneous propagation
of the state and Hamiltonian coefficient vectors. This accumulation is exact and
introduces no approximation beyond the structural cutoff already applied to the
operator.

Backward pass
~~~~~~~~~~~~~~

After evolving the Hamiltonian forward, the backward pass walks each graph layer
in reverse, accumulates the per-layer gradient contributions rank-locally, and
finishes with a single global reduction over the full gradient vector. The net
MPI cost is one collective per layer (for cross-rank coefficient exchange) plus
one global sum at the end:

.. code-block:: python

   expval_grad_fn = sim.expectation_value_and_gradient_functional(param_inds, gen_coeffs)
   expval, grad = expval_grad_fn(parameters)

Candidate functionals
----------------------

ADAPT-VQE requires evaluating the expectation value and gradient for prospective gates
before committing them to the circuit. Candidate functionals handle this without
mutating the base simulator:

.. code-block:: python

   candidate_fn = sim.make_candidate_functional(
       candidate_majorana, param_inds, gen_coeffs
   )
   expval_candidate, grad_candidate = candidate_fn(parameters)

The candidate gate is evaluated against the current circuit without being added
to it. Discarding the functional leaves the simulator unchanged.

Exact paring during evaluation
--------------------------------

Both ``expectation_value_functional`` and ``expectation_value_and_gradient_functional`` accept an
optional ``pare_threshold``:

.. code-block:: python

   pared_fn = sim.expectation_value_and_gradient_functional(
       param_inds, gen_coeffs, pare_threshold=1e-7
   )

When set, a masked execution plan is built once at functional-construction time
and reused for all subsequent calls, filtering out graph edges below the
threshold. See :doc:`/concepts/truncation` for the paring concept and
:doc:`/internals/masked_execution_plan` for the construction algorithm.
