Simulation Modes
================

monoprop provides two physical pictures (Heisenberg and Schrödinger) and three
execution modes that control when and how coefficient work is performed. This
page explains the differences and when to choose each.

Heisenberg and Schrödinger pictures
-------------------------------------

Both pictures compute the same expectation value
:math:`E = \mathrm{Tr}[\varrho H]` but evolve different objects through the
unitary sequence.

Heisenberg picture (default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Hamiltonian is propagated forward while the reference state :math:`\varrho`
is kept fixed:

.. math::

   H_L = U_L^\dagger H_{L-1} U_L,
   \qquad E = \mathrm{Tr}[\varrho\, H_L].

Generators are compiled in reverse order so that replay walks the circuit
in the correct forward direction.

The expectation value gradient with respect to the :math:`k`-th parameter is the
expectation value of the commutator in the reference state:

.. math::

   \frac{\partial E}{\partial \theta_k}
   = \mathrm{Tr}\!\left[\varrho\, [G_k, H_k]\right],

where :math:`H_k` is the Hamiltonian evolved through layers :math:`1,\ldots,k`.
This is the standard mode for ADAPT-VQE inner loops.

Schrödinger picture
~~~~~~~~~~~~~~~~~~~~~

The quantum state is evolved instead of the Hamiltonian:

.. math::

   \varrho_L = U_L\,\varrho_{L-1}\,U_L^\dagger,
   \qquad E = \mathrm{Tr}[H\,\varrho_L].

Generators are compiled in forward order. A separate ``schrodinger_cutoff``
controls state-side truncation independently of the Hamiltonian cutoff, allowing
different approximation levels for the two operators:

.. code-block:: python

   sim = MonomialPropagator(
       ...,
       cutoff=6,
       schrodinger_cutoff=4,   # state truncated more aggressively
   )

Choosing between pictures
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the **Heisenberg picture** for most variational workflows. The ADAPT-VQE
gradient selection complexity is :math:`\mathcal{O}(N^{\ell+p})` in Heisenberg
picture, where :math:`2\ell` is the Hamiltonian cutoff and :math:`2p` is the
generator length.

Use the **Schrödinger picture** when the evolved state is the primary quantity,
or when the state is expected to remain more compact than the Hamiltonian.

Three execution modes
----------------------

Within either picture, ``propagate`` supports three modes that
trade memory and flexibility.

Mode 1 — Graph only
~~~~~~~~~~~~~~~~~~~~~

Supply only the generator list. The simulator compiles each generator into a
graph layer but does not apply it to any coefficient vector:

.. code-block:: python

   sim.propagate(majoranas)

Coefficient work is deferred entirely to the first functional call. This is the
default ADAPT-VQE workflow: compile the circuit once, then optimise parameters
by replaying the stored graph. Memory cost is proportional to graph size.

Mode 2 — Graph with live coefficient truncation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Supply generators together with parameter values and a coefficient vector.
The graph is built layer by layer while simultaneously applying each gate to
the supplied coefficient vector:

.. code-block:: python

   sim.propagate(
       majoranas, param_inds, gen_coeffs, parameters, operator_coeffs
   )

Terms below ``lower_atol`` or failing the structural cutoff are discarded
immediately and do not appear in the graph. This mode is useful when parameter
values are known at circuit construction time and live pruning by coefficient
magnitude is desired during circuit building.

Mode 3 — Immediate contraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Supply generators and parameter values but no coefficient vector. Each gate is
compiled, applied to the current operator, and the layer is immediately
discarded:

.. code-block:: python

   sim.propagate(majoranas, param_inds, gen_coeffs, parameters)

The graph is never stored. Use when only the final evolved operator is needed
and memory is constrained. This mode cannot be used with ``expectation_value_functional``
or ``expectation_value_and_gradient_functional``, which require the stored graph.

Parameter mapping
------------------

The effective rotation angle for generator :math:`i` is

.. math::

   \theta_i^{\mathrm{eff}}
   = g_i \cdot \theta_{\sigma(i)},

where :math:`g_i = \text{gen\_coeffs}[\sigma(i)]` is a per-generator scaling
factor and :math:`\sigma(i) = \text{param\_inds}[i]` is the index into the
``parameters`` vector. Multiple generators can share a parameter by mapping to
the same ``param_inds`` entry. The full parameter-mapping mechanics are
described in :doc:`/internals/evolution_path`.
