Simulation modes
=================

Every simulation computes the same expectation value,

.. math::

   f_L(\theta) = \mathrm{Tr}\!\left[\varrho\, C_L(\theta)^\dagger H\, C_L(\theta)\right],

for an observable :math:`H`, a reference state :math:`\varrho`, and a
parameterised circuit :math:`C_L(\theta)` (see :doc:`/concepts/algorithm`). What
differs between the two **simulation modes** is *which* object is expanded as a sum
of Majorana monomials and pushed through the circuit: the observable
(**Heisenberg picture**) or the state (**Schrödinger picture**). The mode is chosen
at construction time — see :doc:`/features/initialisation`.


Heisenberg picture (default)
----------------------------

The observable is expanded as :math:`H = \sum_\nu c_\nu M_\nu` and propagated
*backwards* through the circuit, applying the gates in reverse order while the
reference state is held fixed,

.. math::

   H_{j+1} = U_{L-j}^\dagger\, H_j\, U_{L-j}, \qquad H_0 = H,

and the expectation value :math:`\mathrm{Tr}[\varrho\, H_L]` is read off at the end
(see :doc:`/concepts/algorithm`).

Schrödinger picture
-------------------

The roles are swapped: the state :math:`\varrho` is expanded in the Majorana basis
and evolved *forwards* through the circuit, with the observable held fixed.
The two pictures are not just cosmetic relabellings of the same computation, but
under common assumptions for fermionic circuits the results of length truncation
are provably equivalent in both
:cite:`chakraborty2026scalablequantumcircuitgeneration`. In practice, matching the
Heisenberg-picture result requires a slightly looser cutoff on the state; see
:doc:`/features/initialisation` for the ``schrodinger_cutoff`` setting.
The Schrödinger picture is less common in practice, but can be useful when you
need to compute the expectation value of many different observables against the
same state.

See also
--------

- :doc:`/concepts/algorithm` — the back-propagation loop, gate application, and
  truncation, written for the Heisenberg picture.
- :doc:`/features/initialisation` — how to select the picture at construction time.
