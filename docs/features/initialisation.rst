Initialisation and updates
==========================

The top-level class is ``MonomialPropagator`` (Python) /
``MonomialPropagator<NumModes>`` (C++). Every significant aspect of the
simulation can be tuned at construction time and updated at runtime without
rebuilding. The structural and coefficient cutoffs are covered separately in
:doc:`/features/cutoff`.

Heisenberg and Schrödinger pictures
-----------------------------------

Pass ``schrodinger_cutoff`` to switch from the default Heisenberg picture
(operator evolution) to the Schrödinger picture (density matrix evolution). In the
Schrödinger picture gates are applied to the state, whereas in the Heisenberg
picture they are applied to the observable.

``schrodinger_cutoff`` is the truncation applied to the initial state. It must be
set higher than the regular ``cutoff`` for the Schrödinger-picture result to
match the Heisenberg-picture one: under length truncation the two pictures agree
on fermionic circuits when ``schrodinger_cutoff = cutoff + 2``
(see Appendix of :cite:`chakraborty2026scalablequantumcircuitgeneration`).


Hamiltonian updates
-------------------

The coefficients of the initial Hamiltonian can be patched after construction
without rebuilding the simulator or the graph:

.. code-block:: python

   # Keys are Majorana monomials (tuples of indices); values are the new
   # coefficients, which must keep each monomial Hermitian (real for length-4
   # terms, imaginary for length-2 — see the Notation page).
   sim.update_coeffs({(0, 1, 2, 3): 0.5})

Only terms that already exist in the initial operator can be updated; supplying a
monomial that is not present (or a non-Hermitian coefficient) raises
``RuntimeError``. This is particularly useful
in the Schrödinger picture: since the state is evolved independently of the
observable, you can re-weight the observable and read it off against the same
evolved state, estimating several observables without re-propagating.
