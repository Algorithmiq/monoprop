Operator Representations
=========================

monoprop accepts operators in several input formats and converts them internally to
the Majorana monomial representation used by the simulator. This page describes
each format and the conversion pipeline.

Input formats
-------------

Three high-level input types cover the most common starting points.

Fermionic operators
~~~~~~~~~~~~~~~~~~~~

``FermiString`` represents a sequence of fermionic creation and annihilation
operators on numbered modes:

.. doctest::

   >>> from monoprop.fermi_data import FermiString, FermiOperator
   >>> # c†_0 c_1
   >>> term = FermiString([(0, '+'), (1, '-')])
   >>> term
   FermiString(a_0^+ a_1^-)

``FermiOperator`` is a weighted sum of ``FermiString`` objects:

.. doctest::

   >>> from monoprop.fermi_data import FermiString, FermiOperator
   >>> h = FermiOperator(
   ...     terms=[FermiString([(0, '+'), (1, '-')]),
   ...            FermiString([(1, '+'), (0, '-')])],
   ...     coefficients=[0.5, 0.5],
   ... )
   >>> h.terms
   [FermiString(a_0^+ a_1^-), FermiString(a_1^+ a_0^-)]

Calling ``h.get_monomial_operator()`` converts to the Majorana representation
via the Jordan-Wigner mapping.

Pauli operators
~~~~~~~~~~~~~~~~

``PauliString`` represents a tensor product of single-qubit Pauli operators
(``I``, ``X``, ``Y``, ``Z``):

.. doctest::

   >>> from monoprop.pauli_data import PauliString, PauliOperator
   >>> ps = PauliString("XZIY")   # X on qubit 0, Z on qubit 1, I on qubit 2, Y on qubit 3
   >>> ps
   PauliString('XZIY')

``PauliOperator`` is a real-coefficient weighted sum. It converts to Majorana
form via the Jordan-Wigner mapping when ``get_monomial_operator()`` is called.
This is the natural entry point when starting from a qubit Hamiltonian.

Majorana operators
~~~~~~~~~~~~~~~~~~~~

``MajoranaOperator`` specifies monomials directly as tuples of Majorana mode
indices:

.. doctest::

   >>> from monoprop.fermi_data import MajoranaOperator
   >>> # 0.5 * γ_0 γ_1  +  0.25 * γ_2 γ_3 γ_4 γ_5
   >>> op = MajoranaOperator(
   ...     majoranas=[(0, 1), (2, 3, 4, 5)],
   ...     coefficients=[0.5, 0.25],
   ...     num_modes=None,
   ... )
   >>> op.majoranas
   [(0, 1), (2, 3, 4, 5)]
   >>> op.coefficients
   [0.5, 0.25]

Use this format when the Majorana representation is already known — for
example, when a Jordan-Wigner mapping has been applied externally.

Circuit-level structures
-------------------------

For workflows that precompute the full circuit before simulation, two structured
containers are available.

``MonomialCircuit``
~~~~~~~~~~~~~~~~~~~

``MonomialCircuit`` bundles everything needed to describe a circuit:

.. code-block:: python

   from monoprop.monomial_data import MonomialCircuit

   circuit = MonomialCircuit(
       initial_state=slater_determinant,   # list of occupied mode indices
       majoranas=[(0, 1), (2, 3)],         # generators in order
       parameters=[0.1, -0.3],             # current parameter values
       gen_coeffs=[1.0, 1.0],              # per-generator scaling factors
       param_inds=[0, 1],                  # which parameter each generator uses
       identical_params=False,
   )

This is the natural representation for ADAPT-VQE states where generators grow
one at a time.

``MPData``
~~~~~~~~~~~

``MPData`` is the top-level data class for a complete simulation state,
including the problem definition, circuit parameters, and expected results:

.. code-block:: python

   from monoprop.mp_data import MPData

   data = MPData(
       majoranas=...,               # circuit generators
       gen_coeffs=...,              # per-generator scaling
       param_inds=...,              # parameter indices
       parameters=...,              # variational parameters
       fermionic_hamiltonian=...,   # input Hamiltonian dict
       hartree_fock=...,            # reference Slater determinant
       num_modes=...,
   )

``MPData`` supports serialisation via ``to_msgpack()`` / ``from_msgpack()``
for checkpointing and dataset sharing. It also records the evolved Hamiltonian,
computed energies, and gradients after simulation.

Conversion pipeline
--------------------

The typical flow from a chemistry Hamiltonian to a simulator call is:

1. **Build** the Hamiltonian in ``FermiOperator``, ``PauliOperator``, or
   ``MajoranaOperator`` form.
2. **Convert** to a monomial dict via ``get_monomial_operator()``.
3. **Construct** the simulator with the dict, the Slater determinant reference,
   and a cutoff.
4. **Propagate** with ``propagate`` to compile the circuit into
   the graph.
5. **Evaluate** with expectation value or gradient functionals.

Steps 1–3 happen once; steps 4–5 are the inner loop for variational
optimisation.
