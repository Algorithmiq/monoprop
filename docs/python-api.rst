Python API Reference
====================

This page is the generated reference for the public ``monoprop`` Python package.
It is organised by stage of a workflow: the simulator itself, the operator and
circuit classes you build a problem from, the conversions from external formats,
and the supporting utilities and data types. For the ideas behind these objects
see :doc:`/concepts`; for task-oriented guides see :doc:`/features`.

The simulator
-------------

``MonomialPropagator`` is the entry point: construct it from an operator, a
circuit, and a cutoff, then :py:meth:`~monoprop.MonomialPropagator.propagate` and
read off expectation values and gradients. See :doc:`/features/initialisation`,
:doc:`/features/cutoff`, and :doc:`/features/evaluation`.

.. automodule:: monoprop.monomial_propagator
   :members:
   :undoc-members:
   :show-inheritance:

Operator and circuit interface
------------------------------

A problem is defined by an **operator** together with a **reference state**, and
a **circuit** of Majorana rotations. monoprop accepts operators in three
equivalent forms — fermionic, qubit (Pauli), and Majorana — each converted into
the internal Majorana representation before simulation. See
:doc:`/concepts/interface` for the conceptual overview.

Fermionic and Majorana operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The natural starting point for quantum chemistry and lattice models: weighted
sums of fermionic creation/annihilation operators (``FermiOperator``), or
monomials given directly as sets of Majorana indices (``MajoranaOperator``).
``FermiEvGate`` and ``FermiCircuit`` describe the evolution and the reference
state.

.. automodule:: monoprop.fermi_data
   :members:
   :undoc-members:
   :show-inheritance:

Qubit (Pauli) operators
~~~~~~~~~~~~~~~~~~~~~~~~~

The natural starting point when you already hold a qubit Hamiltonian:
real-coefficient sums of Pauli strings (``PauliOperator``), with ``PauliEvGate``
and ``PauliEvCircuit`` describing the circuit. To truncate by qubit Pauli weight,
pair these with a ``basis_change`` (see :func:`monoprop.jordan_wigner_basis_change`
and :doc:`/features/cutoff`).

.. automodule:: monoprop.pauli_data
   :members:
   :undoc-members:
   :show-inheritance:

Conversions
-----------

Bridges from external operator formats into the monoprop classes above.

From Qiskit
~~~~~~~~~~~

Convert between Qiskit ``SparsePauliOp`` / ``QuantumCircuit`` and the monoprop
Pauli classes. Requires the optional ``qiskit`` dependency
(``pip install monoprop[qiskit]``).

.. automodule:: monoprop.qiskit_conversion
   :members:
   :undoc-members:
   :show-inheritance:

From electronic-structure integrals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build a ``FermiOperator`` from one- and two-body electronic integrals.

.. automodule:: monoprop.integral_conversion
   :members:
   :undoc-members:
   :show-inheritance:


Utilities
---------

.. automodule:: monoprop.utils
   :members:
   :undoc-members:
   :show-inheritance:

Internal representations
------------------------

The lower-level monomial structures the operators above convert into, and the
protocols a custom operator or circuit type must satisfy.

.. automodule:: monoprop.monomial_data
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: monoprop.quantum_data
   :members:
   :undoc-members:
   :show-inheritance:

Exceptions
----------

.. automodule:: monoprop.exceptions
   :members:
   :undoc-members:
   :show-inheritance:
