Python API Reference
====================

This page is the generated reference for the public ``monoprop`` Python package.
It is organised by stage of a workflow: the simulator itself, the operator and
circuit classes you build a problem from, the conversions from external formats,
and the supporting utilities and data types. For the ideas behind these objects
see :doc:`/concepts`; for task-oriented guides see :doc:`/features`.

The simulator
-------------

Two propagators are the entry point: ``MajoranaPropagator`` for native Majorana
(and fermionic) operators, and ``PauliPropagator`` for qubit (Pauli) operators,
which it takes directly and truncates by Pauli weight. Construct one from an
operator, a reference state, and a cutoff, then
:py:meth:`~monoprop.MajoranaPropagator.propagate` (or
:py:meth:`~monoprop.MajoranaPropagator.build_graph` to store a reusable
graph) and read off expectation values and gradients. The graph owns the gate
information, so evaluation takes only the parameter values. See
:doc:`/features/initialisation`, :doc:`/features/cutoff`, and
:doc:`/features/evaluation`.

.. automodule:: monoprop.majorana_propagator
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: monoprop.pauli_propagator
   :members:
   :undoc-members:
   :show-inheritance:

Authoring gates
~~~~~~~~~~~~~~~

A :class:`~monoprop.Gate` (aliased :class:`~monoprop.MajoranaGate`) is a pure generator: it
wraps the :class:`~monoprop.majorana_data.MajoranaOperator` it generates and carries no
parameter index (:class:`~monoprop.PauliGate` is the qubit analogue). A circuit bundles a sequence of these
gates with the angle *values* that drive them, the ``parameter_mapping`` wiring each gate to
an angle (default: one distinct angle per gate; reusing an index across gates ties their
angles), and the reference ``initial_state``. Circuits form a typed family —
:class:`~monoprop.MajoranaCircuit`, :class:`~monoprop.PauliCircuit`, and
:class:`~monoprop.fermi_data.FermiCircuit` — each fed to the propagator directly (there is
no separate conversion step), and two circuits of the same family compose temporally with
``+``. The propagator consumes a circuit; evaluation accepts either a plain parameter vector
or the circuit itself.

.. automodule:: monoprop.circuit
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
``FermiGate`` and ``FermiCircuit`` describe the evolution and the reference
state.

.. automodule:: monoprop.fermi_data
   :members:
   :undoc-members:
   :show-inheritance:

Qubit (Pauli) operators
~~~~~~~~~~~~~~~~~~~~~~~~~

The natural starting point when you already hold a qubit Hamiltonian:
real-coefficient sums of Pauli strings (``PauliOperator``), with ``PauliGate``
and ``PauliCircuit`` describing the circuit. Feed these to ``PauliPropagator``,
which takes them directly and truncates by qubit Pauli weight (see
:doc:`/features/cutoff`).

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

The canonical Majorana operator and dense circuit the builders above convert into, and
the protocols a custom operator or circuit type must satisfy.

.. automodule:: monoprop.majorana_data
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
