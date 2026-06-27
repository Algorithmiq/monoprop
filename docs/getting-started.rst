===============
Getting Started
===============

This page covers the installation and a minimal introduction to monoprop.
For more detailed usage instructions, see the :doc:`python-api`.

Installation
============

Install monoprop from PyPI using ``pip``:

.. code-block:: bash

   pip install monoprop

If you use `uv <https://docs.astral.sh/uv/>`_ as package manager:

.. code-block:: bash

   uv add monoprop

Introduction
============

Majorana Propagation is a classical simulation method for fermionic quantum
systems. It works by expressing quantum states and operators in terms of
Majorana operators and propagating them through sequences of fermionic gates.
Majorana operators have a particularly clean algebraic structure: in particular,
they are Hermitian and they satisfy simple anticommutation relations, which
makes them a natural basis for tracking how fermionic states evolve under a
circuit or Hamiltonian flow.

By representing the simulation in this Majorana basis, the method can
efficiently track the propagation of operators and evaluate expectation values
at a given set of variational parameters. This makes Majorana Propagation
well-suited for variational quantum algorithms as well as classical simulation
of fermionic Hamiltonians in quantum chemistry and condensed matter physics.

Minimal Example
================

The following example evolves a Majorana operator :math:`\gamma_0\gamma_1\gamma_2\gamma_4`
through an empty circuit and retrieves the resulting operator dictionary:

.. doctest::

   >>> from monoprop import MonomialPropagator
   >>> from monoprop.fermi_data import FermiCircuit, MajoranaOperator
   >>> initial_op = MajoranaOperator([(0, 1, 2, 4)], [1.0], 8)
   >>> circuit = FermiCircuit(initial_state=[], gates=[])
   >>> mbs = MonomialPropagator(initial_op, circuit, cutoff=16)
   >>> result = mbs.evolved_operator_dict()
   >>> result == {(0, 1, 2, 4): 1}
   True
