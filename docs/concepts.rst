========
Concepts
========

This section explains the ideas behind monoprop at a high level: the notation it
uses, how problems are expressed as input, and how the propagation algorithm
works. It is conceptual — for concrete constructors and arguments, see
:doc:`/python-api`.

monoprop is a classical simulator built on **operator propagation**. Rather than
storing the full, exponentially large quantum state, it expands an operator as a
real-coefficient sum of Majorana monomials and pushes that sum through a circuit
one gate at a time, truncating terms that contribute little. Expectation values
and gradients are then read off the propagated operator. The method is described
in full in :cite:`Miller2025-aj`.

The pages below build on one another:

- :doc:`/concepts/notation` — the Majorana basis, monomials and their Hermitian
  normalisation, the commutator structure, and the paired monomials that build up
  a reference state. This is the vocabulary the rest of the section uses.
- :doc:`/concepts/interface` — how a problem is expressed: the fermionic, qubit,
  and Majorana operator formats, the reference state, and how a circuit is
  described for simulation.
- :doc:`/concepts/algorithm` — how propagation works: applying a gate, combining
  monomials, controlling term growth through truncation, and evaluating
  expectation values and gradients.
- :doc:`/concepts/simulation_modes` — the Heisenberg and Schrödinger pictures:
  whether the observable or the state is propagated, and how to choose between
  them.

.. toctree::
   :hidden:
   :maxdepth: 1

   concepts/notation
   concepts/interface
   concepts/algorithm
   concepts/simulation_modes
