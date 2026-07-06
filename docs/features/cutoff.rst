Truncation and cutoffs
======================

monoprop limits the growth of the propagated operator with two independent
mechanisms: a **structural cutoff** on the size of each monomial, and
**coefficient-tolerance** thresholds on the magnitude of each term. Both can be
changed at any point during a simulation.

The structural cutoff discards monomials that grow "too large". What counts as
large is set by the cutoff **type** — ``length`` or ``support`` — and by which
simulator you use: ``MajoranaPropagator`` measures a monomial as a Majorana
operator, whereas ``PauliPropagator`` measures it as a qubit (Pauli) operator.

Cutoff type and value
---------------------

Two structural cutoff strategies are available. Both share one rule: a monomial
that is **fully paired** — every Majorana operator it contains comes as a
complete pair :math:`m_{2j-1}m_{2j}` on some mode — is *always kept*,
regardless of its size. Fully paired monomials are exactly the terms that can
contribute to an expectation value against a computational-basis or Slater-determinant
reference, so discarding them would throw away signal (see
:doc:`/concepts/algorithm`). The strategies differ only in how they measure the
remaining, partially paired monomials:

``length`` (default)
   Keeps a monomial when its **length** — the number of operators it contains in
   the chosen basis — does not exceed the cutoff. Short monomials dominate typical
   expectation values, so bounding the length bounds the number of tracked terms
   while retaining the dominant contributions. This is the physically motivated
   choice for quantum chemistry Hamiltonians. For example, :math:`M_\nu = m_1 m_2
   m_5` contains three Majorana operators, so its length is 3 and a ``length``
   cutoff of 2 discards it.

``support``
   Keeps a monomial when the number of distinct **sites** it touches (its
   support) does not exceed the cutoff. Because one site can carry two operators,
   this is a coarser measure than length; it is useful when the ordering encodes
   spatial locality and you want to bound how far a term spreads rather than how
   many operators it carries. The same :math:`M_\nu = m_1 m_2 m_5` touches only
   mode 1 (carrying both :math:`m_1` and :math:`m_2`) and mode 3 (carrying
   :math:`m_5`), so its support is 2 and a ``support`` cutoff of 2 keeps it — even
   though its length is 3.

Both the cutoff value and the cutoff type can be changed at any point during a
simulation:

.. code-block:: python

   sim.cutoff = 8
   sim.cutoff_type = "length"

Choosing a simulator
--------------------

The cutoff counts operators in whichever representation the simulator works in, so
picking the right simulator is what fixes the meaning of ``length`` and ``support``.

**Majorana operators —** ``MajoranaPropagator``. Choose ``cutoff_type`` between
``length`` (counts Majorana operators :math:`m_j`) and ``support`` (counts the
distinct modes touched). This is the natural choice for fermionic problems.

**Qubit (Pauli) operators —** ``PauliPropagator``. For a qubit Hamiltonian you
usually want to bound the **Pauli weight** instead. ``PauliPropagator`` accepts
Pauli operators and gates directly and the ``cutoff`` parameter bounds the Pauli-weight.
Reach for it whenever your problem is naturally expressed in qubits.

Coefficient tolerance filtering
-------------------------------

Two absolute-tolerance thresholds prune terms by coefficient magnitude,
independently of the structural cutoff:

- ``lower_atol``: discard terms with ``|coeff| < lower_atol`` during gate
  application. Primary tool for suppressing negligible branches in long
  ADAPT loops.
- ``upper_atol``: accept terms with ``|coeff| > upper_atol`` regardless of the structural cutoff. Useful for
  keeping a few large terms that would otherwise be discarded by a tight cutoff.

.. code-block:: python

   sim.lower_atol = 1e-9
   sim.upper_atol = 1e-3
