Notation
========

This page collects the notation and conventions used throughout the concepts
section: the Majorana basis (the chosen basis with which the arithmetic driving the propagator is performed in),
how monomials are indexed and made Hermitian, the
expansion of operators and states, the commutation rule that drives propagation,
and the dual interpretation of the same data as Pauli strings. The
:doc:`/concepts/algorithm` page builds directly on these definitions.

Majorana operators
-------------------

Majorana operators :math:`m_j` are Hermitian (:math:`m_j^\dagger = m_j`) and
satisfy the anticommutation algebra

.. math::

   \{m_i,\,m_j\} = 2\delta_{ij}\,\mathbb{I}.

A system of :math:`N` fermionic modes is described by :math:`2N` Majorana
operators :math:`m_1, \ldots, m_{2N}`. They are related to the usual fermionic
creation and annihilation operators by

.. math::

   c_j = \tfrac{1}{2}(m_{2j-1} + i\,m_{2j}),
   \qquad
   c_j^\dagger = \tfrac{1}{2}(m_{2j-1} - i\,m_{2j}).

Every operator on :math:`N` modes can be written as a polynomial in the
:math:`2N` Majorana operators, so Hamiltonians, quantum states, and circuit
generators can all be expressed in the same language. Through a fermion-to-qubit mapping,
the same data can also be interpreted as Pauli strings.

Monomials and the Hermitian basis
----------------------------------

A Majorana **monomial** is indexed by a binary string
:math:`\nu \in \{0,1\}^{2N}` and defined with a phase prefactor that makes it
Hermitian:

.. math::

   M_\nu = i^{\binom{|\nu|}{2}}\, m_1^{\nu_1} m_2^{\nu_2} \cdots m_{2N}^{\nu_{2N}}
   \;\propto\; \prod_{i=1}^{N} m_{2i-1}^{\nu_{2i-1}}\, m_{2i}^{\nu_{2i}}.

The string :math:`\nu` records which Majorana operators are present (its set of
ones is the monomial's **support**), and the **length** :math:`|\nu|` is the
number of ones - i.e. how many Majorana operators the monomial contains.

The phase :math:`i^{\binom{|\nu|}{2}}` makes :math:`M_\nu`
Hermitian. Taking the dagger reverses the order of the :math:`n` Majorana
operators, and each of the :math:`\binom{n}{2}` transpositions needed to restore
their order flips a sign, so a bare monomial of length :math:`n` satisfies
:math:`(m_{i_1}\cdots m_{i_n})^\dagger = (-1)^{n(n-1)/2}\,m_{i_1}\cdots m_{i_n}`.
The compensating factor :math:`i^{\binom{|\nu|}{2}}` cancels this sign and
restores :math:`M_\nu^\dagger = M_\nu`:

.. list-table::
   :header-rows: 1

   * - Length :math:`|\nu|`
     - Phase :math:`i^{\binom{|\nu|}{2}}`
     - Example
   * - 1
     - :math:`1`
     - :math:`m_j`
   * - 2
     - :math:`i`
     - :math:`i\,m_j m_k`
   * - 4
     - :math:`-1`
     - :math:`-\,m_i m_j m_k m_l`

Because every :math:`M_\nu` is Hermitian, any Hermitian operator :math:`H`
expands with purely **real** coefficients,

.. math::

   H = \sum_\nu c_\nu\, M_\nu, \qquad c_\nu \in \mathbb{R}.

Commutation rule
----------------

For two monomials :math:`M_\nu` and :math:`M_\mu`, whether they commute or
anticommute is fixed by their lengths and the parity of their overlap
:math:`|\nu \cap \mu|`:

.. math::

   M_\nu M_\mu = (-1)^{\,|\nu|\,|\mu| \,+\, |\nu \cap \mu|}\, M_\mu M_\nu.

The three resulting cases are:

- two **even** monomials commute iff the overlap is even;
- two **odd** monomials commute iff the overlap is odd;
- an **even** and an **odd** monomial commute iff the overlap is even.

Throughout propagation the generators and the propagated terms are even
monomials, so the relevant case is the first: two even monomials **anticommute
if and only if their overlap is odd**. When that happens, their product is again
a single monomial, on the symmetric difference of the two supports:

.. math::

   M_\nu M_\mu \;\propto\; M_{\nu \oplus \mu},
   \qquad
   |\nu \oplus \mu| = |\nu| + |\mu| - 2\,|\nu \cap \mu|.

This one-to-one map, an anticommuting pair produces exactly one new monomial,
is what makes gate application deterministic and sparse, and is the algebraic
core of the propagation algorithm.

Reference states and paired monomials
--------------------------------------

A Fock state :math:`|n_1 \dots n_N\rangle` (in particular a Hartree-Fock
reference, or Slater determinant) has an especially simple Majorana expansion.
Introducing the **paired** operator for each orbital,

.. math::

   \overline{m}_j = -i\,m_{2j-1} m_{2j},

the corresponding density operator factorises one orbital at a time,

.. math::

   |n_1 \dots n_N\rangle\!\langle n_1 \dots n_N|
   = \frac{1}{2^N} \prod_{j=1}^{N}\bigl(1 + (-1)^{n_j}\,\overline{m}_j\bigr).

Expanding the product, this is a sum of **paired monomials** -- products of the
:math:`\overline{m}_j` -- with signs fixed by the occupation numbers.

The Pauli representation
------------------------

The engine always works in the Majorana basis: a qubit system enters as a
*Majorana representation* - the binary strings :math:`\nu` above - and every gate,
commutator, and product is evaluated there. Pauli strings never appear during
propagation.

To read out or truncate in the qubit picture, you supply a
fermion-to-qubit encoding, which reinterprets the *same* :math:`\nu` as a Pauli
string. Under the `Jordan-Wigner mapping <https://en.wikipedia.org/wiki/Jordan%E2%80%93Wigner_transformation>`_, orbital :math:`i` maps to qubit :math:`i` and its two
Majoranas become

.. math::

   m_{2i-1} \;\mapsto\; \Bigl(\textstyle\prod_{k<i} Z_k\Bigr) X_i,
   \qquad
   m_{2i} \;\mapsto\; \Bigl(\textstyle\prod_{k<i} Z_k\Bigr) Y_i.

Each Majorana terminates in an :math:`X` or :math:`Y` on its own qubit, preceded by
a **parity string** of :math:`Z`'s on every lower qubit. These tails make
Jordan-Wigner **non-local**: a single Majorana on mode :math:`i` becomes a Pauli of
weight :math:`i`.

The encoding is *linear* in :math:`\nu` - a change of basis on the binary data,
:math:`\nu \mapsto \nu'` over :math:`\mathbb{F}_2`. Propagation is untouched; only
the reading of :math:`\nu` changes. Other mappings, such as the ternary-tree
constructions, are linear in the same way, so the choice of encoding is just a
choice of how to read the data, and the :ref:`propagation algorithm <propagation_algorithm>` is
unchanged.

Support cutoff = Pauli weight
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The encoding is consulted only when truncation is applied (see :ref:`truncation <truncation>`).
The ``support`` cutoff keeps a monomial acting on at most :math:`\omega` distinct *qubits*, which the
engine measures on the transformed string :math:`\nu'`, not the raw :math:`\nu`:

.. math::

   |P_\nu| \;=\; \operatorname{support}(\nu') \;\le\; \omega .

The distinction matters because fermion-to-qubit mappings are non-local: the :math:`Z` parity
tails fall on the qubits below each mode, so the Pauli weight generally *exceeds*
the Majorana support of :math:`\nu` (the number of modes it touches).
