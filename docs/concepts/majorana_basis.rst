Majorana Basis
==============

This page explains how monoprop represents quantum operators in the Majorana
fermion basis, how Majorana operators relate to standard fermionic and qubit
descriptions, and why this basis is well suited for efficient simulation.

Majorana operators
------------------

Majorana operators :math:`\gamma_j` are Hermitian
(:math:`\gamma_j^\dagger = \gamma_j`) and satisfy the anticommutation algebra

.. math::

   \{\gamma_i,\,\gamma_j\} = 2\delta_{ij}.

For a system with :math:`N` fermionic modes one introduces :math:`2N` Majorana
operators :math:`\gamma_0, \gamma_1, \ldots, \gamma_{2N-1}`. Their relation to
complex fermionic creation and annihilation operators is

.. math::

   c_j   = \tfrac{1}{2}(\gamma_{2j} + i\gamma_{2j+1}),
   \qquad
   c_j^\dagger = \tfrac{1}{2}(\gamma_{2j} - i\gamma_{2j+1}).

Every fermionic operator on :math:`N` modes can be written as a polynomial in
the :math:`2N` Majorana operators. In monoprop, Hamiltonians, quantum states, and
circuit generators are all stored in this form.

Monomials and the Hermitian basis
---------------------------------

A Majorana *monomial* :math:`M_\alpha` is an ordered product of distinct
Majorana operators indexed by a subset :math:`\alpha \subseteq \{0,\ldots,2N-1\}`:

.. math::

   M_\alpha = \prod_{j \in \alpha} \gamma_j \qquad (j\ \text{in ascending order}).

The *length* of a monomial is :math:`|M_\alpha| = |\alpha|`. For length
:math:`n`, the Hermitian conjugate satisfies
:math:`M_\alpha^\dagger = (-1)^{n(n-1)/2} M_\alpha`, so the raw monomial is not
always Hermitian. The Hermitian basis element associated with :math:`M_\alpha`
carries the compensating phase:

.. math::

   \hat{M}_\alpha = i^{C(n,2)}\, M_\alpha,
   \qquad C(n,2) = \tfrac{n(n-1)}{2}.

.. list-table::
   :header-rows: 1

   * - Length :math:`n`
     - Phase :math:`i^{C(n,2)}`
     - Example
   * - 1
     - :math:`1`
     - :math:`\gamma_j`
   * - 2
     - :math:`i`
     - :math:`i\gamma_j\gamma_k`
   * - 4
     - :math:`-1`
     - :math:`-\gamma_i\gamma_j\gamma_k\gamma_l`

Every Hermitian operator :math:`O` is expanded with purely **real** coefficients
in this basis:

.. math::

   O = \sum_\alpha h_\alpha\,\hat{M}_\alpha, \qquad h_\alpha \in \mathbb{R}.

The Hermitian-basis encoding ensures all stored coefficients are real even
though the underlying Majorana strings carry imaginary prefactors. The full
phase conventions are defined in :doc:`/concepts/algebra`.

Commutator structure
--------------------

Two even-length Majorana monomials :math:`M_\alpha` and :math:`M_\beta`
commute or anticommute depending on the parity of their overlap:

.. math::

   [M_\alpha,\,M_\beta] = 0
   \;\iff\;
   |\mathrm{supp}(\alpha) \cap \mathrm{supp}(\beta)|\ \text{is even.}

When the overlap is odd the commutator is proportional to the unique new
monomial with support :math:`\alpha \oplus \beta` (symmetric difference):

.. math::

   [M_\alpha,\,M_\beta] \propto M_{\alpha \oplus \beta},
   \quad |M_{\alpha \oplus \beta}| = |M_\alpha| + |M_\beta| - 2|\alpha \cap \beta|.

This one-to-one mapping from an anticommuting pair to a single new monomial is
the foundation of the branch simulation algorithm described in
:doc:`/concepts/branch_simulation`.

Slater determinant and pairing distance
----------------------------------------

The Majorana expansion of a Fock state :math:`|\Phi\rangle` contains only
*paired* monomials — products of pairs :math:`i\gamma_{2j}\gamma_{2j+1}` for
each occupied orbital :math:`j`. For a general monomial :math:`M_\alpha`, the
*pairing distance* from :math:`|\Phi\rangle` counts how many Majorana indices
in :math:`\alpha` fall outside the paired structure of the reference state.

The ``LengthPairingDistance`` cutoff (see :doc:`/concepts/truncation`) discards
monomials whose length plus pairing distance exceeds a threshold, keeping the
operator expansion close to the reference in both degree and occupation.

Jordan-Wigner basis change
---------------------------

When input operators are given in qubit (Pauli) form, monoprop converts them via
the Jordan-Wigner transformation. The helper
``jordan_wigner_basis_change(n_qubits)`` returns a list of :math:`2N` Majorana
index tuples encoding the Jordan-Wigner mode ordering for :math:`N` qubits:

.. doctest::

   >>> from monoprop import jordan_wigner_basis_change
   >>> basis = jordan_wigner_basis_change(n_qubits=4)   # list of 8 index lists
   >>> len(basis)
   8

Passing this list as ``basis_change`` to ``MonomialPropagator`` activates
the Jordan-Wigner ordering for all subsequent algebra. The ``PauliOperator``
and ``PauliEvGate`` types in ``monoprop.pauli_data`` handle the mapping
automatically.

Why the Majorana basis?
-----------------------

**Real arithmetic throughout.**
All stored coefficients are real. Expectation value evaluation reduces to a real dot
product; gradient accumulation involves no complex intermediate values.

**Sparse, deterministic gate updates.**
Each anticommuting pair maps to exactly one new monomial, so a gate application
produces at most one new term per existing term — no search, no ambiguity.

**Physically motivated truncation.**
Pairing distance from the reference Slater determinant provides a
physically interpretable measure of correlation, enabling accurate results at
moderate cutoff values for near-equilibrium quantum chemistry systems.

**Uniform Hamiltonian/state treatment.**
Both the Hamiltonian and the quantum state are stored as Majorana operators with
real coefficients, so the same graph-based evolution machinery serves both the
Heisenberg and Schrödinger pictures without algorithmic changes.
