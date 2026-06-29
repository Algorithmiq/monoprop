.. _propagation_algorithm:

Propagation Algorithm
=====================

This page explains how operator propagation works at a high level: how an
operator is pushed through a circuit, how a single gate updates each
monomial, how the growth of terms is controlled by truncation, and how
expectation values and gradients are read off the result. It builds on the
definitions in :doc:`/concepts/notation`, and the method is described in full in
:cite:`Miller2025-aj`.

Overview
--------

Operator propagation avoids ever storing the full, exponentially large quantum
state. Instead, an operator, either the observable or the state, is expanded as
a real-coefficient sum of Majorana monomials,

.. math::

   H = \sum_\nu c_\nu\, M_\nu, \qquad c_\nu \in \mathbb{R},

and that sum is pushed through the circuit one gate at a time. Given an observable
:math:`H`, a reference state :math:`\varrho`, and a parameterised circuit
:math:`C_L(\theta)`, the goal is to compute the expectation value

.. math::

   f_L(\theta) = \langle H \rangle_\theta
   = \mathrm{Tr}\!\left[\varrho\, C_L(\theta)^\dagger H\, C_L(\theta)\right].

We usually work in the **Heisenberg picture**, propagating the observable *backwards*
through the circuit while keeping the reference state fixed. One can equally
evolve the state forward in the Schrödinger picture.


Circuit and gates
-----------------

The circuit is a product of :math:`L` gates, each a Majorana rotation generated
by a Hermitian monomial :math:`M_{\gamma_j}`:

.. math::

   C_L(\theta) = \prod_{j=1}^{L} U_j(\theta)
   = \prod_{j=1}^{L} e^{-i\theta_j M_{\gamma_j}/2},

with real angles :math:`\theta_j`. Back-propagation applies the
gates in reverse order, updating the observable as

.. math::

   H_{j+1} = U_{L-j}^\dagger\, H_j\, U_{L-j},

starting from :math:`H_0 = H` and collecting all surviving monomials after each
step.

Applying a gate
---------------

Because the gate generator and every propagated term are even monomials, a single
gate :math:`U_j` sends each monomial :math:`M_\nu` into one of three cases,
governed entirely by whether it commutes with the generator:

.. math::

   M_\nu \;\xrightarrow{\;U_j\;}\;
   \begin{cases}
     M_\nu
       & \text{if } [M_\nu, M_{\gamma_j}] = 0, \\[4pt]
     \cos(\theta_j)\, M_\nu + i\sin(\theta_j)\, M_{\gamma_j} M_\nu
       & \text{if } \{M_\nu, M_{\gamma_j}\} = 0 \text{ and kept}, \\[4pt]
     \cos(\theta_j)\, M_\nu
       & \text{if } \{M_\nu, M_{\gamma_j}\} = 0 \text{ and truncated}.
   \end{cases}

If :math:`M_\nu` **commutes** with the generator it passes through unchanged. If
it **anticommutes** it *branches* into the original monomial (the **cosine**
branch) and a single new monomial :math:`M_{\gamma_j} M_\nu` (the **sine**
branch); recall from :doc:`/concepts/notation` that the product of two
anticommuting monomials is again a single monomial. When it is time to truncate,
the sine branch may be discarded, in which case the truncation step discards it
and only the cosine branch survives.

Applied across a whole circuit, the surviving branches grow the observable as a
tree - each branching gate can up to double the number of active terms. That
exponential growth is exactly what truncation keeps in check.

.. _truncation:

Truncation: controlling growth
------------------------------

Without truncation, the number of terms typically grows exponentially with the
number of branching gates, making exact simulation intractable. A **truncation
rule** is a map :math:`\mathcal{T}` on the operator space that discards terms
failing a retention criterion, applied after every gate:

.. math::

   \tilde{H}_{j+1} = \mathcal{T}\!\left(U_{L-j}^\dagger\, \tilde{H}_j\, U_{L-j}\right),
   \qquad
   \tilde{f}_L(\theta) = \mathrm{Tr}[\varrho\, \tilde{H}_L].

Two criteria are central.

Length truncation
~~~~~~~~~~~~~~~~~~

The primary control discards monomials whose length exceeds a cutoff
:math:`\omega`, with one important exception: a **fully paired** monomial - one
whose support consists entirely of complete pairs
:math:`m_{2j-1}m_{2j}` on a mode (see :doc:`/concepts/notation`) - is always kept, no
matter how long it is. Writing :math:`\mathcal{P}` for the set of paired
monomials,

.. math::

   \mathcal{T}_\omega\!\left(\sum_\nu c_\nu M_\nu\right)
   = \sum_{|\nu| \le \omega \,\text{ or }\, \nu \in \mathcal{P}} c_\nu M_\nu.

The length cutoff bounds the number of tracked terms to :math:`O(N^\omega)` for a
system of size :math:`N` at fixed :math:`\omega` — there are at most
:math:`\binom{2N}{\omega}` monomials of length :math:`\le \omega`. It is well
motivated in many
scenarios :cite:`Miller2025-aj`: short monomials tend to dominate expectation
values, and in the Majorana setting length truncation comes with provable
guarantees on the approximation error and cost for typical unstructured circuits.
The same idea applies in the Pauli picture (see :doc:`/concepts/notation`).

Keeping the paired monomials is what makes this safe in the Heisenberg picture.
As shown in *Reading off the expectation value* below, only paired monomials
survive the trace against a computational-basis state or Slater determinant, while every
unpaired monomial contributes nothing.

Coefficient truncation
~~~~~~~~~~~~~~~~~~~~~~~~

A complementary rule discards monomials whose coefficients fall below a
threshold. This is effective whenever the coefficients decay rapidly, suppressing
numerically negligible terms that would otherwise accumulate over a long sequence
of gates. Moreover, one can accept coefficients above a threshold that are above the
operator length cutoff, which is useful in certain scenarios.

Reading off the expectation value
----------------------------------

Once every gate has been applied, the expectation value is obtained by evaluating
the evolved observable against the reference state. Currently, the simulator only accepts
reference states in the form of a single slater determinant (or a single computational basis state).
The key simplification comes from the Majorana basis: distinct Hermitian monomials are orthogonal under the
trace, :math:`\mathrm{Tr}[M_\nu M_\mu] \propto \delta_{\nu,\mu}`. So for an
evolved observable :math:`\tilde{H}_L = \sum_\nu c_\nu M_\nu` and a reference
state :math:`\varrho = \sum_\mu b_\mu M_\mu`, the trace collapses to a product
over shared terms:

.. math::

   \tilde{f}_L(\theta) = \mathrm{Tr}[\varrho\, \tilde{H}_L] = \sum_\nu c_\nu b_\nu.

Only monomials present in *both* the evolved observable and the reference state
contribute. Since the reference state is a sum of paired monomials (see
:doc:`/concepts/notation`), the algorithm never sums over the exponentially many
terms of the full state - it only needs the handful of paired monomials that
overlap with the surviving observable.

Surrogate graphs, gradients, and optimisation
---------------------------------------------

In variational workflows one must evaluate :math:`f_L(\theta)` and its gradient
:math:`\nabla f_L(\theta)` repeatedly, across many parameter updates, so cheap
re-evaluation is essential. To this end the evolution paths of the monomials in
:math:`H` through the circuit are recorded as a **surrogate graph**. Once built,
the graph is replayed to compute :math:`f_L(\theta)` and :math:`\nabla
f_L(\theta)` at new parameter values *without re-propagating* the monomials -
propagating and evaluating become separate steps, and only the (cheap) evaluation
is repeated inside the optimisation loop.
Note that storing the surrogate graph can be memory intensive for large circuits.

See also
--------

- :doc:`/concepts/notation` - the Majorana basis, monomials, commutation rule, and pairing structure this page relies on.
- :doc:`/concepts/interface` - how to express operators and circuits as input.
- :doc:`/python-api` - the full API reference for driving a simulation.
