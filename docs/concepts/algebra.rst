Algebra
=======

This document collects the Majorana algebra conventions used across monoprop.
It covers representation conventions, phase definitions, and coefficient
encoding rules referenced by commutator and evolution routines.

Representation Convention
--------------------------

Hamiltonians and states are stored as real coefficient vectors over Hermitian
Majorana basis elements.

For a Majorana string ``M`` of length ``n`` (popcount ``n``), the internal basis uses

.. code-block::

   H = sum_M h_M * (i^{C(n,2)} * M)

where ``C(n,2) = n(n-1)/2`` and ``hermitian_coefficient(M) = i^{C(n,2)}``.

This is why a length-2 string carries phase ``i^1 = i``, and a length-4 string
carries phase ``i^6 = -1``.

``encode_coeff(complex_coeff, M)`` computes
``complex_coeff / hermitian_coefficient(M)`` and requires the result to be real.
``decode_coeff(real_coeff, M)`` applies the inverse map.

Generator Convention
---------------------

The simulator applies unitaries as

.. code-block::

   U(theta) = exp(i theta G) = cos(theta) I + i sin(theta) G

so ``G`` is Hermitian in this convention.

For antihermitian notation ``A = iG``, the helper
``antihermitian_generator_correction(indices) = i^{C(n,2)+1}`` gives the prefactor
that makes ``c * M`` antihermitian.

.. code-block:: cpp

   VecD gen_coeffs(indices.size(), 1.0);

when passed back into evolution or candidate expectation value helpers.

These are two views of the same generator: ``gen_coeff = +1`` for Hermitian gate
replay, and ``antihermitian_generator_correction(indices)`` for user-pool APIs.

Phase Components
-----------------

For commutator and evolution kernels, the multiplicative sign is built from two
phase factors.

Interleave Phase
~~~~~~~~~~~~~~~~

.. code-block:: cpp

   interleave_phase<N>(maj, gen)

This is the ordering sign from bringing the product ``M * G`` back to canonical
Majorana order.

Equivalent parity form:

.. code-block::

   S = sum_b gen[b] * (sum_{t < b} maj[t]) (mod 2)
   interleave_phase = (-1)^S

The implementation computes ``S`` word-by-word using:

1. ``prefix_xor_64(maj_word)`` for intra-word prefix parity,
2. ``carry`` for prefix parity from previous words,
3. ``popcount(running_parity & gen_word)`` for odd-crossing bit positions.

This is equivalent to counting transpositions modulo 2.

Hermitian Phase
~~~~~~~~~~~~~~~

.. code-block:: cpp

   hermitian_phase(maj_count, gen_count, overlap)

For ``M' = M XOR G``, with ``|M'| = |M| + |G| - 2*overlap``, the Hermitian phase is
the basis-normalization sign

.. code-block::

   Phi_herm = Re[i^{C(|M|,2) + C(|G|,2) - C(|M'|,2) + 3}]

where ``+3`` accounts for one ``1/i`` from commutator normalization.

Combined Multiplicative Phase
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   get_multiplicative_phase<N>(maj, gen, maj_count, gen_count, overlap)
       = interleave_phase(maj, gen) * hermitian_phase(maj_count, gen_count, overlap)

This is the shared implementation used by both evolution and commutator paths.
