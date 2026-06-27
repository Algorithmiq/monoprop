Data Structures
===============

.. admonition:: Developer reference

   This page documents internal implementation details for contributors modifying
   performance-sensitive runtime code.

Bitset<N>
---------

``src/monoprop/Bitset.h``

A fixed-size bitset backed by contiguous ``uint64_t`` words. It is trivially
copyable (memcpy-safe) and exposes its raw word pointer for zero-copy MPI
transmission.

.. code-block:: cpp

   Bitset<8> bs;
   bs.set(3);          // set bit 3
   bs.test(3);         // true
   bs.count();         // popcount over all words
   bs.find_first();    // index of lowest set bit
   bs.find_next(3);    // next set bit after position 3
   bs.data();          // const uint64_t* for MPI send/recv

Hashing uses ``SplitmixHash``, which mixes each word independently for good
avalanche behaviour.

MajoranaSet<NumModes>
----------------------

.. code-block:: cpp

   template <size_t NumModes>
   using MajoranaSet = Bitset<2 * NumModes>;

Encodes the presence of each Majorana operator ``γ_j`` as a bit. An N-mode system
has ``2N`` Majorana modes (2 per fermionic mode: ``γ_{2k}`` and ``γ_{2k+1}``). A
term like ``γ_0 γ_1 γ_3`` is represented as the bitset ``{0,1,3}``.

MajoranaVector / MajoranaOperator
-----------------------------------

.. code-block:: cpp

   template <size_t NumModes>
   using MajoranaVector = std::vector<MajoranaSet<NumModes>>;

   template <size_t NumModes>
   using MajoranaOperator = boost::unordered_flat_map<
       MajoranaSet<NumModes>, double, MPHash<NumModes>>;

``MajoranaVector`` is an ordered list of Majorana terms (used for operator and state
storage). ``MajoranaOperator`` maps each bitset term to its real coefficient.

ShardedIndexMap<NumModes>
-------------------------

``src/monoprop/TypeAliases.h``

A hash map from ``MajoranaSet<NumModes>`` to ``size_t`` (coefficient array index).
Internally it holds ``P = bit_ceil(num_shards)`` independent
``boost::unordered_flat_map`` shards, routing each key by ``hash(key) & (P-1)``.
This enables:

- **Lock-free reads**: ``find()`` routes to a single shard; no locks are taken.
- **Cache-friendly lookup**: each shard is a dense flat map with a single
  contiguous allocation.

.. code-block:: cpp

   ShardedIndexMap<4> idx(threading::effective_parallelism());
   idx.reserve(1000);
   idx.emplace(bitset, 42);

   auto it = idx.find(bitset);
   if (it != idx.end_for(bitset)) {
       size_t i = it->second;
   }

Always use ``end_for(key)`` rather than ``end()`` unless you accept the documented
cross-map sentinel behaviour.

MPOperator<NumModes>
--------------------

``src/monoprop/TypeAliases.h``

The per-rank operator bundle. It owns:

.. list-table::
   :header-rows: 1

   * - Field
     - Type
     - Purpose
   * - ``op``
     - ``MajoranaVector<NumModes>``
     - Ordered list of Majorana bitset terms
   * - ``ham_coeffs``
     - ``VecD``
     - Operator coefficients, aligned with ``op``
   * - ``state_coeffs``
     - ``VecD``
     - State (Slater det overlap) coefficients, aligned with ``op``
   * - ``indexing``
     - ``ShardedIndexMap<NumModes>``
     - bitset → position in ``op``
   * - ``init_ham_map_``
     - ``MajoranaOperator<NumModes>``
     - Staging map for lazy coefficient initialisation
   * - ``slater_determinant_``
     - ``VecZ``
     - Reference Slater determinant occupied modes

``get_operator()`` and ``get_state()`` are lazy: they populate the coefficient vectors
from ``init_op_map_`` and the Slater determinant respectively on first call, then
become O(1) returns.

Type aliases
------------

.. list-table::
   :header-rows: 1

   * - Alias
     - Underlying type
     - Use
   * - ``VecD``
     - ``std::vector<double>``
     - real coefficient vectors
   * - ``VecCD``
     - ``std::vector<std::complex<double>>``
     - complex coefficient vectors
   * - ``VecZ``
     - ``std::vector<size_t>``
     - index lists
   * - ``VecI``
     - ``std::vector<int>``
     - phase sign vectors
   * - ``FermiOperatorMap``
     - ``std::map<VecZ, std::complex<double>>``
     - public Operator input format
   * - ``CutoffFn<N>``
     - ``std::function<bool(const MajoranaSet<N>&)>``
     - cutoff predicate
