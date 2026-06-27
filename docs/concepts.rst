========
Concepts
========

This section explains the model behind monoprop: how operators are represented,
how graph layers are stored, and how MPI and threading shape the runtime.

.. contents:: On this page
   :local:
   :depth: 2

Majorana representation
=======================

monoprop works entirely in the Majorana basis. Every fermionic operator — the
Hamiltonian, the quantum state, and every generator in the circuit — is stored
as a real-coefficient sparse sum of Majorana monomials:

.. math::

   H = \sum_\alpha h_\alpha \, M_\alpha, \qquad h_\alpha \in \mathbb{R}

A monomial :math:`M_\alpha` is represented internally as a
``Bitset<2 * NumModes>`` — one bit per Majorana mode. The Hermitian-basis
encoding ensures that all stored coefficients are real even though the
underlying Majorana strings carry imaginary prefactors. The
``encode_coeff`` / ``decode_coeff`` helpers manage that map.

For the relation to complex fermions and qubits, the Hermitian-basis phase
conventions, the bitset encoding, and the pairing distance from the reference
Slater determinant, see :doc:`/concepts/majorana_basis`.

For the exact phase conventions, including the ``i^{C(n,2)}`` prefactor and
the ``antihermitian_generator_correction`` used in gradient APIs, see
:doc:`/concepts/algebra`.

For the concrete container types (``MajoranaSet``, ``MajoranaOperator``,
``MPOperator``, ``ShardedIndexMap``), see :doc:`/internals/data_structures`.

Operator representations
========================

Input operators can be specified in fermionic, qubit (Pauli), or direct
Majorana form. The three high-level input types —
``FermiOperator``, ``PauliOperator``, and ``MajoranaOperator`` — all convert to
the ``dict[tuple[int, ...], complex]`` format accepted by the simulator
constructor via their ``get_monomial_operator()`` methods.

For circuit-level workflows, ``MonomialCircuit`` bundles generators, parameter
indices, and initial state into a single structure, and ``MPData`` provides
a serialisable container for the complete simulation state.

For the full description of each format, the conversion pipeline, and the
serialisation interface, see :doc:`/concepts/operator_representations`.

Branch simulation algorithm
===========================

The central operation is the application of one unitary gate
:math:`U(\theta) = e^{i\theta G}` to an operator :math:`O`:

.. math::

   U^\dagger O\, U = \cos(2\theta)\, O + i\sin(2\theta)\, [G, O]

Because :math:`G` and every term in :math:`O` are Majorana monomials, the
commutator either vanishes (when :math:`G` and the term commute) or produces a
new monomial (when they anticommute). Applying a sequence of gates therefore
grows the operator as a *tree of branches*: each gate can double the number of
active terms. This is the source of the name "branch simulator".

The branch count is controlled by the **truncation cutoff** (see
`Approximation and truncation`_ below). Gates are *compiled* into a replayable
graph once; subsequent evaluations *replay* the stored graph at concrete
parameter values without re-running Majorana algebra.

For a detailed treatment of the branch tree structure, cycles, half-cycles,
graph compilation, and the compile-once / replay-many architecture, see
:doc:`/concepts/branch_simulation`.

For the full compile-and-replay kernel code, see
:doc:`/internals/graph_evolution`.

Heisenberg vs Schrödinger picture
==================================

monoprop supports both pictures and selects between them at construction time:

**Heisenberg picture** (default)
   The Hamiltonian is evolved forward through the unitary sequence:
   :math:`H_L = U_L^\dagger H_{L-1} U_L`. The reference state is kept fixed.
   Gate replay walks graph layers from back to front. This is the standard
   mode for ADAPT-VQE inner loops.

**Schrödinger picture**
   The quantum state is evolved instead of the Hamiltonian:
   :math:`\rho_L = U_L \rho_{L-1} U_L^\dagger`. A separate ``schrodinger_cutoff``
   controls the state-side truncation independently of the Hamiltonian-side
   ``cutoff``. Replay walks layers from front to back.

The picture determines the sign passed to ``map_params`` during replay and
the order in which ``propagate`` stores layers in the graph.

For the three concrete execution modes and how they interact with live
coefficient truncation, see :doc:`/concepts/simulation_modes`.
For the low-level order bookkeeping, see :doc:`/internals/evolution_path`.

Approximation and truncation
=============================

monoprop provides several independent knobs for controlling the approximation
introduced during simulation. All of them can be updated at runtime.

Structural cutoff
-----------------

The primary truncation: any Majorana monomial produced during gate application
that fails the cutoff is discarded before being inserted into the operator.

Two cutoff *types* are available:

``LengthPairingDistance`` (default)
   Truncates by the sum of the Majorana string length and its pairing distance
   from the reference Slater determinant. This is the physically motivated
   choice: it keeps operators that remain close to the reference state in both
   length and occupation.

``Mode``
   Truncates by the span of the mode indices in the bitset. Useful when the
   orbital ordering encodes spatial locality.

The cutoff *value* (an integer) is set at construction and can be raised or
lowered at any time with ``update_cutoff()``. The type can be changed with
``update_cutoff_type()``. A higher cutoff keeps more branches and increases
accuracy at the cost of runtime and memory.

Coefficient tolerance filtering
--------------------------------

Two absolute tolerance thresholds filter terms by coefficient magnitude
independently of the structural cutoff:

``lower_atol``
   Discard any term whose coefficient falls below this value. Applied during
   sine-branch insertion. This is the primary tool for suppressing
   numerically negligible operators during a long ADAPT loop.

``upper_atol``
   Hard upper bound: terms above this threshold are *also* discarded. Useful
   for stability in certain regimes.

Both can be set at construction and updated via ``update_lower_atol()`` and
``update_upper_atol()``.

Exact paring (masked execution plan)
------------------------------------

When an expectation value functional is created with a non-zero ``pare_threshold``,
monoprop builds a ``MPExecutionPlan`` that masks out graph edges whose
contribution to the expectation value is provably below the threshold. This is a
post-hoc, read-only filter over the stored graph: the graph itself is never
modified. Exact paring can dramatically reduce replay cost when the operator
is sparse in the graph-edge sense.

For a side-by-side comparison of all four knobs and their accuracy/performance
tradeoffs, see :doc:`/concepts/truncation`. For the implementation of exact
paring, see :doc:`/internals/masked_execution_plan`.

Expectation value and gradient evaluation
=========================================

After calling ``propagate``, the stored graph can be replayed an
arbitrary number of times at different parameter values without rerunning
Majorana algebra. Two functional types are available:

``expectation_value_functional(param_inds, gen_coeffs)``
   Returns a callable that accepts a parameter vector and computes the
   expectation value :math:`\langle H \rangle`.

``expectation_value_and_gradient_functional(param_inds, gen_coeffs)``
   Returns a callable that computes both the expectation valueand the gradient
   :math:`\partial E / \partial \theta_i` for all parameters in a single
   backward pass over the graph.

Both accept an optional ``pare_threshold`` that activates exact paring for
that functional instance.

For the analytic gradient formula, the backward pass mechanics, and candidate
functionals, see :doc:`/concepts/evaluation`. For
the underlying replay kernels, see :doc:`/internals/graph_evolution`.

Distributed execution and threading
=====================================

Every ``MajoranaSet`` is permanently assigned to exactly one MPI rank by a
deterministic hash:

.. code-block:: text

   rank(M) = SplitmixHash(M) % num_ranks

No rank table is communicated; any rank can compute the owner of any term
locally. Gate compilation emits *half-cycles* for terms that land on remote
ranks, which are resolved by a two-phase ``alltoallv`` protocol
(``update_mp``). Replay uses cached per-layer exchange layouts to avoid
rebuilding communication buffers on every call.

Within a rank, shared-memory parallelism is provided by oneTBB. MPI
collectives are always issued from the main thread (``MPI_THREAD_FUNNELED``).

For the complete communication model — rank assignment, compilation, replay,
expectation value/gradient collectives, TBB patterns, and single-rank / no-MPI
configurations — see :doc:`/concepts/distributed_execution`.

For full low-level details of the ``update_mp`` protocol, collective wrappers,
and the ``alltoallv`` / ``allreduce`` call sites, see
:doc:`/internals/mpi_and_threading`.

.. toctree::
   :hidden:
   :maxdepth: 1

   concepts/majorana_basis
   concepts/operator_representations
   concepts/branch_simulation
   concepts/simulation_modes
   concepts/truncation
   concepts/evaluation
   concepts/distributed_execution
   concepts/algebra
   internals/runtime_architecture
   internals/graph_evolution
   internals/single_rank_evolution
   internals/data_structures
   internals/graph
   internals/evolution_path
   internals/mpi_and_threading
   internals/masked_execution_plan
