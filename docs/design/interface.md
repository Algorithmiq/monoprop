# monoprop authoring interface — design

Status: implemented on `refact/majorana-qubit-propagator`. Python-only; the C++ engine
(consumed through the flat-array `build_graph`) is untouched.

This document explains *why* the authoring interface is shaped the way it is. For the
task-oriented walkthrough and runnable examples, see the user-facing
[Interface page](../concepts/interface.rst); for exact signatures see the
[Python API reference](../python-api.rst).

## Motivation

The previous interface had three problems that compounded every time a circuit was authored:

- **Too many gate wrappers.** `Gate`, its alias `MajoranaGate`, `PauliGate`, and `FermiGate`
  — four types for one concept ("a generator that gets exponentiated").
- **Two ways to build every operator.** A positional-list constructor
  (`MajoranaOperator(majoranas, coeffs, n)`) *and* a `from_dict` classmethod, with no single
  obvious path.
- **Width in two places.** `num_qubits` lived on both the observable `PauliOperator` *and* the
  `PauliCircuit`, which could silently disagree with the Jordan-Wigner basis width.

## The model: four layers

The interface is now a strict four-layer pipeline, each layer built from the one below:

```
term  →  operator (Σ term·coeff)  →  Exp gate (exponentiate a generator)  →  circuit  →  propagator
```

| Layer | Majorana family | Qubit family | Fermionic family |
|-------|-----------------|--------------|------------------|
| term | `Majorana(*indices)` | `Pauli(string, qubits)` | *(fermionic string tuples)* |
| operator | `MajoranaOperator({term: c}, num_modes)` | `PauliOperator({term: c}, num_qubits)` | `FermiOperator(...)` |
| gate | `Exp(gen, param)` (single type — the generator carries the family) | | |
| circuit | `Circuit` (single type — the gates carry the family) | | |
| propagator | `MajoranaPropagator` | `PauliPropagator` | `MajoranaPropagator` |

There is a **single** `Circuit` type across all families. The gate objects carry the family,
so `Circuit` validates its gates are one consistent family (qubit gates cannot be mixed with
Majorana/fermionic gates) and exposes `Circuit.family` (`"pauli"` / `"majorana"` / `"empty"`);
the propagators dispatch on that. The old per-family circuit names
(`MajoranaCircuit` / `PauliCircuit` / `FermiCircuit`) have been removed — there is only
`Circuit`.

### Layer 1 — terms

`Majorana` and `Pauli` are frozen, hashable dataclasses. They are simultaneously the **keys**
of an operator dict and the **generators** accepted directly by an `Exp` gate, so "an operator
contains terms" is literal.

- `Majorana(*indices)` sorts its indices and rejects negatives / repeats (a repeated index,
  `γ_i² = 1`, is almost always a mistake).
- `Pauli(string, qubits=range(len(string)))` is **local**: it names only the qubits it acts
  on. It canonicalizes by dropping identity letters and co-sorting `(qubit, letter)` pairs, so
  `Pauli("IZ", (0,1))`, `Pauli("Z", (1,))` and `Pauli("Z", 1)` are one key, and
  `Pauli("XY", (1,0)) == Pauli("YX", (0,1))`.

### Layer 2 — operators, dict-constructed

Every operator is built from one `{term: coefficient}` mapping. Raw keys are still accepted for
convenience (`{(0,1): 1.0}` for Majorana, `{"ZZ": 1.0}` for a full-width Pauli), so the common
cases stay terse.

`from_dict` and the positional-list constructor are gone. The dict *is* the constructor.
Internally, `MajoranaOperator._from_terms(majoranas, coeffs, ...)` (and its Pauli twin) keeps
the parallel-list + duplicate-summing path that the Jordan-Wigner and fermionic conversions
need — a dict literal cannot express the colliding monomials those conversions produce — but
that method is private.

### Layer 3 — exponential gates

A single `Exp` type wraps a generator (a bare term or an operator) and makes the
exponentiation `exp(-iθ/2·G)` explicit at the call site. Like `Circuit`, it abstracts over the
family: the **generator type** it is handed selects the family and normalization convention
(see the load-bearing rules below). This explicit `Exp` wrapper is the one deliberate piece of
redundancy we kept over a "circuit holds bare operators" design (see decisions below).

### Layer 4 — circuits

A single `Circuit`: an ordered sequence of gates plus angle `parameters` and an
`initial_state`. No `parameter_mapping` argument and no `num_qubits` (see decisions). The gate
gates carry the family, so one `Circuit` type serves all families; it rejects a mix of qubit
and Majorana/fermionic gates and normalizes any fermionic `Exp` to a structural Majorana `Exp`
at construction (keeping the originals on `Circuit.fermi_generators`). The propagators check
`Circuit.family` rather than the circuit's Python type.

## The parameter model

Each `Exp` gate is the **unit of parameterization**: one gate ⇒ exactly one angle θ, whatever
the term-count of its generator. A multi-term generator is a single exponential
`exp(-iθ/2·Σ_k g_k M_k)` — the `g_k` are fixed structural weights, θ is the lone variable. For
independent angles, use separate gates.

The angle index lives on the gate as `param`:

- No gate sets `param` ⇒ identity mapping (each gate its own angle, in order).
- Any gate sets `param` ⇒ all must, indices contiguous `0..n-1`, and gates sharing an index
  share an angle (weight tying).

`Circuit.resolved_mapping` derives the per-gate index vector from the gates; `n_parameters`
is the count of distinct indices. This replaces the old circuit-level `parameter_mapping`
list. (The **propagator**'s graph-level `parameter_mapping` property — a cheap post-build
re-labelling of the built graph — is a different layer and is unchanged.)

## Two load-bearing rules

1. **The generator type carries the normalization rule.** A single `Exp` dispatches on the
   generator it is handed: a `Majorana`/`MajoranaOperator` is a *native* generator whose
   coefficients are the *structural* `g` used directly (imaginary part rejected as
   non-Hermitian); a `Pauli`/`PauliOperator` is Jordan-Wigner mapped and
   antihermitian-normalized; a `FermiOperator` is a fermionic generator, held in raw Majorana
   form and antihermitian-normalized. So `Exp(MajoranaOperator({(0,1): 1j}))` is always
   structural and correctly raises (non-Hermitian structural coeff), while
   `Exp(FermiOperator(...))` normalizes. There is no raw-`MajoranaOperator`-as-fermionic path:
   a `MajoranaOperator` means "structural", full stop — fermionic generators go through
   `FermiOperator`. `Circuit` performs the fermionic normalization **at construction**,
   converting each fermionic `Exp` into a structural Majorana `Exp`, so the expander only ever
   sees Majorana and Pauli gates.

2. **Width lives on the observable, and flows to gate expansion via the propagator.**
   `PauliPropagator` reads `num_qubits` from the observable and stores it as `self._num_qubits`;
   `MajoranaPropagator.build_graph`/`propagate` pass that to `expand_monomials`, which places
   each local `Pauli` into the full-width Jordan-Wigner string. A `PauliCircuit` carries no
   width, so the gate placement and the cutoff basis can no longer disagree.

## Design decisions

- **A single `Circuit`, not a typed circuit per family.** The gate objects already carry the
  family, so one `Circuit` type is enough; it validates gate-family consistency and exposes
  `Circuit.family`, and the propagators check that instead of the circuit's Python type. This
  lets qubit and fermionic problems share one circuit object. The per-family circuit names
  (`MajoranaCircuit` / `PauliCircuit` / `FermiCircuit`) have been removed — there is only
  `Circuit`. The family guardrails are enforced in two places: `Circuit` rejects a mix of
  qubit and Majorana/fermionic gates, and each propagator rejects the wrong family.
- **A single explicit `Exp` wrapper, not bare operators in the circuit.** A circuit *could*
  hold bare operators (exponentiation implied by position, as in `qiskit.QuantumCircuit`). We
  kept the `Exp` wrapper so "this generator is exponentiated" is visible at the call site, but
  collapsed the three family-specific wrappers (`MajoranaExp`/`PauliExp`/`FermiExp`) into one
  `Exp` that infers its family from the generator type — mirroring the single `Circuit`.
- **`param` on the gate, not a circuit-level list.** Sharing an angle reads locally
  (`Exp(op, param=k)` next to the other gates using `k`) and removes an aligned-array
  argument from the circuit. This reverses the earlier "gates carry no parameter index"
  decision at the authoring layer only.
- **Single dict constructor.** Drops `from_dict` and positional lists from the public surface;
  `_from_terms` preserves the internal duplicate-summing path.
- **`num_qubits` only on the observable.** Removed from `PauliCircuit`. The qiskit round-trip,
  which genuinely needs a width to reconstruct, now takes it as an explicit argument:
  `to_qiskit_circuit(circuit, num_qubits)`.
- **`FermiOperator` unchanged.** The dict-constructor ask was scoped to `Majorana`/`Pauli`;
  `FermiOperator` keeps its list constructor and `from_dict` to limit blast radius.

## Migration reference

| Old | New |
|-----|-----|
| `MajoranaOperator([(0,1)], [c], n)` | `MajoranaOperator({(0,1): c}, n)` |
| `MajoranaOperator.from_dict(d, n)` | `MajoranaOperator(d, n)` |
| `PauliOperator(["ZZ"], [c], num_qubits=n)` | `PauliOperator({"ZZ": c}, num_qubits=n)` |
| `PauliOperator.from_dict(d, num_qubits=n)` | `PauliOperator(d, num_qubits=n)` |
| `Gate(g)` / `MajoranaGate(g)` / `MajoranaExp(g)` | `Exp(g)` |
| `PauliGate((0,), PauliOperator(["X"],[1.0],num_qubits=1))` / `PauliExp(Pauli("X", 0))` | `Exp(Pauli("X", 0))` |
| `FermiGate(generator=fo)` / `FermiExp(fo)` (a `FermiOperator`) | `Exp(fo)` |
| `FermiExp(MajoranaOperator({(0,1): 1j}))` (raw Majorana-as-fermionic) | `Exp(FermiOperator(...))` (removed — use a `FermiOperator`) |
| `MajoranaCircuit(...)` / `PauliCircuit(...)` / `FermiCircuit(...)` | `Circuit(...)` |
| `PauliCircuit(gates=..., num_qubits=n)` | `Circuit(gates=...)` |
| `MajoranaCircuit(gates=[a,b,c], parameter_mapping=[0,1,0])` | `Circuit([Exp(a,param=0), Exp(b,param=1), Exp(c,param=0)])` |
| `circuit.parameter_mapping` | `circuit.resolved_mapping` |
| `circuit.fermi_gates` | `circuit.fermi_generators` |
| `PauliString("XY")` | `Pauli("XY", (0,1))` |
| `pauli_op.strings` / `.coefficients` | `pauli_op.terms` (`dict[Pauli, complex]`) |
| `to_qiskit_circuit(circuit)` | `to_qiskit_circuit(circuit, num_qubits)` |
