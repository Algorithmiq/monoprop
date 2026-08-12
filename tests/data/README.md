# Test data fixtures

The `*.msgpack` files in this directory are pre-computed reference cases for the test suite
(exact energies/gradients for small fermionic problems). They are loaded by:

- Python: `load_problem()` in `tests/cases.py`
- C++: `test_utils::load_case()` in `cpp/tests/TestData.h`

## msgpack schema

Each file is a flat msgpack map with exactly these keys:

| key               | type                          | meaning                                              |
| ----------------- | ----------------------------- | ---------------------------------------------------- |
| `majoranas`       | `list[list[int]]`             | Hermitian Majorana operators (sorted index tuples)   |
| `gen_coeffs`      | `list[float]`                 | Real coefficients (anti-Hermitian sign factors)      |
| `param_inds`      | `list[int]`                   | Majorana → parameter index map                       |
| `parameters`      | `list[float]`                 | Evolution parameters                                 |
| `hartree_fock`    | `list[int]`                   | Initial state: occupied modes (legacy key name)      |
| `num_modes`       | `int`                         | Number of modes                                      |
| `actual_energy`   | `float`                       | Exact expectation value                              |
| `actual_gradient` | `list[float]`                 | Exact gradient (may be empty)                        |
| `hamiltonian`     | `{keys, real, imag}`          | Fermionic Hamiltonian (see below)                    |

The Hamiltonian is stored as three parallel arrays (msgpack has no native complex type):

```
"hamiltonian": {
  "keys": [[int, ...], ...],   # non-Hermitian Majorana products
  "real": [float, ...],        # real part of each coefficient
  "imag": [float, ...],        # imaginary part of each coefficient
}
```

i.e. `terms = {tuple(k): complex(r, i) for k, r, i in zip(keys, real, imag)}`.

## Wide cases are derived, not stored

No fixture here is wider than 28 modes, so every one of them stores its monomials in a single 32-mode
block. The wide-system regime — several words per monomial, and the storage width at which the
support-form row store is selected — is reached by *relabelling* one of these fixtures into a wider
system rather than by adding a file:

- Python: `ModeEmbedding` and `WIDE_EMBEDDING` in `tests/cases.py`, passed to `load_problem()`
- C++: `test_utils::ModeEmbedding` and `embed_case()` in `cpp/tests/TestData.h`

A monotone injection of modes is a canonical transformation: a sorted Majorana index tuple stays
sorted, so no anticommutation sign appears, and `actual_energy` / `actual_gradient` still describe the
embedded problem exactly. Prefer that to a new binary whose only difference from an existing one is a
permutation of its mode labels — it needs no second reference calculation, and the wide run then owes
the narrow run's evolved operator term for term, which is a sharper check than either value alone.

The on-disk key names are **frozen** — the fixtures are checked-in binaries and are not rewritten
when the code's notation changes. `hartree_fock` is the historical name for what the library now
calls the *initial state*; both loaders (`tests/cases.py`, `cpp/tests/TestData.cpp`) read that key
into a field named `initial_state`.
