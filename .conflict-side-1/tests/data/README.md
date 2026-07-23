# Test data fixtures

The `*.msgpack` files in this directory are pre-computed reference cases for the test suite
(exact energies/gradients for small fermionic problems). They are loaded by:

- Python: `load_problem()` in `tests/cases.py`
- C++: `test_utils::load_case()` in `tests/cpp/TestData.h`

## msgpack schema

Each file is a flat msgpack map with exactly these keys:

| key               | type                          | meaning                                              |
| ----------------- | ----------------------------- | ---------------------------------------------------- |
| `majoranas`       | `list[list[int]]`             | Hermitian Majorana operators (sorted index tuples)   |
| `gen_coeffs`      | `list[float]`                 | Real coefficients (anti-Hermitian sign factors)      |
| `param_inds`      | `list[int]`                   | Majorana → parameter index map                       |
| `parameters`      | `list[float]`                 | Evolution parameters                                 |
| `hartree_fock`    | `list[int]`                   | Initial (Hartree-Fock) occupied modes                |
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
