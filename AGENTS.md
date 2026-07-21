# Agent Instructions for monoprop

monoprop is a high-performance C++/Python hybrid library implementing Majorana and Pauli propagation. The project combines modern C++23 with Python bindings via nanobind. It is based on the paper arXiv:2503.18939


## Repository rules
- **C++23**: All C++ code must use C++23 features and idioms.
- **Auto style**: Use `clang-format` with the provided `.clang-format` file
- Python code must pass lints as defined in pre-commit hooks.
- We use conventional commits with syntax: `<type>(<optional scope>): <gitmoji> <description>`. See `CONTRIBUTING.md` for details. Breaking changes should have `!` in the commit message.
- If you make a commit, add a trailer: `Assisted-by: <harness>:<model>`, where `<harness>` is the current agent harness (like ClaudeCode), and `<model>` is the AI model (like claude-opus-4.8). You don't need to add a coauthored-by when you have this.
- PR titles must adhere to the same conventional commits format.
- Prefix PR descriptions and comments on PRs with the line ":robot: _AI text below_ :robot:" to indicate you are an agent speaking on a user's behalf.
- Python docstrings use Google style. C++ docstrings use Doxygen style.


## Architecture Overview

- **Core C++ Engine**: High-performance simulation logic in `src/` and `include/monoprop/`
- **Python Interface**: User-facing API in `src/monoprop/` with C++ bindings in `src/bindings/`
- **Template-Based Design**: Heavily templated C++ code with compile-time mode limits (`monoprop_MAX_NUM_MODES`)
- **Generated Code**: Python dispatch and C++ bindings auto-generated via `tools/generate-*.py`

Key files:
- `src/monoprop/monomial_propagator.py`: abstract base `MonomialPropagator`; the concrete
  user-facing front-ends are `src/monoprop/majorana_propagator.py` (`MajoranaPropagator`) and
  `src/monoprop/pauli_propagator.py` (`PauliPropagator`).
- `include/monoprop/MonomialPropagator.h`: the single templated C++ engine `MonomialPropagator<NumModes>`
  (the Majorana/Pauli choice is a runtime `Basis`, not a separate class).
- `src/monoprop/bindings/binder.h`: hand-written binding template; `tools/generate-*.py` generate the
  per-mode-width `bindings.cpp` and `_dispatch.py` from it (do not hand-edit the generated files).

### Core abstractions (the propagation backbone)

- **`Monomial<N>`** (`src/monoprop/core/Monomial.h`) = `Bitset<2*N>`: ONE basis operator, two bits per
  mode/qubit. Basis-agnostic — read as a Majorana product, or as a Pauli string (JW image).
  Collections: `MonomialList<N>` (no coeffs) and `MonomialMap<N>` (monomial → real coeff).
- **`Basis` / the `Algebra` policy** (`src/monoprop/algebra/`): the two algebras are sibling models
  (`MajoranaAlgebra`, `PauliAlgebra` in `algebra/Algebra.h`) over shared structural primitives
  (`algebra/AlgebraCommon.h`). The propagation backbone (the scan/fold in `detail/evolution/...`) is
  templated on the algebra policy and bound to a runtime `Basis` once, via `with_algebra`.


### Environment Management
We use `uv` for environment management.
We also use [`just`](https://github.com/casey/just) for task automation.

```bash
uv sync --all-extras -v  # Build & install
uv run pytest  # Run tests
just build-docs  # Build documentation
```


### Template Metaprogramming

C++ code uses extensive compile-time templates with `NumModes` parameter:
```cpp
template <size_t NumModes>
class MonomialPropagator { /* ... */ };
```

### Mode-Based Dispatching

Python automatically dispatches to the appropriate C++ template based on the operator's mode count.
`MonomialPropagator` is an abstract base; construct a concrete front-end (which reads the mode count
off the operator — there is no `num_modes` argument):
```python
# Routes to MonomialPropagator<4> in C++ (Basis::Majorana here; PauliPropagator uses Basis::Pauli)
mp = MajoranaPropagator(operator, initial_state, cutoff=4)
```

### Testing Structure

- `tests/cases.py`: Parametrized test cases using `pytest-cases`; `load_problem()` loads a `tests/data/*.msgpack` fixture directly into the public API (`MonomialCircuit` + `MonomialOperator`). C++ tests use the equivalent `test_utils::load_case()` in `tests/cpp/TestData.h`
- Fixture msgpack schema is documented in `tests/data/README.md`
- Tests validate against exact solutions for small systems
- Heavy use of `@parametrize_with_cases` decorators

## Key Dependencies & Integration

- **nanobind**: Modern Python-C++ binding (prefer over pybind11)
- **scikit-build-core**: Modern build system replacing setuptools
- **uv**: Package management
- **fmt**: C++ formatting library
- **Boost**: Used for various utilities (unordered_map, unit tests)
- **msgpack**: Serialization of the test-data fixtures only (`tests/data/*.msgpack`); consumed by the Python test loaders and the C++ test suite, not by the shipped library

## Common Tasks

### Adding New C++ Functionality
1. Add to appropriate header in `include/monoprop/`
2. Use C++23 syntax and idioms.
3. Use almost always auto style.
4. Use trailing return type syntax in function declarations.
5. Add Doxygen docstrings.
6. Implement in corresponding `.cpp` in `src/`
7. Add Python bindings in `src/monoprop/bindings/binder.h`
8. Regenerate bindings with `tools/generate-binders.py`
9. Test with both C++ and Python tests

### Debugging Build Issues
- Check `build/*/compile_commands.json` for compilation flags
- Use `rm -rf build` to clear environment-specific builds
- Verify `monoprop_MAX_NUM_MODES` matches your use case (default: 250)

This is a sophisticated scientific computing project requiring careful attention to template instantiation, build system configuration, and the C++/Python boundary.
