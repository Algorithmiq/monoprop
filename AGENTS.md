# Agent Instructions for monoprop

monoprop is a high-performance C++/Python hybrid library implementing Majorana and Pauli propagation. The project combines modern C++23 with Python bindings via nanobind. It is based on the paper arXiv:2503.18939


## Repository rules
- **C++23**: All C++ code must use C++23 features and idioms.
- **Auto style**: Use `clang-format` with the provided `.clang-format` file
- Python code must pass lints as defined in pre-commit hooks (including `aislop` for AI-generated code patterns).
- We use conventional commits with syntax: `<type>(<optional scope>): <gitmoji> <description>`. See `CONTRIBUTING.md` for details.
See `CONTRIBUTING.md` for details. Breaking changes should have `!` in the commit message. Make sure PR titles are adhering to the same format.
- Python docstrings use Google style. C++ docstrings use Doxygen style.

## Architecture Overview

- **Core C++ Engine**: High-performance simulation logic in `src/` and `include/monoprop/`
- **Python Interface**: User-facing API in `src/monoprop/` with C++ bindings in `src/bindings/`
- **Template-Based Design**: Heavily templated C++ code with compile-time mode limits (`monoprop_MAX_NUM_MODES`)
- **Generated Code**: Python dispatch and C++ bindings auto-generated via `tools/generate-*.py`

Key files:
- `src/monoprop/monomial_propagator.py`: Main Python API
- `include/monoprop/MonomialPropagator.h`: Core C++ simulator (1000+ lines)
- `src/bindings/bindings.cpp`: auto-generated Python bindings, using the nanobind library.


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

Python automatically dispatches to appropriate C++ template based on `num_modes`:
```python
# This routes to MonomialPropagator<4> in C++
mp = MonomialPropagator(operator, num_modes=4, ...)
```

### Testing Structure

- `tests/cases.py`: Parametrized test cases using `pytest-cases`; `load_problem()` loads a `tests/data/*.msgpack` fixture directly into the public API (`MonomialCircuit` + `MonomialOperator`). C++ tests use the equivalent `test_utils::load_case()` in `tests/cpp/TestData.h`
- Fixture msgpack schema is documented in `tests/data/README.md`
- Tests validate against exact solutions for small systems
- Heavy use of `@parametrize_with_cases` decorators

## Key Dependencies & Integration

- **nanobind**: Modern Python-C++ binding (prefer over pybind11)
- **oneTBB**: Parallel computation (required build dependency)
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
7. Add Python bindings in `src/bindings/binder.h`
8. Regenerate bindings with `tools/generate-binders.py`
9. Test with both C++ and Python tests

### Debugging Build Issues
- Check `build/*/compile_commands.json` for compilation flags
- Use `rm -rf build` to clear environment-specific builds
- Verify `monoprop_MAX_NUM_MODES` matches your use case (default: 250)

This is a sophisticated scientific computing project requiring careful attention to template instantiation, build system configuration, and the C++/Python boundary.
