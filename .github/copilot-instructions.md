# Copilot Instructions for monoprop

monoprop is a high-performance C++/Python hybrid library implementing Majorana and Pauli propagation. The project combines modern C++23 with Python bindings via nanobind.

## Architecture Overview

- **Core C++ Engine**: High-performance simulation logic in `src/` and `include/monoprop/`
- **Python Interface**: User-facing API in `src/monoprop/` with C++ bindings in `src/bindings/`
- **Template-Based Design**: Heavily templated C++ code with compile-time mode limits (`monoprop_MAX_NUM_MODES`)
- **Generated Code**: Python dispatch and C++ bindings auto-generated via `tools/generate-*.py`

Key files:
- `src/monoprop/monomial_propagator.py`: Main Python API
- `include/monoprop/MonomialPropagator.h`: Core C++ simulator (1000+ lines)
- `src/bindings/bindings.cpp`: auto-generated Python bindings, using the nanobind library.


### Environment Management (uv-based)
```bash
uv sync --all-groups --all-extras -v  # Build & install
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

- `tests/cases.py`: Parametrized test cases using `pytest-cases`
- Tests validate against exact solutions for small systems
- Heavy use of `@parametrize_with_cases` decorators

## Key Dependencies & Integration

- **nanobind**: Modern Python-C++ binding (prefer over pybind11)
- **oneTBB**: Parallel computation (required build dependency)
- **scikit-build-core**: Modern build system replacing setuptools
- **uv**: Package management
- **fmt**: C++ formatting library
- **quill**: High-performance C++ logging
- **Boost**: Used for various utilities (unordered_map, unit tests)
- **msgpack**: Data serialization for input/output

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

### Building and Running C++ Examples
ALWAYS test your code with example.cpp by writing a test script there and then running it.
The project includes a standalone C++ example (`src/example.cpp`) that demonstrates the core simulator:

```bash
# Build the C++ example
cmake --build build/release-gcc --target example.x

# Run with default molecule (S0_14e14o_majoranic_c6)
./build/release-gcc/bin/example.x

# Run with specific molecule
./build/release-gcc/bin/example.x lih_fermionic_spin_exact

# Available test molecules in tests/data/:
# - lih_fermionic_spin_exact (12 modes)
# - random_exact (8 modes)
# - rx_rz_ry_rz_exact (1 mode)
# - S0_8e8o_majoranic_c6 (16 modes)
# - S0_14e14o_majoranic_c6 (28 modes)
```

The example loads MsgPack data, runs Majorana evolution simulations, and reports timing/energy results.

### Performance Considerations
- Templates are instantiated up to `monoprop_MAX_NUM_MODES` - higher values = longer compile times
- C++ code uses `Bitset<2*NumModes>` for efficient bit operations
- Python overhead minimized through compile-time dispatch

This is a sophisticated scientific computing project requiring careful attention to template instantiation, build system configuration, and the C++/Python boundary.
