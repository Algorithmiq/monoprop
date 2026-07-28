#!/usr/bin/env bash

set -euo pipefail

# Script to install dependencies for monoprop project

show_help() {
    cat << EOF
Usage: $0 [INSTALL_PREFIX] [OPTIONS]

Install C++ dependencies for monoprop project.

This script can install Boost Unordered, Boost Test, and msgpack-cxx.
Each component can be skipped with the corresponding option.
The default installation prefix is /usr/local.

Arguments:
    INSTALL_PREFIX      Directory to install dependencies (default: /usr/local)

Options:
    --skip-boost-unordered  Skip installing Boost unordered
    --skip-boost-test   Skip installing Boost Test library (only install unordered)
    --skip-msgpack      Skip installing msgpack-cxx library
    --help, -h          Show this help message

Examples:
    $0                                   # Install all deps to /usr/local
    $0 \$HOME/Software                   # Install all deps to \$HOME/Software
    $0 --skip-boost-test                 # Skip Boost Test, install rest to default location
    $0 /opt --skip-msgpack               # Install to /opt, skip msgpack
    $0 --skip-boost-test --skip-msgpack  # Minimal install
EOF
}

# Default values
DEFAULT_PREFIX="/usr/local"
INSTALL_PREFIX="$DEFAULT_PREFIX"
INSTALL_BOOST_UNORDERED=true
INSTALL_BOOST_TEST=true
INSTALL_MSGPACK=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-boost-unordered)
            INSTALL_BOOST_UNORDERED=false
            shift
            ;;
        --skip-boost-test)
            INSTALL_BOOST_TEST=false
            shift
            ;;
        --skip-msgpack)
            INSTALL_MSGPACK=false
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        --*)
            echo "Unknown option: $1" >&2
            echo "Use --help for usage information." >&2
            exit 1
            ;;
        *)
            # Assume it's the install prefix if it doesn't start with --
            if [[ -z "${INSTALL_PREFIX_SET:-}" ]]; then
                INSTALL_PREFIX="$1"
                INSTALL_PREFIX_SET=true
            else
                echo "Multiple install prefixes specified: $INSTALL_PREFIX and $1" >&2
                exit 1
            fi
            shift
            ;;
    esac
done

echo "Installing C++ dependencies to: $INSTALL_PREFIX"
echo "Boost unordered: $([ "$INSTALL_BOOST_UNORDERED" = true ] && echo "YES" || echo "SKIP")"
echo "Boost Test: $([ "$INSTALL_BOOST_TEST" = true ] && echo "YES" || echo "SKIP")"
echo "msgpack-cxx: $([ "$INSTALL_MSGPACK" = true ] && echo "YES" || echo "SKIP")"
echo

# Create install directory if it doesn't exist
mkdir -p "$INSTALL_PREFIX"

# Function to clean up build artifacts
cleanup_build() {
    local src_dir="$1"
    echo "Cleaning up $src_dir..."
    cd -
    rm -rf "$src_dir" build
}

install_boost() {
    if [ "$INSTALL_BOOST_UNORDERED" != true ] && [ "$INSTALL_BOOST_TEST" != true ]; then
        echo "Skipping Boost installation"
        return 0
    fi

    local boost_version="1.89.0"
    echo "Installing Boost $boost_version..."
    git clone https://github.com/boostorg/boost.git -b boost-$boost_version boost_src --depth 1
    cd boost_src
    git submodule update --depth 1 -q --init tools/boostdep

    if [ "$INSTALL_BOOST_UNORDERED" = true ]; then
        git submodule update --depth 1 -q --init libs/unordered
        python3 tools/boostdep/depinst/depinst.py -X test -g "--depth 1" unordered
    else
        echo "  - Skipping Boost unordered library"
    fi

    if [ "$INSTALL_BOOST_TEST" = true ]; then
        echo "  - Including Boost Test library"
        git submodule update --depth 1 -q --init libs/test
        python3 tools/boostdep/depinst/depinst.py -X test -g "--depth 1" test
    else
        echo "  - Skipping Boost Test library"
    fi

    cmake -S. -Bbuild -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"
    cmake --build build --target install --parallel
    cleanup_build boost_src
    if [[ "$INSTALL_PREFIX" == "$DEFAULT_PREFIX" ]]; then
        echo "Remember to export Boost_DIR=$INSTALL_PREFIX/lib/cmake/Boost-$boost_version"
    fi
}

install_msgpack() {
    if [ "$INSTALL_MSGPACK" != true ]; then
        echo "Skipping msgpack-cxx installation"
        return 0
    fi

    local msgpack_version="cpp-7.0.0"
    echo "Installing msgpack-cxx $msgpack_version..."
    git clone https://github.com/msgpack/msgpack-c.git -b $msgpack_version msgpack_src --depth 1
    cd msgpack_src
    cmake -S. -Bbuild \
        -DCMAKE_BUILD_TYPE=Release \
        -DMSGPACK_CXX20=ON \
        -DMSGPACK_USE_BOOST=OFF \
        -DMSGPACK_BUILD_DOCS=OFF \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"
    cmake --build build --target install --parallel
    cleanup_build msgpack_src
    if [[ "$INSTALL_PREFIX" == "$DEFAULT_PREFIX" ]]; then
        echo "Remember to export msgpack-cxx_DIR=$INSTALL_PREFIX/lib/cmake/msgpack-cxx"
    fi
}

# check that we're running on Ubuntu
. /etc/os-release
echo "Detected OS: $PRETTY_NAME"
install_boost


install_msgpack

echo
echo "Dependencies installation completed successfully!"
echo "Install location: $INSTALL_PREFIX"
echo "Make sure to set CMAKE_PREFIX_PATH=$INSTALL_PREFIX when building monoprop"

# Show what was installed
echo
echo "Installed components:"
[ "$INSTALL_BOOST_UNORDERED" = true ] && echo "  ✓ Boost unordered" || echo "  ✗ Boost unordered (skipped)"
[ "$INSTALL_BOOST_TEST" = true ] && echo "  ✓ Boost Test" || echo "  ✗ Boost Test (skipped)"
[ "$INSTALL_MSGPACK" = true ] && echo "  ✓ msgpack-cxx" || echo "  ✗ msgpack-cxx (skipped)"
