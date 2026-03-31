#!/bin/bash
# Run all CPU kernel benchmarks with thread scaling
# Usage: ./run_all.sh [build_dir]
#   build_dir defaults to ./build

set -e

BUILD_DIR="${1:-build}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Build if needed
if [ ! -d "$BUILD_DIR" ]; then
    echo "=== Building benchmarks ==="
    cmake -B "$BUILD_DIR" -S "$SCRIPT_DIR" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$BUILD_DIR" -j$(nproc)
    echo ""
fi

BENCHMARKS=(
    bench_laplacian
    bench_hamiltonian
    bench_gradient
    bench_chebyshev
    bench_nonlocal
    bench_xc
    bench_mixing
)

THREAD_COUNTS=(1 2 4 8)

echo "================================================================"
echo "  LYNX CPU Kernel Benchmarks — $(date)"
echo "  CPU: $(lscpu | grep 'Model name' | sed 's/.*:\s*//')"
echo "  Cores: $(nproc)"
echo "================================================================"
echo ""

for bench in "${BENCHMARKS[@]}"; do
    BINARY="$BUILD_DIR/$bench"
    if [ ! -x "$BINARY" ]; then
        echo "--- $bench: SKIPPED (not built) ---"
        echo ""
        continue
    fi

    echo "================================================================"
    echo "  $bench — Thread Scaling"
    echo "================================================================"

    for nt in "${THREAD_COUNTS[@]}"; do
        if [ "$nt" -gt "$(nproc)" ]; then
            continue
        fi
        echo ""
        echo ">>> OMP_NUM_THREADS=$nt"
        OMP_NUM_THREADS=$nt OMP_PROC_BIND=close OMP_PLACES=cores "$BINARY"
    done
    echo ""
    echo ""
done

echo "================================================================"
echo "  All benchmarks complete"
echo "================================================================"
