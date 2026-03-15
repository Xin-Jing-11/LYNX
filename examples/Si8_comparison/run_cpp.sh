#!/bin/bash
# ============================================================
#  Run Si8 with the C++ executable
# ============================================================
#
#  Usage (from this directory):
#    ./run_cpp.sh
#
#  Prerequisites:
#    - Build LYNX:  cd ../../build && cmake .. && make -j
#    - The binary is at ../../build/src/lynx (or ../../build_python/src/lynx)
#
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Find the LYNX binary
LYNX_BIN=""
for candidate in ../../build/src/lynx ../../build_python/src/lynx; do
    if [ -x "$candidate" ]; then
        LYNX_BIN="$candidate"
        break
    fi
done

if [ -z "$LYNX_BIN" ]; then
    echo "ERROR: Cannot find LYNX binary. Build first:"
    echo "  mkdir -p ../../build && cd ../../build && cmake .. && make -j"
    exit 1
fi

echo "========================================"
echo " Si8 Diamond — C++ executable"
echo "========================================"
echo "Binary: $LYNX_BIN"
echo "Input:  Si8.json"
echo ""

mpirun -np 1 "$LYNX_BIN" Si8.json
