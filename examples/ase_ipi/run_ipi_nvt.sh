#!/bin/bash
# Launch i-PI server + LYNX client for NVT MD.
#
# Usage:
#   bash run_ipi_nvt.sh
#
# Prerequisites:
#   pip install ipi ase
#   pip install -e ../../python   (LYNX Python bindings)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------
# 1. Generate initial structure if not present
# ---------------------------------------------------------------
if [ ! -f init.xyz ]; then
    echo "Generating init.xyz (2x2x2 Si diamond supercell)..."
    python3 -c "
from ase.build import bulk
from ase.io import write
atoms = bulk('Si', 'diamond', a=5.43).repeat((2, 2, 2))
write('init.xyz', atoms)
print(f'Created init.xyz with {len(atoms)} atoms')
"
fi

# ---------------------------------------------------------------
# 2. Count atoms for species list
# ---------------------------------------------------------------
NATOMS=$(python3 -c "
from ase.io import read
atoms = read('init.xyz')
print(len(atoms))
")
SPECIES=$(python3 -c "
from ase.io import read
atoms = read('init.xyz')
print(' '.join(atoms.get_chemical_symbols()))
")
echo "System: $NATOMS atoms, species: $SPECIES"

# ---------------------------------------------------------------
# 3. Clean up any previous socket/output files
# ---------------------------------------------------------------
rm -f /tmp/ipi_lynx-ipi
rm -f si_nvt.out si_nvt.pos_*.xyz si_nvt.checkpoint*
rm -f RESTART EXIT

# ---------------------------------------------------------------
# 4. Launch i-PI server in background
# ---------------------------------------------------------------
echo "Starting i-PI server..."
i-pi ipi_nvt.xml &> ipi_server.log &
IPI_PID=$!
echo "i-PI PID: $IPI_PID"

# Wait for socket to appear
sleep 2

# ---------------------------------------------------------------
# 5. Launch LYNX client
# ---------------------------------------------------------------
echo "Starting LYNX client..."
python3 09_ipi_client.py \
    --unix /tmp/ipi_lynx-ipi \
    --psp-dir ../../psps \
    --species $SPECIES

# ---------------------------------------------------------------
# 6. Clean up
# ---------------------------------------------------------------
wait $IPI_PID 2>/dev/null || true
echo ""
echo "Done. Output files:"
echo "  si_nvt.out        — thermodynamic properties per step"
echo "  si_nvt.pos_*.xyz  — trajectory snapshots"
echo "  ipi_server.log    — i-PI server log"
