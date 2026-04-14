"""LYNX as an i-PI socket client.

i-PI is a universal force engine that drives MD simulations via a
socket protocol. LYNX acts as the "driver" -- i-PI sends atomic
positions, LYNX returns energy, forces, and stress (virial).

This script connects to a running i-PI server and serves DFT
forces from LYNX.  Converged electron density is reused as the
initial guess for the next step (density restart), which greatly
reduces SCF iterations during MD.

Architecture:
    i-PI server (MD integrator, thermostat, barostat)
        |  UNIX or TCP socket
    LYNX client (DFT energy + forces + stress)

Usage:
    # Terminal 1: start i-PI server
    i-pi ipi_nvt.xml

    # Terminal 2: start LYNX client
    python 09_ipi_client.py --species Si Si --device cpu

    # Or use the convenience script:
    bash run_ipi_nvt.sh
"""

import socket
import struct
import numpy as np

# ---------------------------------------------------------------
# i-PI socket protocol constants
# ---------------------------------------------------------------
HDRLEN = 12  # header is 12 bytes (padded string)
BOHR_TO_ANG = 0.529177249
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG
HA_TO_EV = 27.211386245988


def pad_header(msg):
    """Pad a message to HDRLEN bytes."""
    return msg.ljust(HDRLEN).encode("ascii")


def recv_header(sock):
    """Receive and decode a HDRLEN-byte header."""
    data = b""
    while len(data) < HDRLEN:
        chunk = sock.recv(HDRLEN - len(data))
        if not chunk:
            raise ConnectionError("Socket closed")
        data += chunk
    return data.decode("ascii").strip()


def recv_data(sock, nbytes):
    """Receive exactly nbytes from socket."""
    data = b""
    while len(data) < nbytes:
        chunk = sock.recv(nbytes - len(data))
        if not chunk:
            raise ConnectionError("Socket closed")
        data += chunk
    return data


def send_data(sock, data):
    """Send raw bytes."""
    sock.sendall(data)


# ---------------------------------------------------------------
# i-PI client main loop
# ---------------------------------------------------------------
def run_client(address="localhost", port=31415, unix_socket=None,
               psp_dir="../../psps", species=None, device="cpu"):
    """
    Connect to i-PI server and serve LYNX forces.

    Density restart: the converged electron density from each step is
    used as the initial guess for the next step, reducing SCF iterations
    from ~15-20 (atomic guess) to ~3-5 for typical MD displacements.

    Args:
        address: TCP hostname (ignored if unix_socket is set).
        port: TCP port (ignored if unix_socket is set).
        unix_socket: Path to UNIX domain socket (preferred).
        psp_dir: Path to pseudopotential directory.
        species: List of element symbols in atom order, e.g. ["Si", "Si"].
        device: "cpu" or "gpu".
    """
    from lynx.dft import DFT
    from lynx.atoms import Atoms as LynxAtoms

    calc = DFT(
        xc="LDA_PZ",
        kpts=[2, 2, 2],
        mesh_spacing=0.5,
        max_scf=60,
        scf_tol=1e-4,       # relaxed for MD
        mixing_beta=0.3,
        temperature=315.77,
        device=device,
        verbose=0,
    )

    # Connect
    if unix_socket:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(unix_socket)
        print(f"Connected to i-PI via UNIX socket: {unix_socket}")
    else:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((address, port))
        print(f"Connected to i-PI via TCP: {address}:{port}")

    step = 0
    prev_density = None  # density restart buffer

    try:
        while True:
            header = recv_header(sock)

            if header == "STATUS":
                send_data(sock, pad_header("READY"))

            elif header == "POSDATA":
                # 1. Receive cell vectors (3x3 in Bohr, row-major)
                cell_data = recv_data(sock, 9 * 8)
                cell = np.frombuffer(cell_data, dtype=np.float64).reshape(3, 3)

                # 2. Receive inverse cell (3x3) — not used
                recv_data(sock, 9 * 8)

                # 3. Receive number of atoms
                natoms_data = recv_data(sock, 4)
                natoms = struct.unpack("i", natoms_data)[0]

                # 4. Receive positions (natoms x 3 in Bohr)
                pos_data = recv_data(sock, natoms * 3 * 8)
                positions = np.frombuffer(pos_data, dtype=np.float64).reshape(natoms, 3)

                # 5. Build LYNX Atoms (already in Bohr)
                if species is None:
                    raise ValueError(
                        "Species not provided. Pass --species Si Si ..."
                    )

                atoms = LynxAtoms(
                    symbols=species,
                    positions=positions,  # Bohr
                    cell=cell,            # Bohr
                    units="bohr",
                    psp_dir=psp_dir,
                )

                # 6. Run DFT with density restart
                result = calc(atoms, initial_density=prev_density)

                # Save converged density for next step
                if result.density is not None:
                    prev_density = result.density.copy()

                step += 1
                print(f"Step {step}: E = {result.energy:.8f} Ha, "
                      f"Fmax = {abs(result.forces).max():.6f} Ha/Bohr")

                # 7. Send results back to i-PI
                send_data(sock, pad_header("FORCEREADY"))

                # Energy (1 float64, in Hartree)
                send_data(sock, struct.pack("d", result.energy))

                # Number of atoms
                send_data(sock, struct.pack("i", natoms))

                # Forces (natoms x 3 float64, in Ha/Bohr)
                send_data(sock, result.forces.astype(np.float64).tobytes())

                # Virial tensor (3x3 float64, in Ha)
                volume = abs(np.linalg.det(cell))
                if result.stress is not None:
                    s = result.stress
                    stress_3x3 = np.array([
                        [s[0], s[5], s[4]],
                        [s[5], s[1], s[3]],
                        [s[4], s[3], s[2]],
                    ])
                    virial = -stress_3x3 * volume
                else:
                    virial = np.zeros((3, 3))
                send_data(sock, virial.astype(np.float64).tobytes())

                # Extra string (0 bytes)
                send_data(sock, struct.pack("i", 0))

            elif header == "EXIT":
                print("Received EXIT from i-PI. Shutting down.")
                break

            else:
                print(f"Unknown header: '{header}', ignoring.")

    except (ConnectionError, BrokenPipeError):
        print("Connection closed by i-PI.")
    finally:
        sock.close()

    print(f"Completed {step} force evaluations.")


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LYNX i-PI client")
    parser.add_argument("--address", default="localhost", help="TCP host")
    parser.add_argument("--port", type=int, default=31415, help="TCP port")
    parser.add_argument("--unix", default=None, help="UNIX socket path")
    parser.add_argument("--psp-dir", default="../../psps",
                        help="Pseudopotential directory")
    parser.add_argument("--species", nargs="+", required=True,
                        help="Element symbols in order, e.g. Si Si")
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"],
                        help="Compute device (cpu or gpu)")
    args = parser.parse_args()

    run_client(
        address=args.address,
        port=args.port,
        unix_socket=args.unix,
        psp_dir=args.psp_dir,
        species=args.species,
        device=args.device,
    )
