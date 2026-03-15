#!/usr/bin/env python3
"""LYNX DFT Web UI — Flask backend.

Manages simulation jobs, serves results, and streams SCF progress.
Parses C++ stdout in real-time for SCF iterations, energy, forces, stress.
Reads LYNXRHO binary density files directly from their headers.
"""

# Monkey-patch MUST be first — makes threading/subprocess/IO work with eventlet
import eventlet
eventlet.monkey_patch()

import base64
import json
import os
import re
import struct
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO, emit

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
JOBS_DIR = BASE_DIR / "jobs"
LYNX_BIN_CPU = PROJECT_DIR / "build" / "src" / "lynx"
LYNX_BIN_GPU = PROJECT_DIR / "build_gpu" / "src" / "lynx"
EXAMPLES_DIR = PROJECT_DIR / "examples"
PSEUDO_DIR = PROJECT_DIR / "psps"

JOBS_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = "lynx-dft-ui"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# In-memory job registry  {job_id: {status, pid, ...}}
jobs: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BOHR_TO_ANG = 0.529177249

ELEMENT_DATA = {
    "H": {"Z": 1, "color": "#FFFFFF", "radius": 0.31},
    "He": {"Z": 2, "color": "#D9FFFF", "radius": 0.28},
    "Li": {"Z": 3, "color": "#CC80FF", "radius": 1.28},
    "Be": {"Z": 4, "color": "#C2FF00", "radius": 0.96},
    "B": {"Z": 5, "color": "#FFB5B5", "radius": 0.84},
    "C": {"Z": 6, "color": "#909090", "radius": 0.76},
    "N": {"Z": 7, "color": "#3050F8", "radius": 0.71},
    "O": {"Z": 8, "color": "#FF0D0D", "radius": 0.66},
    "F": {"Z": 9, "color": "#90E050", "radius": 0.57},
    "Ne": {"Z": 10, "color": "#B3E3F5", "radius": 0.58},
    "Na": {"Z": 11, "color": "#AB5CF2", "radius": 1.66},
    "Mg": {"Z": 12, "color": "#8AFF00", "radius": 1.41},
    "Al": {"Z": 13, "color": "#BFA6A6", "radius": 1.21},
    "Si": {"Z": 14, "color": "#F0C8A0", "radius": 1.11},
    "P": {"Z": 15, "color": "#FF8000", "radius": 1.07},
    "S": {"Z": 16, "color": "#FFFF30", "radius": 1.05},
    "Cl": {"Z": 17, "color": "#1FF01F", "radius": 1.02},
    "Ar": {"Z": 18, "color": "#80D1E3", "radius": 1.06},
    "K": {"Z": 19, "color": "#8F40D4", "radius": 2.03},
    "Ca": {"Z": 20, "color": "#3DFF00", "radius": 1.76},
    "Ti": {"Z": 22, "color": "#BFC2C7", "radius": 1.60},
    "Fe": {"Z": 26, "color": "#E06633", "radius": 1.32},
    "Cu": {"Z": 29, "color": "#C88033", "radius": 1.32},
    "Zn": {"Z": 30, "color": "#7D80B0", "radius": 1.22},
    "Ga": {"Z": 31, "color": "#C28F8F", "radius": 1.22},
    "Ge": {"Z": 32, "color": "#668F8F", "radius": 1.20},
    "As": {"Z": 33, "color": "#BD80E3", "radius": 1.19},
    "Ba": {"Z": 56, "color": "#00C900", "radius": 2.22},
    "Au": {"Z": 79, "color": "#FFD123", "radius": 1.36},
}


def element_info(sym: str) -> dict:
    return ELEMENT_DATA.get(sym, {"Z": 0, "color": "#FF69B4", "radius": 1.0})


# Map element symbol → full psp8 path in PSEUDO_DIR
_PSP_CACHE: dict[str, str] = {}

def _build_psp_cache():
    """Scan PSEUDO_DIR and build element symbol → path mapping."""
    if not PSEUDO_DIR.exists():
        return
    for f in PSEUDO_DIR.glob("*.psp8"):
        # Filename like: 56_Ba_10_2.8_2.8_pbe_n_v1.0.psp8
        parts = f.stem.split("_")
        if len(parts) >= 2:
            sym = parts[1]  # e.g. "Ba"
            _PSP_CACHE[sym] = str(f)

_build_psp_cache()


def resolve_pseudo_file(pseudo_file: str, element: str) -> str:
    """Resolve a pseudo_file to a full path.

    If it's already an absolute path that exists, return as-is.
    Otherwise, look up by element symbol in the pseudopotential directory.
    """
    p = Path(pseudo_file)
    if p.is_absolute() and p.exists():
        return pseudo_file
    # Try cache
    if element in _PSP_CACHE:
        return _PSP_CACHE[element]
    # Try direct path in PSEUDO_DIR
    direct = PSEUDO_DIR / pseudo_file
    if direct.exists():
        return str(direct)
    return pseudo_file  # give back as-is, will fail with clear error


# ---------------------------------------------------------------------------
# LYNXRHO binary density reader (reads header directly, no JSON needed)
# ---------------------------------------------------------------------------

LYNXRHO_MAGIC = b"LYNX_RHO"
LYNXRHO_HEADER_SIZE = 128
# Header layout: magic(8) + version(u32) + nspin(u32) + nx(u32) + ny(u32) + nz(u32)
#   + _pad(u32) + latvec(9*f64) + dV(f64) + reserved(to 128)
LYNXRHO_HEADER_FMT = "<8s 5I I 9d d"
LYNXRHO_HEADER_STRUCT_SIZE = struct.calcsize(LYNXRHO_HEADER_FMT)


def read_density_lynxrho(filepath: Path, max_dim: int = 60) -> dict | None:
    """Read a LYNXRHO binary density file by parsing its header.

    Returns dict with shape, min, max, cell, data_base64 (Float32, C-order)
    or None if the file doesn't exist / is invalid.
    """
    if not filepath.exists():
        return None

    file_size = filepath.stat().st_size
    if file_size < LYNXRHO_HEADER_SIZE:
        return None

    with open(filepath, "rb") as f:
        hdr_bytes = f.read(LYNXRHO_HEADER_SIZE)

        # Parse header
        parsed = struct.unpack_from(LYNXRHO_HEADER_FMT, hdr_bytes)
        magic = parsed[0]
        version = parsed[1]
        nspin = parsed[2]
        nx, ny, nz = parsed[3], parsed[4], parsed[5]
        _pad = parsed[6]
        latvec_flat = parsed[7:16]  # 9 doubles
        dV = parsed[16]

        if magic != LYNXRHO_MAGIC:
            return None
        if version != 1:
            return None
        if nx == 0 or ny == 0 or nz == 0:
            return None

        # Reconstruct lattice vectors as [[ax,ay,az],[bx,by,bz],[cx,cy,cz]]
        cell = [
            [latvec_flat[0], latvec_flat[1], latvec_flat[2]],
            [latvec_flat[3], latvec_flat[4], latvec_flat[5]],
            [latvec_flat[6], latvec_flat[7], latvec_flat[8]],
        ]

        # Read total density (first spin channel for nspin=1, or sum for nspin=2)
        Nd = nx * ny * nz
        expected_data = nspin * Nd * 8  # float64
        if file_size < LYNXRHO_HEADER_SIZE + expected_data:
            return None

        rho_raw = np.frombuffer(f.read(Nd * 8), dtype=np.float64)
        if nspin == 2:
            rho_dn = np.frombuffer(f.read(Nd * 8), dtype=np.float64)
            rho_raw = rho_raw + rho_dn  # total density = up + down

    # Reshape column-major (Fortran order, as written by C++ with x-fastest)
    rho = rho_raw.reshape((nx, ny, nz), order="F")

    # Downsample for browser
    factor = max_dim / max(nx, ny, nz)
    if factor < 1.0:
        from scipy.ndimage import zoom as nd_zoom
        rho = nd_zoom(rho, factor, order=1)

    rho_f32 = rho.astype(np.float32)
    return {
        "shape": list(rho_f32.shape),
        "min": float(rho_f32.min()),
        "max": float(rho_f32.max()),
        "cell": cell,
        "data_base64": _array_to_base64(rho_f32),
    }


def read_density(job_dir: Path, max_dim: int = 60) -> dict | None:
    """Read density from job directory. Tries LYNXRHO binary first, then legacy JSON."""
    density_file = job_dir / "electron_density.bin"

    # Try LYNXRHO binary header first
    data = read_density_lynxrho(density_file, max_dim)
    if data is not None:
        return data

    # Fallback: legacy format (plain float64 array, shape from lynx_results.json)
    results_file = job_dir / "lynx_results.json"
    if not density_file.exists() or not results_file.exists():
        return None

    with open(results_file) as f:
        results = json.load(f)
    fmt = results.get("density_format", {})
    shape = tuple(fmt.get("shape", [0, 0, 0]))
    cell = results["system"]["lattice_vectors"]

    Nx, Ny, Nz = shape
    if Nx == 0:
        return None

    rho = np.fromfile(str(density_file), dtype=np.float64)
    expected = Nx * Ny * Nz
    if rho.size < expected:
        return None
    rho = rho[:expected].reshape((Nx, Ny, Nz), order="F")

    factor = max_dim / max(Nx, Ny, Nz)
    if factor < 1.0:
        from scipy.ndimage import zoom as nd_zoom
        rho = nd_zoom(rho, factor, order=1)

    rho_f32 = rho.astype(np.float32)
    return {
        "shape": list(rho_f32.shape),
        "min": float(rho_f32.min()),
        "max": float(rho_f32.max()),
        "cell": cell,
        "data_base64": _array_to_base64(rho_f32),
    }


def _array_to_base64(arr: np.ndarray) -> str:
    return base64.b64encode(arr.tobytes(order="C")).decode("ascii")


def generate_demo_density(config: dict) -> dict:
    """Generate a synthetic electron density for demo/preview purposes."""
    latvec = config["lattice"]["vectors"]
    Nx = config["grid"].get("Nx", 30)
    Ny = config["grid"].get("Ny", 30)
    Nz = config["grid"].get("Nz", 30)

    cell = np.array(latvec)
    a_len = np.linalg.norm(cell[0])
    b_len = np.linalg.norm(cell[1])
    c_len = np.linalg.norm(cell[2])

    fx = np.linspace(0, 1, Nx, endpoint=False)
    fy = np.linspace(0, 1, Ny, endpoint=False)
    fz = np.linspace(0, 1, Nz, endpoint=False)
    FX, FY, FZ = np.meshgrid(fx, fy, fz, indexing="ij")

    rho = np.zeros((Nx, Ny, Nz), dtype=np.float64)

    for atom_group in config["atoms"]:
        elem = atom_group["element"]
        info = element_info(elem)
        Z = info["Z"]
        is_frac = atom_group.get("fractional", True)

        for coord in atom_group["coordinates"]:
            if is_frac:
                af, bf, cf = coord
            else:
                cart = np.array(coord)
                inv_cell = np.linalg.inv(cell)
                frac = inv_cell @ cart
                af, bf, cf = frac

            dx = FX - af
            dy = FY - bf
            dz = FZ - cf
            dx -= np.round(dx)
            dy -= np.round(dy)
            dz -= np.round(dz)

            cart_dx = dx * a_len
            cart_dy = dy * b_len
            cart_dz = dz * c_len
            r2 = cart_dx**2 + cart_dy**2 + cart_dz**2

            sigma = 0.8 + 0.02 * Z
            amplitude = Z * 0.1
            rho += amplitude * np.exp(-r2 / (2 * sigma**2))

    max_dim = 60
    factor = max_dim / max(Nx, Ny, Nz)
    if factor < 1.0:
        from scipy.ndimage import zoom as nd_zoom
        rho = nd_zoom(rho, factor, order=1)

    rho_f32 = rho.astype(np.float32)
    return {
        "shape": list(rho_f32.shape),
        "min": float(rho_f32.min()),
        "max": float(rho_f32.max()),
        "cell": latvec,
        "data_base64": _array_to_base64(rho_f32),
    }


# ---------------------------------------------------------------------------
# Stdout parser — extracts SCF progress, energy, forces, stress from C++ output
# ---------------------------------------------------------------------------

# SCF iter  1: Etot =  -136.9227982239 Ha, SCF error =  1.234e-03, Ef =  -0.12345
# Matches both CPU and GPU SCF output:
#   SCF iter  1: Etot =  -136.92 Ha, SCF error =  1.23e-03, Ef =  -0.12, mag =  0.00
#   SCF iter  1: Etot =  -136.92 Ha, SCF error =  1.23e-03
RE_SCF_ITER = re.compile(
    r"SCF iter\s+(\d+):\s+Etot\s+=\s+([\d.eE+-]+)\s+Ha,\s+SCF error\s+=\s+([\d.eE+-]+)"
    r"(?:,\s+Ef\s+=\s+([\d.eE+-]+))?"
    r"(?:,\s+mag\s+=\s+([\d.eE+-]+))?"
)

RE_SCF_CONVERGED = re.compile(r"===== SCF (CONVERGED|NOT CONVERGED) =====")

# Energy lines:   Eband   =   -136.9227982239 Ha
RE_ENERGY = re.compile(r"^\s+(Eband|Exc|Ehart|Eself|Ec|Entropy|Etotal|Eatom|Ef)\s+=\s+([\d.eE+-]+)")

# Total forces:   Atom   1:  0.0012345678  -0.0023456789   0.0034567890
RE_FORCE = re.compile(r"^\s+Atom\s+(\d+):\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)")

# Stress: σ_xx =   -0.001234  σ_xy =   0.000000  σ_xz =   0.000000
RE_STRESS_1 = re.compile(
    r"σ_xx\s+=\s+([\d.eE+-]+)\s+σ_xy\s+=\s+([\d.eE+-]+)\s+σ_xz\s+=\s+([\d.eE+-]+)"
)
RE_STRESS_2 = re.compile(
    r"σ_yy\s+=\s+([\d.eE+-]+)\s+σ_yz\s+=\s+([\d.eE+-]+)\s+σ_zz\s+=\s+([\d.eE+-]+)"
)
RE_PRESSURE = re.compile(r"Pressure:\s+([\d.eE+-]+)\s+GPa")

RE_SCF_DID_CONVERGE = re.compile(r"SCF converged after (\d+) iterations")
RE_SCF_NOT_CONVERGE = re.compile(r"WARNING: SCF did not converge")
RE_DENSITY_WRITTEN = re.compile(r"Density written to (.+)")


class StdoutParser:
    """Parses LYNX stdout line-by-line, extracts structured results."""

    def __init__(self):
        self.scf_history: list[dict] = []
        self.energy: dict = {}
        self.forces: list[dict] = []
        self.stress_voigt: list[float] = []
        self.pressure_gpa: float | None = None
        self.converged: bool | None = None
        self.n_iterations: int | None = None
        self.fermi_energy: float | None = None
        self.density_file: str | None = None
        self._in_total_forces = False
        self._force_section = ""

    def parse_line(self, line: str) -> dict | None:
        """Parse one line. Returns an SCF progress dict if this line is an SCF iteration."""
        line = line.rstrip()

        # SCF iteration
        m = RE_SCF_ITER.search(line)
        if m:
            entry = {
                "iter": int(m.group(1)),
                "energy": float(m.group(2)),
                "residual": float(m.group(3)),
            }
            if m.group(4):
                entry["fermi"] = float(m.group(4))
                self.fermi_energy = entry["fermi"]
            if m.group(5):
                entry["mag"] = float(m.group(5))
            self.scf_history.append(entry)
            return entry

        # Convergence status
        m = RE_SCF_DID_CONVERGE.search(line)
        if m:
            self.converged = True
            self.n_iterations = int(m.group(1))
            return None

        if RE_SCF_NOT_CONVERGE.search(line):
            self.converged = False
            return None

        # Energy components
        m = RE_ENERGY.match(line)
        if m:
            key = m.group(1)
            val = float(m.group(2))
            key_map = {
                "Eband": "band", "Exc": "xc", "Ehart": "hartree",
                "Eself": "self", "Ec": "correction", "Entropy": "entropy",
                "Etotal": "total", "Eatom": "per_atom", "Ef": "fermi",
            }
            mapped = key_map.get(key, key)
            if mapped == "fermi":
                self.fermi_energy = val
            else:
                self.energy[mapped] = val
            return None

        # Force sections
        if "Total forces (Ha/Bohr):" in line:
            self._in_total_forces = True
            self.forces = []
            return None

        if self._in_total_forces:
            m = RE_FORCE.match(line)
            if m:
                self.forces.append({
                    "atom": int(m.group(1)) - 1,  # 0-indexed
                    "fx": float(m.group(2)),
                    "fy": float(m.group(3)),
                    "fz": float(m.group(4)),
                })
                return None
            elif line.strip() == "" or "=====" in line or "Stress" in line:
                self._in_total_forces = False

        # Stress
        m = RE_STRESS_1.search(line)
        if m:
            self.stress_voigt = [float(m.group(1)), float(m.group(2)), float(m.group(3))]
            return None

        m = RE_STRESS_2.search(line)
        if m:
            self.stress_voigt.extend([float(m.group(1)), float(m.group(2)), float(m.group(3))])
            return None

        m = RE_PRESSURE.search(line)
        if m:
            self.pressure_gpa = float(m.group(1))
            return None

        # Density file written
        m = RE_DENSITY_WRITTEN.search(line)
        if m:
            self.density_file = m.group(1).strip()
            return None

        return None

    def build_results(self, config: dict) -> dict:
        """Build a lynx_results.json-compatible dict from parsed data."""
        latvec = config["lattice"]["vectors"]
        Nx = config["grid"]["Nx"]
        Ny = config["grid"]["Ny"]
        Nz = config["grid"]["Nz"]

        # Build atom list from config
        atoms = []
        cell_arr = np.array(latvec)
        for ag in config["atoms"]:
            elem = ag["element"]
            info = element_info(elem)
            is_frac = ag.get("fractional", True)
            for coord in ag["coordinates"]:
                if is_frac:
                    cart = cell_arr[0] * coord[0] + cell_arr[1] * coord[1] + cell_arr[2] * coord[2]
                else:
                    cart = np.array(coord)
                atoms.append({
                    "element": elem,
                    "Z": info["Z"],
                    "position_frac": list(coord) if is_frac else [0, 0, 0],
                    "position_cart": cart.tolist() if isinstance(cart, np.ndarray) else list(cart),
                })

        # Attach element to forces
        for i, f in enumerate(self.forces):
            if i < len(atoms):
                f["element"] = atoms[i]["element"]

        results = {
            "system": {
                "lattice_vectors": latvec,
                "cell_type": config["lattice"].get("cell_type", "orthogonal"),
                "grid": {"Nx": Nx, "Ny": Ny, "Nz": Nz},
                "atoms": atoms,
                "n_atoms": len(atoms),
            },
            "scf": {
                "converged": self.converged if self.converged is not None else False,
                "n_iterations": self.n_iterations or len(self.scf_history),
                "final_residual": self.scf_history[-1]["residual"] if self.scf_history else None,
                "history": self.scf_history,
            },
            "energy": self.energy,
            "fermi_energy": self.fermi_energy,
            "forces": self.forces,
            "density_format": {
                "dtype": "float64",
                "shape": [Nx, Ny, Nz],
                "order": "column_major",
                "units": "electrons/Bohr^3",
            },
        }

        if self.stress_voigt:
            results["stress"] = {
                "voigt": self.stress_voigt,
                "pressure_GPa": self.pressure_gpa,
            }

        return results


# ---------------------------------------------------------------------------
# Routes — Pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------

@app.route("/api/examples")
def list_examples():
    examples = []
    if EXAMPLES_DIR.exists():
        for f in sorted(EXAMPLES_DIR.glob("*.json")):
            examples.append(f.stem)
    return jsonify(examples)


@app.route("/api/examples/<name>")
def get_example(name: str):
    path = EXAMPLES_DIR / f"{name}.json"
    if not path.exists():
        return jsonify({"error": "not found"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/api/elements")
def get_elements():
    return jsonify(ELEMENT_DATA)


@app.route("/api/preview-density", methods=["POST"])
def preview_density():
    """Generate a synthetic density preview from the input config (no simulation)."""
    config = request.get_json()
    if not config:
        return jsonify({"error": "no config"}), 400
    try:
        data = generate_demo_density(config)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/run", methods=["POST"])
def run_simulation():
    """Start a LYNX simulation."""
    config = request.get_json()
    if not config:
        return jsonify({"error": "no config"}), 400

    job_id = str(uuid.uuid4())[:8]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Resolve pseudopotential paths
    for ag in config.get("atoms", []):
        elem = ag.get("element", "")
        pf = ag.get("pseudo_file", elem + ".psp8")
        ag["pseudo_file"] = resolve_pseudo_file(pf, elem)

    # Inject density_output_file so LYNX dumps the binary density
    if "output" not in config:
        config["output"] = {}
    config["output"]["density_output_file"] = str(job_dir / "electron_density.bin")
    # Ensure forces and stress are computed
    config["output"]["print_forces"] = config["output"].get("print_forces", True)
    config["output"]["calc_stress"] = config["output"].get("calc_stress", True)

    input_file = job_dir / "input.json"
    with open(input_file, "w") as f:
        json.dump(config, f, indent=2)

    nprocs = config.get("parallel", {}).get("nprocs", 1)
    device = config.get("device", "cpu")

    jobs[job_id] = {
        "status": "queued",
        "dir": str(job_dir),
        "config": config,
        "device": device,
        "pid": None,
        "start_time": time.time(),
        "log_lines": [],
    }

    socketio.start_background_task(_run_job, job_id, job_dir, nprocs, device)

    return jsonify({"job_id": job_id, "status": "queued"})


def _run_job(job_id: str, job_dir: Path, nprocs: int, device: str = "cpu"):
    """Run LYNX in a subprocess, parse stdout in real-time."""
    jobs[job_id]["status"] = "running"
    socketio.emit("job_status", {"job_id": job_id, "status": "running"})

    lynx_bin_path = LYNX_BIN_GPU if device == "gpu" else LYNX_BIN_CPU
    lynx_bin = str(lynx_bin_path)
    input_file = str(job_dir / "input.json")

    if not Path(lynx_bin).exists():
        jobs[job_id]["status"] = "error"
        build_hint = (
            "cmake -B build_gpu -DUSE_CUDA=ON -DBUILD_TESTS=ON -DUSE_MKL=OFF "
            "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc && cmake --build build_gpu"
            if device == "gpu" else
            "cmake -B build -DBUILD_TESTS=ON -DUSE_MKL=OFF && cmake --build build"
        )
        jobs[job_id]["error"] = (
            f"LYNX {device.upper()} binary not found at {lynx_bin}. "
            f"Build with: {build_hint}"
        )
        socketio.emit("job_status", {
            "job_id": job_id, "status": "error",
            "error": jobs[job_id]["error"],
        })
        # Generate demo results so UI still works
        _generate_demo_results(job_id, job_dir)
        return

    # LYNX takes: lynx <input.json>  (no --output-dir)
    cmd = ["mpirun", "--allow-run-as-root", "-np", str(nprocs), lynx_bin, input_file]

    parser = StdoutParser()

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(job_dir), text=True, bufsize=1,
            preexec_fn=os.setsid,  # new process group for clean kill
        )
        jobs[job_id]["pid"] = proc.pid

        # Read stdout line-by-line in real-time
        skip_eigenvalues = False
        for line in proc.stdout:
            line_stripped = line.rstrip("\n")

            # Always store in full log file
            jobs[job_id]["log_lines"].append(line_stripped)

            # Track eigenvalue dump section (skip until blank line)
            if "Final eigenvalues" in line_stripped or "Spin " in line_stripped and ":" in line_stripped:
                skip_eigenvalues = True
                continue
            if skip_eigenvalues:
                if line_stripped.strip() == "" or line_stripped.startswith("SCF") or line_stripped.startswith("====="):
                    skip_eigenvalues = False
                else:
                    continue

            # Filter noisy debug lines
            if line_stripped.startswith(("DEBUG_OURS:", "DUMP_SCF_ITER=")):
                continue

            # Stream to browser (filtered)
            socketio.emit("job_log", {"job_id": job_id, "line": line_stripped})

            # Parse for structured data
            scf_entry = parser.parse_line(line_stripped)
            if scf_entry:
                socketio.emit("scf_progress", {"job_id": job_id, **scf_entry})

            # Yield to eventlet so emits are flushed to clients
            eventlet.sleep(0)

        proc.wait()
        retcode = proc.returncode

        if retcode == 0:
            jobs[job_id]["status"] = "completed"
        else:
            jobs[job_id]["status"] = "error"
            last_lines = "\n".join(jobs[job_id]["log_lines"][-50:])
            jobs[job_id]["error"] = f"Exit code {retcode}\n{last_lines}"

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)

    # Build results from parsed stdout
    config = jobs[job_id]["config"]
    results = parser.build_results(config)

    # Save lynx_results.json for later retrieval
    with open(job_dir / "lynx_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save SCF progress as JSONL for later retrieval
    with open(job_dir / "scf_progress.jsonl", "w") as f:
        for entry in parser.scf_history:
            f.write(json.dumps(entry) + "\n")

    # Save full stdout log
    with open(job_dir / "lynx_stdout.log", "w") as f:
        f.write("\n".join(jobs[job_id]["log_lines"]))

    socketio.emit("job_status", {
        "job_id": job_id,
        "status": jobs[job_id]["status"],
        "error": jobs[job_id].get("error", ""),
    })


def _generate_demo_results(job_id: str, job_dir: Path):
    """Generate demo results when LYNX binary is not available."""
    config_file = job_dir / "input.json"
    if not config_file.exists():
        return

    with open(config_file) as f:
        config = json.load(f)

    latvec = config["lattice"]["vectors"]
    Nx = config["grid"]["Nx"]
    Ny = config["grid"]["Ny"]
    Nz = config["grid"]["Nz"]

    # Build atom list
    atoms = []
    total_Z = 0
    for ag in config["atoms"]:
        elem = ag["element"]
        info = element_info(elem)
        is_frac = ag.get("fractional", True)
        cell = np.array(latvec)
        for coord in ag["coordinates"]:
            if is_frac:
                cart = cell[0] * coord[0] + cell[1] * coord[1] + cell[2] * coord[2]
            else:
                cart = np.array(coord)
            atoms.append({
                "element": elem,
                "Z": info["Z"],
                "position_frac": list(coord) if is_frac else [0, 0, 0],
                "position_cart": cart.tolist() if isinstance(cart, np.ndarray) else list(cart),
            })
            total_Z += info["Z"]

    n_atoms = len(atoms)

    # Fake SCF history
    scf_history = []
    demo_log_lines = []
    np.random.seed(42)
    energy = -n_atoms * 3.5
    for i in range(1, 26):
        residual = 10 ** (-1.0 - i * 0.2 + np.random.normal(0, 0.1))
        energy += np.random.normal(0, 0.01 * max(residual, 1e-6))
        fermi = round(-0.15 + np.random.normal(0, 0.001), 6)
        scf_history.append({
            "iter": i,
            "energy": round(energy, 10),
            "residual": round(residual, 8),
            "fermi": fermi,
        })
        demo_log_lines.append(
            f"SCF iter {i:3d}: Etot = {energy:18.10f} Ha, "
            f"SCF error = {residual:10.3e}, Ef = {fermi:10.5f}"
        )

    demo_log_lines.append(f"SCF converged after {len(scf_history)} iterations.")
    demo_log_lines.append("")
    demo_log_lines.append(f"===== SCF CONVERGED =====")
    demo_log_lines.append(f"  Eband   = {energy * 0.3:18.10f} Ha")
    demo_log_lines.append(f"  Exc     = {energy * 0.25:18.10f} Ha")
    demo_log_lines.append(f"  Ehart   = {abs(energy) * 1.2:18.10f} Ha")
    demo_log_lines.append(f"  Eself   = {-abs(energy) * 2.1:18.10f} Ha")
    demo_log_lines.append(f"  Ec      = {energy * 0.01:18.10f} Ha")
    demo_log_lines.append(f"  Entropy = {-0.001:18.10f} Ha")
    demo_log_lines.append(f"  Etotal  = {energy:18.10f} Ha")
    demo_log_lines.append(f"  Eatom   = {energy / n_atoms:18.10f} Ha/atom")
    demo_log_lines.append(f"  Ef      = {scf_history[-1]['fermi']:18.10f} Ha")

    # Fake forces
    forces = []
    demo_log_lines.append("")
    demo_log_lines.append("Total forces (Ha/Bohr):")
    for i, at in enumerate(atoms):
        fx = round(np.random.normal(0, 0.005), 10)
        fy = round(np.random.normal(0, 0.005), 10)
        fz = round(np.random.normal(0, 0.005), 10)
        forces.append({"atom": i, "element": at["element"], "fx": fx, "fy": fy, "fz": fz})
        demo_log_lines.append(f"  Atom {i+1:3d}: {fx:14.10f} {fy:14.10f} {fz:14.10f}")

    # Fake stress
    stress_voigt = [round(np.random.normal(-0.001, 0.0002), 6) for _ in range(6)]
    pressure = round(np.random.normal(0.5, 0.1), 4)
    demo_log_lines.append("")
    demo_log_lines.append("Stress tensor (GPa):")
    demo_log_lines.append(
        f"  σ_xx = {stress_voigt[0]:14.6f}  σ_xy = {stress_voigt[1]:14.6f}  σ_xz = {stress_voigt[2]:14.6f}"
    )
    demo_log_lines.append(
        f"  σ_yy = {stress_voigt[3]:14.6f}  σ_yz = {stress_voigt[4]:14.6f}  σ_zz = {stress_voigt[5]:14.6f}"
    )
    demo_log_lines.append(f"Pressure: {pressure:.6f} GPa")
    demo_log_lines.append("")
    demo_log_lines.append("LYNX calculation complete.")

    results = {
        "system": {
            "lattice_vectors": latvec,
            "cell_type": config["lattice"].get("cell_type", "orthogonal"),
            "grid": {"Nx": Nx, "Ny": Ny, "Nz": Nz},
            "atoms": atoms,
            "n_atoms": n_atoms,
        },
        "scf": {
            "converged": True,
            "n_iterations": len(scf_history),
            "final_residual": scf_history[-1]["residual"],
            "history": scf_history,
        },
        "energy": {
            "total": round(energy, 6),
            "band": round(energy * 0.3, 6),
            "xc": round(energy * 0.25, 6),
            "hartree": round(abs(energy) * 1.2, 6),
            "self": round(-abs(energy) * 2.1, 6),
            "correction": round(energy * 0.01, 6),
            "entropy": round(-0.001, 6),
            "per_atom": round(energy / n_atoms, 6),
        },
        "fermi_energy": scf_history[-1]["fermi"],
        "forces": forces,
        "stress": {
            "voigt": stress_voigt,
            "pressure_GPa": pressure,
        },
        "density_format": {
            "dtype": "float64",
            "shape": [Nx, Ny, Nz],
            "order": "column_major",
            "units": "electrons/Bohr^3",
        },
    }

    with open(job_dir / "lynx_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(job_dir / "scf_progress.jsonl", "w") as f:
        for entry in scf_history:
            f.write(json.dumps(entry) + "\n")

    # Write synthetic density
    density_data = generate_demo_density(config)
    rho_shape = density_data["shape"]
    rho_bytes = base64.b64decode(density_data["data_base64"])
    rho_f32 = np.frombuffer(rho_bytes, dtype=np.float32).reshape(rho_shape)
    rho_f64 = rho_f32.astype(np.float64)
    rho_f64.tofile(str(job_dir / "electron_density.bin"))

    results["density_format"]["shape"] = rho_shape
    with open(job_dir / "lynx_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save demo stdout log
    jobs[job_id]["log_lines"] = demo_log_lines
    with open(job_dir / "lynx_stdout.log", "w") as f:
        f.write("\n".join(demo_log_lines))

    jobs[job_id]["status"] = "completed"
    jobs[job_id]["demo"] = True
    socketio.emit("job_status", {
        "job_id": job_id,
        "status": "completed",
        "demo": True,
    })
    # Send SCF progress
    for entry in scf_history:
        socketio.emit("scf_progress", {"job_id": job_id, **entry})
    # Send log lines
    for line in demo_log_lines:
        socketio.emit("job_log", {"job_id": job_id, "line": line})


@app.route("/api/jobs")
def list_jobs():
    return jsonify({jid: {"status": j["status"], "start_time": j.get("start_time")}
                    for jid, j in jobs.items()})


@app.route("/api/jobs/<job_id>")
def get_job(job_id: str):
    if job_id not in jobs:
        return jsonify({"error": "not found"}), 404
    job = jobs[job_id]
    result = {"job_id": job_id, "status": job["status"]}
    job_dir = Path(job["dir"])

    results_file = job_dir / "lynx_results.json"
    if results_file.exists():
        with open(results_file) as f:
            result["results"] = json.load(f)

    result["scf_progress"] = read_scf_progress(job_dir)
    result["log"] = job.get("log_lines", [])
    if "error" in job:
        result["error"] = job["error"]
    if job.get("demo"):
        result["demo"] = True
    return jsonify(result)


@app.route("/api/jobs/<job_id>/density")
def get_density(job_id: str):
    if job_id not in jobs:
        return jsonify({"error": "not found"}), 404
    job_dir = Path(jobs[job_id]["dir"])

    data = read_density(job_dir)
    if data is None:
        # Try generating from config
        config_file = job_dir / "input.json"
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
            data = generate_demo_density(config)
        else:
            return jsonify({"error": "no density data"}), 404

    return jsonify(data)


@app.route("/api/jobs/<job_id>/log")
def get_job_log(job_id: str):
    """Return the full stdout log for a job."""
    if job_id not in jobs:
        return jsonify({"error": "not found"}), 404
    job = jobs[job_id]
    job_dir = Path(job["dir"])

    # Try in-memory first, then file
    lines = job.get("log_lines", [])
    if not lines:
        log_file = job_dir / "lynx_stdout.log"
        if log_file.exists():
            lines = log_file.read_text().splitlines()

    return jsonify({"lines": lines})


@app.route("/api/jobs/<job_id>/stop", methods=["POST"])
def stop_job(job_id: str):
    if job_id not in jobs:
        return jsonify({"error": "not found"}), 404
    job = jobs[job_id]
    if job.get("pid"):
        import signal
        pid = job["pid"]
        try:
            # Kill the whole process group (mpirun + LYNX children)
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    job["status"] = "stopped"
    socketio.emit("job_status", {"job_id": job_id, "status": "stopped"})
    return jsonify({"status": "stopped"})


def read_scf_progress(job_dir: Path) -> list[dict]:
    """Read scf_progress.jsonl, return list of iteration dicts."""
    progress_file = job_dir / "scf_progress.jsonl"
    if not progress_file.exists():
        return []
    entries = []
    with open(progress_file) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@socketio.on("connect")
def ws_connect():
    emit("connected", {"msg": "Connected to LYNX Web UI"})


@socketio.on("subscribe_job")
def ws_subscribe(data):
    job_id = data.get("job_id")
    if job_id and job_id in jobs:
        job_dir = Path(jobs[job_id]["dir"])
        progress = read_scf_progress(job_dir)
        for entry in progress:
            emit("scf_progress", {"job_id": job_id, **entry})
        # Also send existing log lines
        for line in jobs[job_id].get("log_lines", []):
            emit("job_log", {"job_id": job_id, "line": line})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LYNX DFT Web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"\n  LYNX DFT Web UI")
    print(f"  Open http://{args.host}:{args.port} in your browser\n")

    socketio.run(app, host=args.host, port=args.port, debug=args.debug)
