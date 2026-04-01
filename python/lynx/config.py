"""
Programmatic configuration builder for LYNX — no JSON files needed.

Usage:
    import lynx
    from lynx.config import DFTConfig

    # Pseudopotential is auto-selected based on xc functional:
    config = DFTConfig(
        cell=[[10.26, 0, 0], [0, 10.26, 0], [0, 0, 10.26]],  # Bohr
        positions=[[0,0,0], [2.565,2.565,2.565]],              # Bohr (Cartesian)
        symbols=['Si', 'Si'],
        kpts=(2, 2, 2),
        Nstates=10,
        xc='GGA_PBE',     # -> automatically picks psps/ONCVPSP-PBE-PDv0.4/Si/Si.psp8
    )
    calc = config.create_calculator()
    calc.run()
"""

import os
import numpy as np

from . import _core
from .units import ANG_TO_BOHR

# XC functional -> PseudoDojo submodule directory name
_XC_TO_PSP_DIR = {
    'GGA_PBE':    'ONCVPSP-PBE-PDv0.4',
    'GGA_PBEsol': 'ONCVPSP-PBE-PDv0.4',   # PBEsol uses PBE pseudopotentials
    'GGA_RPBE':   'ONCVPSP-PBE-PDv0.4',   # RPBE uses PBE pseudopotentials
    'LDA_PZ':     'ONCVPSP-LDA-PDv0.4',
    'LDA_PW':     'ONCVPSP-LDA-PDv0.4',
    'MGGA_SCAN':  'ONCVPSP-PBE-PDv0.4',   # mGGA uses PBE pseudopotentials
    'MGGA_RSCAN': 'ONCVPSP-PBE-PDv0.4',
    'MGGA_R2SCAN':'ONCVPSP-PBE-PDv0.4',
    'HYB_PBE0':   'ONCVPSP-PBE-PDv0.4',   # Hybrids use PBE pseudopotentials
    'HYB_HSE':    'ONCVPSP-PBE-PDv0.4',
}

# Preferred .psp8 file for each element (shortest name = standard, no semicore).
# Falls back to first .psp8 found if element isn't listed here.
def _pick_best_psp8(elem_dir, element):
    """Pick the best .psp8 file from a PseudoDojo element directory.

    Preference order:
    1. {Element}.psp8          (standard, no semicore)
    2. {Element}-sp.psp8       (with semicore — needed for elements like Ba, Ti, Fe)
    3. First .psp8 found       (fallback)
    """
    candidates = sorted(f for f in os.listdir(elem_dir) if f.endswith('.psp8'))
    if not candidates:
        return None

    # Exact match: Si.psp8, O.psp8, etc.
    standard = f'{element}.psp8'
    if standard in candidates:
        return os.path.join(elem_dir, standard)

    # Semicore: Fe-sp.psp8, Ba-sp.psp8, etc.
    semicore = f'{element}-sp.psp8'
    if semicore in candidates:
        return os.path.join(elem_dir, semicore)

    # Fallback: first available
    return os.path.join(elem_dir, candidates[0])


def _get_psp_search_paths(psp_dir=None):
    """Build the list of directories to search for pseudopotentials."""
    paths = []
    if psp_dir:
        paths.append(psp_dir)
    env_path = os.environ.get('LYNX_PSP_PATH', '')
    if env_path:
        paths.append(env_path)
    # Package-relative psps/ directory (LYNX/psps/)
    pkg_psps = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            '..', 'psps')
    if os.path.isdir(pkg_psps):
        paths.append(os.path.abspath(pkg_psps))
    return paths


def find_psp(element, xc='GGA_PBE', psp_dir=None):
    """Find the pseudopotential file for an element, matched to the XC functional.

    Automatically selects the correct PseudoDojo submodule based on the XC
    functional (PBE-family -> ONCVPSP-PBE-PDv0.4, LDA-family -> ONCVPSP-LDA-PDv0.4).

    Parameters
    ----------
    element : str
        Chemical symbol (e.g. 'Si', 'Fe', 'O').
    xc : str
        XC functional name ('GGA_PBE', 'LDA_PW', etc.). Determines which
        pseudopotential set to use.
    psp_dir : str, optional
        Additional directory to search.

    Returns
    -------
    str
        Absolute path to the .psp8 file.

    Raises
    ------
    FileNotFoundError
        If no pseudopotential is found for the element.
    """
    search_paths = _get_psp_search_paths(psp_dir)

    # Determine the preferred submodule directory for this XC functional
    preferred_dir = _XC_TO_PSP_DIR.get(xc)

    # Order: preferred XC match first, then the other
    all_xc_dirs = ['ONCVPSP-PBE-PDv0.4', 'ONCVPSP-LDA-PDv0.4']
    if preferred_dir:
        xc_dirs = [preferred_dir] + [d for d in all_xc_dirs if d != preferred_dir]
    else:
        xc_dirs = all_xc_dirs

    for base in search_paths:
        if not os.path.isdir(base):
            continue

        # Search PseudoDojo-style: {base}/ONCVPSP-{XC}-PDv0.4/{element}/
        for xc_subdir in xc_dirs:
            elem_dir = os.path.join(base, xc_subdir, element)
            if os.path.isdir(elem_dir):
                psp = _pick_best_psp8(elem_dir, element)
                if psp:
                    return psp

        # Fallback: flat directory with old naming convention ({base}/*_{element}_*.psp8)
        for fname in os.listdir(base):
            if fname.endswith('.psp8') and f'_{element}_' in fname:
                return os.path.join(base, fname)

    raise FileNotFoundError(
        f"No pseudopotential found for '{element}' (xc={xc}). "
        f"Run 'git submodule update --init' to fetch PseudoDojo pseudopotentials, "
        f"or set LYNX_PSP_PATH, or pass pseudo_files={{'{element}': '/path/to/{element}.psp8'}}."
    )


class DFTConfig:
    """Build a LYNX SystemConfig programmatically.

    All lengths are in **Bohr** and all energies in **Hartree** (LYNX internal units).
    Use `DFTConfig.from_ase()` for automatic Angstrom->Bohr conversion from ASE Atoms.

    If `pseudo_files` is not provided, pseudopotentials are automatically selected
    from the PseudoDojo submodules based on the `xc` functional:
      - GGA_PBE / GGA_PBEsol / GGA_RPBE -> psps/ONCVPSP-PBE-PDv0.4/
      - LDA_PZ / LDA_PW                  -> psps/ONCVPSP-LDA-PDv0.4/

    Parameters
    ----------
    cell : array_like (3,3)
        Lattice vectors in Bohr, row-major: cell[i] is the i-th lattice vector.
    positions : array_like (N, 3), optional
        Cartesian atomic positions in Bohr. Mutually exclusive with fractional.
    fractional : array_like (N, 3), optional
        Fractional atomic coordinates. Mutually exclusive with positions.
    symbols : list of str
        Chemical symbols, one per atom (e.g. ['Si', 'Si']).
    pseudo_files : dict, optional
        Map element -> pseudopotential file path. If not given, auto-selected
        from PseudoDojo based on xc.
    psp_dir : str, optional
        Additional directory to search for .psp8 files.
    Nx, Ny, Nz : int, optional
        FD grid dimensions. If omitted, computed from mesh_spacing.
    mesh_spacing : float, optional
        Grid spacing in Bohr (default: 0.5).
    fd_order : int
        Finite-difference order (default: 12).
    xc : str
        XC functional: 'GGA_PBE', 'LDA_PW', 'GGA_PBEsol', etc. (default: 'GGA_PBE').
    kpts : tuple of 3 ints
        Monkhorst-Pack k-point grid (default: (1,1,1) = Gamma only).
    kpt_shift : tuple of 3 floats
        K-point shift (default: (0,0,0)).
    Nstates : int, optional
        Number of electronic states. If 0, auto-computed as Nelectron/2 + 10.
    spin : str
        Spin type: 'none', 'collinear', 'noncollinear' (default: 'none').
    elec_temp : float
        Electronic temperature in Kelvin (default: 300).
    smearing : str
        Smearing type: 'gaussian' or 'fermi-dirac' (default: 'gaussian').
    max_scf_iter : int
        Maximum SCF iterations (default: 100).
    scf_tol : float
        SCF convergence tolerance (default: 1e-6).
    mixing_param : float
        Density mixing parameter (default: 0.3).
    bc : str or tuple of str
        Boundary conditions: 'periodic' or 'dirichlet', or tuple of 3
        (default: 'periodic').
    """

    def __init__(self, cell, symbols, positions=None, fractional=None,
                 pseudo_files=None, psp_dir=None,
                 Nx=0, Ny=0, Nz=0, mesh_spacing=0.5, fd_order=12,
                 xc='GGA_PBE', kpts=(1, 1, 1), kpt_shift=(0, 0, 0),
                 Nstates=0, spin='none', elec_temp=300.0, smearing='gaussian',
                 max_scf_iter=100, scf_tol=1e-6, mixing_param=0.3,
                 bc='periodic'):

        cell = np.asarray(cell, dtype=float)
        if cell.shape != (3, 3):
            raise ValueError(f"cell must be (3,3), got {cell.shape}")

        symbols = list(symbols)
        N = len(symbols)

        if positions is not None and fractional is not None:
            raise ValueError("Specify positions or fractional, not both")
        if positions is None and fractional is None:
            raise ValueError("Must specify either positions or fractional")

        is_frac = fractional is not None
        coords = np.asarray(fractional if is_frac else positions, dtype=float)
        if coords.shape != (N, 3):
            raise ValueError(f"coordinates shape must be ({N},3), got {coords.shape}")

        # Determine cell type
        off_diag = [cell[0, 1], cell[0, 2], cell[1, 0], cell[1, 2], cell[2, 0], cell[2, 1]]
        is_orth = all(abs(v) < 1e-10 for v in off_diag)

        # Parse boundary conditions
        if isinstance(bc, str):
            bc = (bc, bc, bc)
        bc_map = {'periodic': _core.BCType.Periodic, 'dirichlet': _core.BCType.Dirichlet}

        # Auto-compute grid dimensions from mesh_spacing
        if Nx <= 0 or Ny <= 0 or Nz <= 0:
            lengths = np.linalg.norm(cell, axis=1)
            h = mesh_spacing
            if Nx <= 0: Nx = max(1, int(np.ceil(lengths[0] / h)))
            if Ny <= 0: Ny = max(1, int(np.ceil(lengths[1] / h)))
            if Nz <= 0: Nz = max(1, int(np.ceil(lengths[2] / h)))

        # Parse XC
        xc_map = {
            'GGA_PBE': _core.XCType.GGA_PBE,
            'LDA_PZ': _core.XCType.LDA_PZ,
            'LDA_PW': _core.XCType.LDA_PW,
            'GGA_PBEsol': _core.XCType.GGA_PBEsol,
            'GGA_RPBE': _core.XCType.GGA_RPBE,
            'MGGA_SCAN': _core.XCType.MGGA_SCAN,
            'MGGA_RSCAN': _core.XCType.MGGA_RSCAN,
            'MGGA_R2SCAN': _core.XCType.MGGA_R2SCAN,
            'HYB_PBE0': _core.XCType.HYB_PBE0,
            'HYB_HSE': _core.XCType.HYB_HSE,
        }
        if xc not in xc_map:
            raise ValueError(f"Unknown xc '{xc}'. Options: {list(xc_map.keys())}")

        # Parse spin
        spin_map = {
            'none': _core.SpinType.NoSpin,
            'collinear': _core.SpinType.Collinear,
            'noncollinear': _core.SpinType.NonCollinear,
        }
        if spin not in spin_map:
            raise ValueError(f"Unknown spin '{spin}'. Options: {list(spin_map.keys())}")

        # Parse smearing
        smearing_map = {
            'gaussian': _core.SmearingType.GaussianSmearing,
            'fermi-dirac': _core.SmearingType.FermiDirac,
        }
        if smearing not in smearing_map:
            raise ValueError(f"Unknown smearing '{smearing}'. Options: {list(smearing_map.keys())}")

        # Resolve pseudopotentials — auto-select based on xc if not provided
        if pseudo_files is None:
            pseudo_files = {}
        unique_symbols = list(dict.fromkeys(symbols))  # preserve order
        psp_resolved = {}
        for sym in unique_symbols:
            if sym in pseudo_files:
                psp_resolved[sym] = pseudo_files[sym]
            else:
                psp_resolved[sym] = find_psp(sym, xc=xc, psp_dir=psp_dir)

        # Build SystemConfig
        cfg = _core.SystemConfig()

        # Lattice
        lv = _core.Mat3()
        for i in range(3):
            for j in range(3):
                lv.set(i, j, cell[i, j])
        cfg.latvec = lv
        cfg.cell_type = _core.CellType.Orthogonal if is_orth else _core.CellType.NonOrthogonal

        # Grid
        cfg.Nx = Nx
        cfg.Ny = Ny
        cfg.Nz = Nz
        cfg.mesh_spacing = mesh_spacing
        cfg.fd_order = fd_order
        cfg.bcx = bc_map[bc[0]]
        cfg.bcy = bc_map[bc[1]]
        cfg.bcz = bc_map[bc[2]]

        # Atoms — group by element
        atom_types_list = []
        for sym in unique_symbols:
            at = _core.AtomTypeInput()
            at.element = sym
            at.pseudo_file = psp_resolved[sym]
            at.fractional = is_frac
            indices = [i for i, s in enumerate(symbols) if s == sym]
            at.coords = [_core.Vec3(coords[i, 0], coords[i, 1], coords[i, 2])
                         for i in indices]
            atom_types_list.append(at)
        cfg.atom_types = atom_types_list

        # Electronic
        cfg.Nstates = Nstates
        cfg.spin_type = spin_map[spin]
        cfg.elec_temp = elec_temp
        cfg.smearing = smearing_map[smearing]
        cfg.xc = xc_map[xc]

        # K-points
        cfg.Kx = kpts[0]
        cfg.Ky = kpts[1]
        cfg.Kz = kpts[2]
        cfg.kpt_shift = _core.Vec3(kpt_shift[0], kpt_shift[1], kpt_shift[2])

        # SCF
        cfg.max_scf_iter = max_scf_iter
        cfg.scf_tol = scf_tol
        cfg.mixing_param = mixing_param

        self._config = cfg

    @classmethod
    def from_ase(cls, atoms, pseudo_files=None, psp_dir=None, **kwargs):
        """Create DFTConfig from an ASE Atoms object.

        Converts ASE units (Angstrom) to LYNX units (Bohr) automatically.
        If pseudo_files is not provided, pseudopotentials are auto-selected
        based on the xc functional.

        Parameters
        ----------
        atoms : ase.Atoms
            ASE Atoms object with cell, positions, and pbc set.
        pseudo_files : dict, optional
            Map element -> pseudopotential file path.
        psp_dir : str, optional
            Directory to search for .psp8 files.
        **kwargs
            Additional keyword arguments passed to DFTConfig (xc, kpts, etc.).

        Returns
        -------
        DFTConfig
        """
        cell_bohr = np.array(atoms.cell) * ANG_TO_BOHR
        positions_bohr = np.array(atoms.positions) * ANG_TO_BOHR
        symbols = atoms.get_chemical_symbols()

        # Map ASE pbc to boundary conditions
        pbc = atoms.pbc
        bc = tuple('periodic' if p else 'dirichlet' for p in pbc)

        return cls(
            cell=cell_bohr,
            positions=positions_bohr,
            symbols=symbols,
            pseudo_files=pseudo_files,
            psp_dir=psp_dir,
            bc=bc,
            **kwargs,
        )

    @property
    def system_config(self):
        """Return the underlying SystemConfig C++ object."""
        return self._config

    def create_calculator(self, auto_run=False, use_gpu=False):
        """Create a LYNX Calculator from this config.

        Parameters
        ----------
        auto_run : bool
            If True, run SCF immediately after setup.
        use_gpu : bool
            If True, use GPU acceleration (requires CUDA build).

        Returns
        -------
        lynx.Calculator
        """
        _core._ensure_mpi()
        calc = _core.Calculator()
        calc.set_config(self._config)
        calc.set_use_gpu(use_gpu)
        calc.setup()
        if auto_run:
            calc.run()
        return calc
