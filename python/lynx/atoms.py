"""
Atomic structure for DFT calculations.

The central data object in LYNX -- analogous to torch.Tensor.
Stores cell, positions, and species. Internal units are always Bohr.

Immutable-style API: manipulation methods (repeat, deform) return new Atoms
objects rather than modifying in place.

Usage:
    import lynx

    # From scratch
    atoms = lynx.Atoms(
        cell=[[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]],
        positions=[[0, 0, 0], [1.3575, 1.3575, 1.3575]],
        symbols=["Si", "Si"],
        units="angstrom",
    )

    # From ASE
    from ase.build import bulk
    atoms = lynx.Atoms.from_ase(bulk("Si", "diamond", a=5.43))

    # Built-in crystals
    atoms = lynx.Atoms.bulk("Si", crystal="diamond", a=5.43)
"""

import json
import os
from collections import Counter

import numpy as np

from .units import ANG_TO_BOHR, BOHR_TO_ANG

# ---------------------------------------------------------------------------
# Pseudopotential search (inlined from config.py)
# ---------------------------------------------------------------------------

_XC_TO_PSP_DIR = {
    'GGA_PBE': 'ONCVPSP-PBE-PDv0.4', 'GGA_PBEsol': 'ONCVPSP-PBE-PDv0.4',
    'GGA_RPBE': 'ONCVPSP-PBE-PDv0.4', 'LDA_PZ': 'ONCVPSP-LDA-PDv0.4',
    'LDA_PW': 'ONCVPSP-LDA-PDv0.4', 'MGGA_SCAN': 'ONCVPSP-PBE-PDv0.4',
    'MGGA_RSCAN': 'ONCVPSP-PBE-PDv0.4', 'MGGA_R2SCAN': 'ONCVPSP-PBE-PDv0.4',
    'HYB_PBE0': 'ONCVPSP-PBE-PDv0.4', 'HYB_HSE': 'ONCVPSP-PBE-PDv0.4',
}


def _pick_best_psp8(elem_dir, element):
    candidates = sorted(f for f in os.listdir(elem_dir) if f.endswith('.psp8'))
    if not candidates:
        return None
    standard = f'{element}.psp8'
    if standard in candidates:
        return os.path.join(elem_dir, standard)
    semicore = f'{element}-sp.psp8'
    if semicore in candidates:
        return os.path.join(elem_dir, semicore)
    return os.path.join(elem_dir, candidates[0])


def _get_psp_search_paths(psp_dir=None):
    paths = []
    if psp_dir:
        paths.append(psp_dir)
    env_path = os.environ.get('LYNX_PSP_PATH', '')
    if env_path:
        paths.append(env_path)
    pkg_psps = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            '..', 'psps')
    if os.path.isdir(pkg_psps):
        paths.append(os.path.abspath(pkg_psps))
    return paths


def find_psp(element, xc='GGA_PBE', psp_dir=None):
    """Find .psp8 pseudopotential file for an element."""
    search_paths = _get_psp_search_paths(psp_dir)
    preferred_dir = _XC_TO_PSP_DIR.get(xc)
    all_xc_dirs = ['ONCVPSP-PBE-PDv0.4', 'ONCVPSP-LDA-PDv0.4']
    xc_dirs = ([preferred_dir] + [d for d in all_xc_dirs if d != preferred_dir]) if preferred_dir else all_xc_dirs

    for base in search_paths:
        if not os.path.isdir(base):
            continue
        for xc_subdir in xc_dirs:
            elem_dir = os.path.join(base, xc_subdir, element)
            if os.path.isdir(elem_dir):
                psp = _pick_best_psp8(elem_dir, element)
                if psp:
                    return psp
        for fname in os.listdir(base):
            if fname.endswith('.psp8') and f'_{element}_' in fname:
                return os.path.join(base, fname)

    raise FileNotFoundError(
        f"No pseudopotential found for '{element}' (xc={xc}). "
        f"Set LYNX_PSP_PATH or pass pseudo_files={{'{element}': '/path/to/{element}.psp8'}}."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_3x3_cell(cell):
    """Convert cell input to a (3,3) numpy array.

    Accepts:
      - (3,3) array: used directly
      - (3,) array or list of 3 scalars: interpreted as orthorhombic diagonal
      - scalar: interpreted as cubic cell
    """
    cell = np.asarray(cell, dtype=float)
    if cell.ndim == 0:
        # scalar -> cubic
        return np.diag([cell.item()] * 3)
    if cell.shape == (3,):
        return np.diag(cell)
    if cell.shape == (3, 3):
        return cell.copy()
    raise ValueError(
        f"cell must be scalar, (3,), or (3,3); got shape {cell.shape}"
    )


def _chemical_formula(symbols):
    """Build a reduced chemical formula string like 'Fe2O3'."""
    counts = Counter(symbols)
    # Sort by electronegativity convention: C first, H second, then alphabetical
    # Simplified: just alphabetical for now, matching common DFT output style
    parts = []
    for elem in sorted(counts):
        n = counts[elem]
        parts.append(f"{elem}{n}" if n > 1 else elem)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Atoms
# ---------------------------------------------------------------------------

class Atoms:
    """Atomic structure for DFT calculations.

    The central data object in LYNX -- analogous to torch.Tensor.
    Stores cell, positions, and species. Internal units are always Bohr.
    """

    # ---- Construction -------------------------------------------------------

    def __init__(self, *, cell, positions=None, fractional=None,
                 symbols, units="angstrom", spin=None, pseudo_files=None,
                 psp_dir=None):
        """
        Args:
            cell: (3,3) or (3,) lattice vectors. If (3,), interpreted as
                orthorhombic. Input units determined by *units* parameter.
            positions: (N,3) Cartesian coordinates. Mutually exclusive with
                *fractional*.
            fractional: (N,3) fractional coordinates. Mutually exclusive with
                *positions*.
            symbols: list of element symbols, e.g. ["Si", "Si"].
            units: "angstrom" (default) or "bohr". Applies to *cell* and
                *positions* (not *fractional*, which is dimensionless).
            spin: optional list/array of initial magnetic moments per atom
                (in units of mu_B).
            pseudo_files: optional dict ``{element: path}`` for
                pseudopotentials. Overrides automatic resolution.
            psp_dir: optional directory to search for pseudopotentials.
        """
        # --- Validate mutually-exclusive position args -----------------------
        if positions is not None and fractional is not None:
            raise ValueError("Specify positions or fractional, not both")
        if positions is None and fractional is None:
            raise ValueError("Must specify either positions or fractional")

        # --- Symbols ---------------------------------------------------------
        symbols = list(symbols)
        n_atoms = len(symbols)
        if n_atoms == 0:
            raise ValueError("symbols must contain at least one element")

        # --- Cell (always stored in Bohr) ------------------------------------
        cell_arr = _to_3x3_cell(cell)
        units = units.lower()
        if units == "angstrom":
            cell_arr = cell_arr * ANG_TO_BOHR
        elif units == "bohr":
            pass
        else:
            raise ValueError(f"units must be 'angstrom' or 'bohr', got '{units}'")

        # --- Positions (always stored in Bohr, Cartesian) --------------------
        if fractional is not None:
            frac = np.asarray(fractional, dtype=float)
            if frac.shape != (n_atoms, 3):
                raise ValueError(
                    f"fractional shape must be ({n_atoms},3), got {frac.shape}"
                )
            # Convert fractional -> Cartesian (Bohr)
            # pos_i = frac_i . cell  (row-vector convention)
            pos_arr = frac @ cell_arr
        else:
            pos_arr = np.asarray(positions, dtype=float)
            if pos_arr.shape != (n_atoms, 3):
                raise ValueError(
                    f"positions shape must be ({n_atoms},3), got {pos_arr.shape}"
                )
            if units == "angstrom":
                pos_arr = pos_arr * ANG_TO_BOHR
            # else already Bohr

        # --- Spin ------------------------------------------------------------
        if spin is not None:
            spin = np.asarray(spin, dtype=float)
            if spin.shape != (n_atoms,):
                raise ValueError(
                    f"spin shape must be ({n_atoms},), got {spin.shape}"
                )

        # --- Store internal state --------------------------------------------
        self._cell = cell_arr           # (3,3), Bohr
        self._positions = pos_arr       # (N,3), Bohr, Cartesian
        self._symbols = symbols         # list[str]
        self._spin = spin               # (N,) or None
        self._pseudo_files = dict(pseudo_files) if pseudo_files else {}
        self._psp_dir = psp_dir

    # ---- Class methods (alternate constructors) -----------------------------

    @classmethod
    def from_ase(cls, ase_atoms, **kwargs):
        """Create from an ASE Atoms object.

        Converts ASE Angstrom positions and cell to Bohr internally.

        Args:
            ase_atoms: ase.Atoms instance.
            **kwargs: Additional keyword arguments forwarded to Atoms.__init__
                (e.g. pseudo_files, psp_dir, spin). The *units* parameter is
                forced to "angstrom".

        Returns:
            Atoms
        """
        cell = np.array(ase_atoms.cell[:])         # (3,3) Angstrom
        positions = np.array(ase_atoms.positions)   # (N,3) Angstrom
        symbols = ase_atoms.get_chemical_symbols()

        # Extract initial magnetic moments if present and spin not provided
        if "spin" not in kwargs:
            try:
                magmoms = ase_atoms.get_initial_magnetic_moments()
                if magmoms is not None and np.any(np.abs(magmoms) > 0):
                    kwargs["spin"] = magmoms
            except Exception:
                pass

        # Force units to angstrom (ASE convention)
        kwargs.pop("units", None)
        kwargs.pop("fractional", None)

        return cls(
            cell=cell,
            positions=positions,
            symbols=symbols,
            units="angstrom",
            **kwargs,
        )

    @classmethod
    def read(cls, filename, format=None, **kwargs):
        """Read structure from file.

        Supported formats:
          - JSON (.json): LYNX native JSON format
          - Everything else: delegates to ASE (requires ase package)

        Args:
            filename: Path to structure file.
            format: File format string. If None, inferred from extension.
            **kwargs: Forwarded to Atoms.__init__ (e.g. pseudo_files, psp_dir).

        Returns:
            Atoms
        """
        filename = str(filename)

        # Detect format from extension if not specified
        if format is None:
            _, ext = os.path.splitext(filename)
            ext = ext.lower()
        else:
            ext = f".{format.lower()}"

        # LYNX native JSON
        if ext == ".json":
            return cls._read_json(filename, **kwargs)

        # Delegate to ASE for all other formats (POSCAR, XYZ, CIF, ...)
        try:
            import ase.io
        except ImportError:
            raise ImportError(
                f"ASE is required to read '{ext}' files. "
                "Install with: pip install ase"
            )

        ase_atoms = ase.io.read(filename, format=format if format else None)
        return cls.from_ase(ase_atoms, **kwargs)

    @classmethod
    def _read_json(cls, filename, **kwargs):
        """Read LYNX JSON config and extract atomic structure."""
        with open(filename) as f:
            data = json.load(f)

        cell = np.array(data["cell"], dtype=float)
        symbols = []
        positions = []
        for at_type in data.get("atom_types", []):
            elem = at_type["element"]
            coords = at_type["coordinates"]
            for c in coords:
                symbols.append(elem)
                positions.append(c)
        positions = np.array(positions, dtype=float)

        # JSON configs are assumed to be in Bohr
        kwargs.pop("units", None)
        return cls(
            cell=cell,
            positions=positions,
            symbols=symbols,
            units="bohr",
            **kwargs,
        )

    @classmethod
    def bulk(cls, symbol, crystal="diamond", a=None, cubic=False, **kwargs):
        """Create a bulk crystal structure.

        Uses ASE's ``ase.build.bulk`` under the hood for crystal generation,
        then converts to LYNX Atoms.

        Args:
            symbol: Chemical symbol (e.g. "Si", "Fe", "NaCl").
            crystal: Crystal structure type ("diamond", "fcc", "bcc", "hcp",
                "rocksalt", "zincblende", etc.).
            a: Lattice constant in Angstrom. If None, uses ASE's default.
            cubic: If True, use conventional cubic cell instead of primitive.
            **kwargs: Forwarded to Atoms.__init__ (e.g. pseudo_files, psp_dir).

        Returns:
            Atoms
        """
        try:
            from ase.build import bulk as ase_bulk
        except ImportError:
            raise ImportError(
                "ASE is required for Atoms.bulk(). "
                "Install with: pip install ase"
            )

        build_kwargs = {"crystalstructure": crystal, "cubic": cubic}
        if a is not None:
            build_kwargs["a"] = a

        ase_atoms = ase_bulk(symbol, **build_kwargs)
        return cls.from_ase(ase_atoms, **kwargs)

    # ---- Conversion ---------------------------------------------------------

    def to_ase(self):
        """Convert to an ASE Atoms object.

        Converts Bohr positions and cell back to Angstrom.

        Returns:
            ase.Atoms
        """
        try:
            import ase
        except ImportError:
            raise ImportError(
                "ASE is required for to_ase(). Install with: pip install ase"
            )

        ase_atoms = ase.Atoms(
            symbols=self._symbols,
            positions=self._positions * BOHR_TO_ANG,   # Bohr -> Angstrom
            cell=self._cell * BOHR_TO_ANG,              # Bohr -> Angstrom
            pbc=True,
        )

        if self._spin is not None:
            ase_atoms.set_initial_magnetic_moments(self._spin)

        return ase_atoms

    def write(self, filename, format=None):
        """Write structure to file.

        For JSON, writes LYNX native format (Bohr). For all other formats,
        delegates to ASE (Angstrom).

        Args:
            filename: Output file path.
            format: File format. If None, inferred from extension.
        """
        filename = str(filename)

        if format is None:
            _, ext = os.path.splitext(filename)
            ext = ext.lower()
        else:
            ext = f".{format.lower()}"

        if ext == ".json":
            self._write_json(filename)
            return

        # Delegate to ASE
        ase_atoms = self.to_ase()
        try:
            import ase.io
        except ImportError:
            raise ImportError(
                f"ASE is required to write '{ext}' files. "
                "Install with: pip install ase"
            )
        ase.io.write(filename, ase_atoms, format=format if format else None)

    def _write_json(self, filename):
        """Write LYNX native JSON format (Bohr)."""
        # Group atoms by element
        elem_map = {}
        for i, sym in enumerate(self._symbols):
            elem_map.setdefault(sym, []).append(self._positions[i].tolist())

        data = {
            "cell": self._cell.tolist(),
            "atom_types": [
                {"element": elem, "coordinates": coords}
                for elem, coords in elem_map.items()
            ],
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

    # ---- Properties (read-only, always in Bohr) -----------------------------

    @property
    def cell(self):
        """Lattice vectors as (3,3) numpy array in Bohr (read-only copy)."""
        return self._cell.copy()

    @property
    def positions(self):
        """Cartesian positions as (N,3) numpy array in Bohr (read-only copy)."""
        return self._positions.copy()

    @property
    def fractional(self):
        """Fractional coordinates as (N,3) numpy array (read-only).

        Computed from Cartesian positions and cell: frac = pos @ inv(cell).
        """
        return self._positions @ np.linalg.inv(self._cell)

    @property
    def symbols(self):
        """List of element symbols (read-only copy)."""
        return list(self._symbols)

    @property
    def n_atoms(self):
        """Number of atoms."""
        return len(self._symbols)

    @property
    def formula(self):
        """Chemical formula string, e.g. 'Fe2O3'."""
        return _chemical_formula(self._symbols)

    @property
    def volume(self):
        """Cell volume in Bohr^3."""
        return abs(np.linalg.det(self._cell))

    @property
    def unique_elements(self):
        """Sorted list of unique element symbols."""
        return sorted(set(self._symbols))

    @property
    def spin(self):
        """Initial magnetic moments per atom, or None if not set."""
        if self._spin is not None:
            return self._spin.copy()
        return None

    # ---- Pseudopotential resolution -----------------------------------------

    def resolve_pseudopotentials(self, xc="GGA_PBE"):
        """Auto-find .psp8 files for each unique element.

        Search order:
        1. ``self._pseudo_files`` dict (user-provided overrides)
        2. ``self._psp_dir`` directory (user-provided search path)
        3. Default PseudoDojo paths via ``lynx.config.find_psp``

        Args:
            xc: XC functional name (e.g. "GGA_PBE", "LDA_PW"). Used to
                select the correct pseudopotential set.

        Returns:
            dict mapping element symbol to absolute path of .psp8 file.

        Raises:
            FileNotFoundError: If a pseudopotential cannot be found.
        """
        resolved = {}
        for elem in self.unique_elements:
            if elem in self._pseudo_files:
                path = self._pseudo_files[elem]
                if not os.path.isfile(path):
                    raise FileNotFoundError(
                        f"Pseudopotential file not found: {path}"
                    )
                resolved[elem] = os.path.abspath(path)
            else:
                resolved[elem] = find_psp(elem, xc=xc, psp_dir=self._psp_dir)
        return resolved

    # ---- Manipulation (immutable -- returns new Atoms) ----------------------

    def repeat(self, n):
        """Repeat the cell in each direction.

        Args:
            n: int (repeat equally in all directions) or tuple (nx, ny, nz).

        Returns:
            New Atoms with the repeated structure. Cell is scaled, positions
            are replicated. Internal Bohr units are preserved.
        """
        if isinstance(n, (int, np.integer)):
            nx, ny, nz = int(n), int(n), int(n)
        else:
            nx, ny, nz = (int(x) for x in n)

        if nx < 1 or ny < 1 or nz < 1:
            raise ValueError(f"Repeat factors must be >= 1, got ({nx},{ny},{nz})")

        # New cell: scale lattice vectors
        new_cell = self._cell.copy()
        new_cell[0] *= nx
        new_cell[1] *= ny
        new_cell[2] *= nz

        # New positions: replicate atoms across all image cells
        frac = self.fractional  # (N, 3)
        new_symbols = []
        new_positions = []
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    shift = np.array([ix, iy, iz], dtype=float)
                    # In the supercell, fractional coords are scaled down
                    # new_frac = (frac + shift) / [nx, ny, nz]
                    new_frac = (frac + shift) / np.array([nx, ny, nz])
                    new_cart = new_frac @ new_cell
                    new_positions.append(new_cart)
                    new_symbols.extend(self._symbols)

        new_positions = np.vstack(new_positions)

        # Repeat spin if present
        new_spin = None
        if self._spin is not None:
            new_spin = np.tile(self._spin, nx * ny * nz)

        return Atoms(
            cell=new_cell,
            positions=new_positions,
            symbols=new_symbols,
            units="bohr",
            spin=new_spin,
            pseudo_files=self._pseudo_files.copy() if self._pseudo_files else None,
            psp_dir=self._psp_dir,
        )

    def deform(self, strain):
        """Apply a strain tensor to the cell.

        Positions are deformed affinely (fractional coordinates preserved).

        Args:
            strain: (3,3) strain tensor or (6,) Voigt notation
                [eps_xx, eps_yy, eps_zz, eps_yz, eps_xz, eps_xy].
                The deformation gradient is F = I + strain.

        Returns:
            New Atoms with the deformed cell and positions.
        """
        strain = np.asarray(strain, dtype=float)

        if strain.shape == (6,):
            # Voigt -> full symmetric tensor
            # [xx, yy, zz, yz, xz, xy]
            eps = np.zeros((3, 3))
            eps[0, 0] = strain[0]
            eps[1, 1] = strain[1]
            eps[2, 2] = strain[2]
            eps[1, 2] = eps[2, 1] = strain[3]
            eps[0, 2] = eps[2, 0] = strain[4]
            eps[0, 1] = eps[1, 0] = strain[5]
        elif strain.shape == (3, 3):
            eps = strain
        else:
            raise ValueError(
                f"strain must be (3,3) or (6,) Voigt, got shape {strain.shape}"
            )

        # Deformation gradient: F = I + eps
        F = np.eye(3) + eps

        # New cell: each lattice vector is transformed by F
        # new_cell[i] = cell[i] @ F^T  (row-vector convention)
        new_cell = self._cell @ F.T

        # Preserve fractional coordinates -> new Cartesian positions
        frac = self.fractional
        new_positions = frac @ new_cell

        return Atoms(
            cell=new_cell,
            positions=new_positions,
            symbols=list(self._symbols),
            units="bohr",
            spin=self._spin.copy() if self._spin is not None else None,
            pseudo_files=self._pseudo_files.copy() if self._pseudo_files else None,
            psp_dir=self._psp_dir,
        )

    # ---- Internal: convert to C++ config types ------------------------------

    def _to_config_atoms(self, xc="GGA_PBE"):
        """Build a list of AtomTypeInput objects for SystemConfig.

        Groups atoms by element, resolves pseudopotentials, and returns the
        C++ binding objects needed by DFTConfig / SystemConfig.

        Args:
            xc: XC functional name for pseudopotential selection.

        Returns:
            list of _core.AtomTypeInput
        """
        from . import _core

        psp_map = self.resolve_pseudopotentials(xc=xc)
        frac = self.fractional

        # Group atom indices by element (preserve order of first appearance)
        seen = {}
        order = []
        for i, sym in enumerate(self._symbols):
            if sym not in seen:
                seen[sym] = []
                order.append(sym)
            seen[sym].append(i)

        atom_types = []
        for sym in order:
            at = _core.AtomTypeInput()
            at.element = sym
            at.pseudo_file = psp_map[sym]
            at.fractional = True
            at.coords = [
                _core.Vec3(frac[i, 0], frac[i, 1], frac[i, 2])
                for i in seen[sym]
            ]
            # Set initial magnetic moments if available
            if self._spin is not None:
                at.spin = [float(self._spin[i]) for i in seen[sym]]
            atom_types.append(at)

        return atom_types

    # ---- Display ------------------------------------------------------------

    def __repr__(self):
        diag = f"{self._cell[0,0]:.2f}, {self._cell[1,1]:.2f}, {self._cell[2,2]:.2f}"
        return (
            f"Atoms({self.formula}, n_atoms={self.n_atoms}, "
            f"cell=[{diag}] Bohr)"
        )

    def __len__(self):
        return self.n_atoms

    def __eq__(self, other):
        if not isinstance(other, Atoms):
            return NotImplemented
        return (
            self._symbols == other._symbols
            and np.allclose(self._cell, other._cell)
            and np.allclose(self._positions, other._positions)
        )
