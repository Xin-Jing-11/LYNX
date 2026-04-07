"""Self-consistent DFT calculator -- the core of LYNX's Python interface.

Usage:
    calc = lynx.DFT(xc="PBE", kpts=[2, 2, 2])
    result = calc(atoms)
    print(result.energy)
"""

import numpy as np
from typing import Optional, Union

from lynx.result import DFTResult, EnergyDecomposition
from lynx.device import device as validate_device
from lynx import xc as xc_module
from lynx.xc.base import Functional
from lynx.solvers.eigen import EigenSolver, CheFSI
from lynx.solvers.poisson import PoissonSolver, AAR
from lynx.solvers.mixer import Mixer, PulayMixer
from lynx.solvers.occupation import Occupation, FermiDirac, Gaussian


class DFT:
    """Self-consistent DFT calculator.

    Callable module that takes Atoms and returns DFTResult.
    Like torch.nn.Module -- configurable, composable, device-aware.

    Example:
        calc = DFT(xc="PBE", kpts=[2, 2, 2], device="gpu")
        result = calc(atoms)
        print(result.energy)
    """

    def __init__(self, *,
                 xc: Union[str, Functional] = "PBE",
                 kpts=(1, 1, 1),
                 kpt_shift=(0, 0, 0),
                 mesh_spacing: Optional[float] = None,
                 grid_shape: Optional[tuple] = None,
                 fd_order: int = 12,
                 smearing: str = "gaussian",
                 temperature: float = 315.77,
                 n_bands: Optional[int] = None,
                 spin: Optional[str] = None,
                 max_scf: int = 100,
                 scf_tol: float = 1e-6,
                 mixing: Union[str, Mixer] = "pulay",
                 mixing_beta: float = 0.3,
                 mixing_history: int = 7,
                 eigensolver: Optional[EigenSolver] = None,
                 poisson: Optional[PoissonSolver] = None,
                 occupation: Optional[Occupation] = None,
                 device: str = "cpu",
                 verbose: int = 1,
                 # Advanced options
                 bc: str = "periodic",
                 cheb_degree: int = 0,
                 rho_trigger: int = 4,
                 restart_density: Optional[str] = None,
                 ):
        """Configure a DFT calculator.

        Args:
            xc: Exchange-correlation functional. String name ("PBE", "SCAN",
                "HSE06") or lynx.xc.Functional instance.
            kpts: Monkhorst-Pack k-point grid (Kx, Ky, Kz).
            kpt_shift: K-point shift in fractional coordinates.
            mesh_spacing: Target mesh spacing in Bohr (auto-computes
                grid_shape). Mutually exclusive with grid_shape.
            grid_shape: Explicit (Nx, Ny, Nz). Mutually exclusive with
                mesh_spacing.
            fd_order: Finite-difference stencil order (default 12).
            smearing: "gaussian" or "fermi-dirac".
            temperature: Electronic temperature in Kelvin.
            n_bands: Number of bands (auto if None).
            spin: None (non-spin), "collinear", or "noncollinear".
            max_scf: Maximum SCF iterations.
            scf_tol: SCF convergence tolerance (Ha/atom).
            mixing: "pulay", "anderson", or Mixer instance.
            mixing_beta: Mixing parameter.
            mixing_history: Pulay/Anderson history depth.
            eigensolver: EigenSolver instance (default: CheFSI).
            poisson: PoissonSolver instance (default: AAR).
            occupation: Occupation instance (default: from smearing).
            device: "cpu" or "gpu".
            verbose: 0=silent, 1=summary, 2=per-step.
            bc: Boundary condition type ("periodic" or "dirichlet").
            cheb_degree: Chebyshev polynomial degree for CheFSI (0=auto).
            rho_trigger: SCF iteration to start updating density mixing.
            restart_density: Path to density restart file, or None.
        """
        if mesh_spacing is not None and grid_shape is not None:
            raise ValueError(
                "mesh_spacing and grid_shape are mutually exclusive"
            )

        # XC functional
        if isinstance(xc, str):
            self._xc = xc_module.get(xc)
        elif isinstance(xc, Functional):
            self._xc = xc
        else:
            raise TypeError(f"xc must be str or Functional, got {type(xc)}")

        # K-points
        self._kpts = tuple(int(k) for k in kpts)
        self._kpt_shift = tuple(float(s) for s in kpt_shift)

        # Grid
        self._mesh_spacing = mesh_spacing
        self._grid_shape = (
            tuple(int(s) for s in grid_shape) if grid_shape else None
        )
        self._fd_order = fd_order

        # Electronic structure
        self._smearing = smearing.lower()
        self._temperature = temperature
        self._n_bands = n_bands
        self._spin = spin
        self._bc = bc
        self._cheb_degree = cheb_degree
        self._rho_trigger = rho_trigger
        self._restart_density = restart_density

        # SCF parameters
        self._max_scf = max_scf
        self._scf_tol = scf_tol
        self._mixing_beta = mixing_beta
        self._mixing_history = mixing_history

        # Components -- eigensolver, Poisson solver, mixer, occupation
        self._eigensolver = eigensolver or CheFSI()
        self._poisson = poisson or AAR()

        if isinstance(mixing, str):
            if mixing.lower() == "pulay":
                self._mixer = PulayMixer(
                    beta=mixing_beta, history=mixing_history
                )
            elif mixing.lower() == "anderson":
                from lynx.solvers.mixer import AndersonMixer
                self._mixer = AndersonMixer(
                    beta=mixing_beta, history=mixing_history
                )
            else:
                raise ValueError(f"Unknown mixer: '{mixing}'")
        elif isinstance(mixing, Mixer):
            self._mixer = mixing
        else:
            raise TypeError(
                f"mixing must be str or Mixer, got {type(mixing)}"
            )

        if occupation is not None:
            self._occupation = occupation
        elif self._smearing == "fermi-dirac":
            self._occupation = FermiDirac(temperature=temperature)
        else:
            self._occupation = Gaussian(temperature=temperature)

        # Device
        self._device = validate_device(device)

        # Verbosity
        self._verbose = verbose

        # Internal state (set during __call__)
        self._calculator = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, atoms) -> DFTResult:
        """Run full DFT calculation on atoms.

        Args:
            atoms: lynx.Atoms instance.

        Returns:
            DFTResult with energy, forces, stress, eigenvalues, etc.
        """
        return self._run(atoms, compute_forces=True, compute_stress=True)

    def energy(self, atoms) -> float:
        """Compute only energy (faster -- skips forces/stress)."""
        result = self._run(atoms, compute_forces=False, compute_stress=False)
        return result.energy

    def forces(self, atoms) -> np.ndarray:
        """Compute energy + forces."""
        result = self._run(atoms, compute_forces=True, compute_stress=False)
        return result.forces

    def stress(self, atoms) -> np.ndarray:
        """Compute energy + stress."""
        result = self._run(atoms, compute_forces=False, compute_stress=True)
        return result.stress

    # ------------------------------------------------------------------
    # Device management (PyTorch-style)
    # ------------------------------------------------------------------

    def to(self, device_name: str) -> "DFT":
        """Move calculator to device. Returns self for chaining.

        Example:
            calc = DFT(xc="PBE").to("gpu")
        """
        self._device = validate_device(device_name)
        return self

    @property
    def device(self) -> str:
        """Current compute device ("cpu" or "gpu")."""
        return self._device

    # ------------------------------------------------------------------
    # Component access
    # ------------------------------------------------------------------

    @property
    def xc(self) -> Functional:
        """Exchange-correlation functional."""
        return self._xc

    @xc.setter
    def xc(self, value):
        if isinstance(value, str):
            self._xc = xc_module.get(value)
        elif isinstance(value, Functional):
            self._xc = value
        else:
            raise TypeError(f"xc must be str or Functional, got {type(value)}")

    @property
    def eigensolver(self) -> EigenSolver:
        """Eigensolver used in the SCF loop."""
        return self._eigensolver

    @eigensolver.setter
    def eigensolver(self, value):
        if not isinstance(value, EigenSolver):
            raise TypeError(f"Expected EigenSolver, got {type(value)}")
        self._eigensolver = value

    @property
    def mixer(self) -> Mixer:
        """Density/potential mixer used in the SCF loop."""
        return self._mixer

    @mixer.setter
    def mixer(self, value):
        if not isinstance(value, Mixer):
            raise TypeError(f"Expected Mixer, got {type(value)}")
        self._mixer = value

    @property
    def kpts(self) -> tuple:
        """Monkhorst-Pack k-point grid."""
        return self._kpts

    # ------------------------------------------------------------------
    # Hooks for subclassing
    # ------------------------------------------------------------------

    def on_scf_step(self, step: int, state: dict) -> None:
        """Override for per-step callbacks.

        Args:
            step: SCF iteration number (1-based).
            state: dict with keys 'energy', 'error', 'fermi_energy'.
        """
        pass

    def on_converged(self, result: DFTResult) -> None:
        """Override for post-convergence actions."""
        pass

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _run(self, atoms, compute_forces=True, compute_stress=True) -> DFTResult:
        """Run the calculation via the C++ Calculator backend."""
        from lynx import _core  # C++ pybind11 module

        # Build SystemConfig from Atoms + DFT settings
        config = self._build_config(atoms)

        # Create and initialise C++ calculator
        calc = _core.Calculator()
        calc.set_config(config)
        calc.set_use_gpu(self._device == "gpu")
        calc.setup()  # MPI_COMM_SELF for single-process Python usage

        self._calculator = calc

        # Run SCF
        total_energy = calc.run()

        # Collect energy decomposition
        energy_dict = calc.energy  # property returning dict
        energies = EnergyDecomposition(
            band=energy_dict.get("Eband", 0.0),
            xc=energy_dict.get("Exc", 0.0),
            hartree=energy_dict.get("Ehart", 0.0),
            self_energy=energy_dict.get("Eself", 0.0),
            correction=energy_dict.get("Ec", 0.0),
            entropy=energy_dict.get("Entropy", 0.0),
            exx=energy_dict.get("Eexx", 0.0),
            total=energy_dict.get("Etotal", 0.0),
            per_atom=energy_dict.get("Eatom", 0.0),
        )

        # Forces
        forces_arr = None
        if compute_forces:
            f_flat = calc.compute_forces()
            n_atoms = atoms.n_atoms
            forces_arr = np.array(f_flat).reshape(n_atoms, 3)

        # Stress
        stress_arr = None
        pressure = 0.0
        if compute_stress:
            s = calc.compute_stress()
            stress_arr = np.array(s)
            # Pressure = -1/3 * Tr(sigma), convert from Ha/Bohr^3 to GPa
            HA_BOHR3_TO_GPA = 29421.01569650548
            pressure = -(stress_arr[0] + stress_arr[3] + stress_arr[5]) / 3.0 * HA_BOHR3_TO_GPA

        # Eigenvalues and occupations (all spins x k-points)
        wfn = calc.get_wavefunction()
        eigenvalues = []
        occupations = []
        for s in range(wfn.Nspin()):
            for k in range(wfn.Nkpts()):
                eig = np.array(wfn.eigenvalues(s, k))
                occ = np.array(wfn.occupations(s, k))
                eigenvalues.append(eig)
                occupations.append(occ)

        # Electron density (best-effort; not all backends expose this)
        density = None
        try:
            scf = calc.scf_solver
            rho = scf.density()
            density = np.array(rho.rho_total())
        except Exception:
            pass

        result = DFTResult(
            energy=total_energy,
            forces=forces_arr,
            stress=stress_arr,
            pressure=pressure,
            energies=energies,
            eigenvalues=eigenvalues,
            occupations=occupations,
            fermi_energy=calc.fermi_energy,
            density=density,
            converged=calc.converged,
            n_iterations=0,  # TODO: expose iteration count from C++
            atoms=atoms,
        )

        # Post-convergence hook
        self.on_converged(result)

        if self._verbose >= 1:
            print(result.summary())

        return result

    def _build_config(self, atoms):
        """Build a C++ SystemConfig from Atoms + DFT settings.

        Mirrors what the C++ InputParser would produce so the Calculator
        receives an identical configuration regardless of whether the
        calculation was launched from an input file or the Python API.
        """
        from lynx import _core

        config = _core.SystemConfig()

        # --- Lattice vectors ---
        cell = atoms.cell  # (3, 3) in Bohr
        config.latvec = _core.Mat3.from_numpy(cell.astype(float))

        # Detect orthogonal vs non-orthogonal cell
        off_diag = (abs(cell[0, 1]) + abs(cell[0, 2])
                    + abs(cell[1, 0]) + abs(cell[1, 2])
                    + abs(cell[2, 0]) + abs(cell[2, 1]))
        if off_diag < 1e-10:
            config.cell_type = _core.CellType.Orthogonal
        else:
            config.cell_type = _core.CellType.NonOrthogonal

        # --- Grid ---
        if self._grid_shape is not None:
            config.Nx, config.Ny, config.Nz = self._grid_shape
        elif self._mesh_spacing is not None:
            lengths = np.linalg.norm(cell, axis=1)
            config.Nx = max(int(np.ceil(lengths[0] / self._mesh_spacing)), 1)
            config.Ny = max(int(np.ceil(lengths[1] / self._mesh_spacing)), 1)
            config.Nz = max(int(np.ceil(lengths[2] / self._mesh_spacing)), 1)
        else:
            # Default: ~0.3 Bohr spacing
            lengths = np.linalg.norm(cell, axis=1)
            config.Nx = max(int(np.ceil(lengths[0] / 0.3)), 1)
            config.Ny = max(int(np.ceil(lengths[1] / 0.3)), 1)
            config.Nz = max(int(np.ceil(lengths[2] / 0.3)), 1)

        config.fd_order = self._fd_order
        config.mesh_spacing = self._mesh_spacing or 0.0

        # --- Boundary conditions ---
        bc_map = {
            "periodic": _core.BCType.Periodic,
            "dirichlet": _core.BCType.Dirichlet,
        }
        bc = bc_map.get(self._bc.lower(), _core.BCType.Periodic)
        config.bcx = config.bcy = config.bcz = bc

        # --- Atoms ---
        xc_type_str = (
            self._xc._xc_type
            if hasattr(self._xc, "_xc_type")
            else "GGA_PBE"
        )
        atom_inputs = atoms._to_config_atoms(xc=xc_type_str)
        config.atom_types = atom_inputs

        # --- Electronic structure ---
        config.Nstates = self._n_bands or 0  # 0 = auto

        # Spin
        if self._spin is None:
            config.spin_type = _core.SpinType.NoSpin
        elif self._spin.lower() == "collinear":
            config.spin_type = _core.SpinType.Collinear
        elif self._spin.lower() == "noncollinear":
            config.spin_type = _core.SpinType.NonCollinear
        else:
            raise ValueError(f"Unknown spin type: '{self._spin}'")

        # Auto-enable collinear spin if atoms carry spin moments
        if atoms.spin is not None and self._spin is None:
            if any(abs(s) > 1e-10 for s in atoms.spin):
                config.spin_type = _core.SpinType.Collinear

        config.elec_temp = self._temperature

        if self._smearing == "fermi-dirac":
            config.smearing = _core.SmearingType.FermiDirac
        else:
            config.smearing = _core.SmearingType.GaussianSmearing

        # --- XC functional ---
        config.xc = getattr(_core.XCType, xc_type_str)

        # Hybrid functional parameters (EXX fraction, range separation)
        if self._xc.is_hybrid:
            config.exx_params = _core.EXXParams()
            config.exx_params.exx_frac = self._xc.exx_fraction
            if hasattr(self._xc, "omega"):
                config.exx_params.hyb_range_fock = self._xc.omega

        # --- K-points ---
        config.Kx, config.Ky, config.Kz = self._kpts
        config.kpt_shift = _core.Vec3(*self._kpt_shift)

        # --- SCF parameters ---
        config.max_scf_iter = self._max_scf
        config.scf_tol = self._scf_tol
        config.mixing_param = self._mixing_beta
        config.mixing_history = self._mixing_history
        config.cheb_degree = self._cheb_degree
        config.rho_trigger = self._rho_trigger

        # Mixing variable and preconditioner
        config.mixing_var = _core.MixingVariable.Potential
        config.mixing_precond = _core.MixingPrecond.Kerker

        # --- Output flags ---
        config.print_forces = (self._verbose >= 2)
        config.print_atoms = (self._verbose >= 2)
        config.print_eigen = (self._verbose >= 2)
        config.calc_stress = True
        config.calc_pressure = True

        # --- Density restart ---
        if self._restart_density:
            config.density_restart_file = self._restart_density

        return config

    def __repr__(self):
        return (
            f"DFT(xc={self._xc}, kpts={self._kpts}, "
            f"device='{self._device}')"
        )
