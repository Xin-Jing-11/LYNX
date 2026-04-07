import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List

# Import unit conversion constants
BOHR_TO_ANG = 0.529177249
HA_TO_EV = 27.211386245988
HA_BOHR3_TO_GPA = 29421.01569650548

@dataclass
class EnergyDecomposition:
    """Energy components of a DFT calculation."""
    band: float = 0.0         # Band structure energy
    xc: float = 0.0           # Exchange-correlation
    hartree: float = 0.0      # Hartree (electrostatic)
    self_energy: float = 0.0  # Self energy
    correction: float = 0.0   # Correction energy
    entropy: float = 0.0      # Electronic entropy (-T*S)
    exx: float = 0.0          # Exact exchange (hybrids)
    total: float = 0.0        # Total free energy
    per_atom: float = 0.0     # Energy per atom

    def __repr__(self):
        lines = [f"EnergyDecomposition("]
        lines.append(f"  total    = {self.total:.8f} Ha ({self.total * HA_TO_EV:.6f} eV)")
        lines.append(f"  band     = {self.band:.8f} Ha")
        lines.append(f"  xc       = {self.xc:.8f} Ha")
        lines.append(f"  hartree  = {self.hartree:.8f} Ha")
        if abs(self.exx) > 1e-15:
            lines.append(f"  exx      = {self.exx:.8f} Ha")
        if abs(self.entropy) > 1e-15:
            lines.append(f"  entropy  = {self.entropy:.8f} Ha")
        lines.append(f"  per_atom = {self.per_atom:.8f} Ha")
        lines.append(")")
        return "\n".join(lines)


@dataclass
class DFTResult:
    """Results of a DFT calculation.

    All quantities in atomic units (Hartree, Bohr) unless noted.
    Use convenience methods for eV/Angstrom conversion.
    """

    # Primary outputs
    energy: float = 0.0                          # Total energy (Ha)
    forces: Optional[np.ndarray] = None          # (N, 3) in Ha/Bohr
    stress: Optional[np.ndarray] = None          # (6,) Voigt in Ha/Bohr^3
    pressure: float = 0.0                        # GPa

    # Energy decomposition
    energies: EnergyDecomposition = field(default_factory=EnergyDecomposition)

    # Electronic structure
    eigenvalues: Optional[List[np.ndarray]] = None    # per (spin, kpt)
    occupations: Optional[List[np.ndarray]] = None    # per (spin, kpt)
    fermi_energy: float = 0.0                         # Ha
    density: Optional[np.ndarray] = None              # electron density

    # Convergence info
    converged: bool = False
    n_iterations: int = 0
    scf_history: Optional[List[Dict]] = None   # per-step: {energy, error}

    # Reference to input (set by DFT calculator)
    atoms: object = None        # lynx.Atoms

    # Convenience: unit conversion
    @property
    def energy_eV(self) -> float:
        return self.energy * HA_TO_EV

    @property
    def forces_eV_A(self) -> Optional[np.ndarray]:
        if self.forces is None:
            return None
        return self.forces * (HA_TO_EV / BOHR_TO_ANG)

    @property
    def stress_GPa(self) -> Optional[np.ndarray]:
        if self.stress is None:
            return None
        return self.stress * HA_BOHR3_TO_GPA

    def summary(self) -> str:
        """Pretty-print summary of results."""
        lines = []
        lines.append("=" * 50)
        lines.append("LYNX DFT Calculation Summary")
        lines.append("=" * 50)
        if self.atoms is not None:
            lines.append(f"System:     {self.atoms.formula} ({self.atoms.n_atoms} atoms)")
        lines.append(f"Converged:  {self.converged} ({self.n_iterations} iterations)")
        lines.append(f"Energy:     {self.energy:.8f} Ha ({self.energy_eV:.6f} eV)")
        lines.append(f"Fermi:      {self.fermi_energy:.8f} Ha")
        if self.forces is not None:
            fmax = np.max(np.abs(self.forces))
            lines.append(f"Max force:  {fmax:.6e} Ha/Bohr")
        if self.stress is not None:
            smax = np.max(np.abs(self.stress_GPa))
            lines.append(f"Max stress: {smax:.4f} GPa")
            lines.append(f"Pressure:   {self.pressure:.4f} GPa")
        lines.append("=" * 50)
        return "\n".join(lines)

    def __repr__(self):
        conv = "converged" if self.converged else "NOT converged"
        return f"DFTResult(energy={self.energy:.8f} Ha, {conv}, {self.n_iterations} iter)"
