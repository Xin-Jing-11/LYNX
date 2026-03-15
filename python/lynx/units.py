"""
Unit conversion constants for LYNX <-> ASE interop.

LYNX internal units: Bohr (length), Hartree (energy), amu (mass)
ASE internal units:  Angstrom (length), eV (energy), amu (mass)
"""

# Length
BOHR_TO_ANG = 0.5291772105638411  # 1 Bohr in Angstrom (CODATA 2014)
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG  # 1 Angstrom in Bohr

# Energy
HA_TO_EV = 27.211386024367243  # 1 Hartree in eV (CODATA 2014)
EV_TO_HA = 1.0 / HA_TO_EV      # 1 eV in Hartree

# Force: Ha/Bohr -> eV/Ang
HA_BOHR_TO_EV_ANG = HA_TO_EV / BOHR_TO_ANG

# Stress: Ha/Bohr^3 -> eV/Ang^3
HA_BOHR3_TO_EV_ANG3 = HA_TO_EV / BOHR_TO_ANG**3

# Stress: Ha/Bohr^3 -> GPa
HA_BOHR3_TO_GPA = 29421.01569650548
