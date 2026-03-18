"""Post-SCF property wrappers: forces and stress.

Thin wrappers that delegate to _core.Calculator's compute_forces() and
compute_stress(). These require the C++ Calculator object (from SystemSetup)
because the force/stress computation needs many internal C++ objects.
"""

import numpy as np


def compute_forces(system_setup):
    """Compute atomic forces after SCF convergence.

    Requires that the system's Calculator has completed an SCF run
    (either via Calculator.run() or by running the C++ SCF internally).

    Args:
        system_setup: A SystemSetup instance whose calculator has been run.

    Returns:
        numpy array of shape (Natom, 3) in Hartree/Bohr.
    """
    return system_setup.calculator.compute_forces()


def compute_stress(system_setup):
    """Compute stress tensor after SCF convergence.

    Args:
        system_setup: A SystemSetup instance whose calculator has been run.

    Returns:
        numpy array of shape (6,) in Voigt notation (xx, xy, xz, yy, yz, zz)
        in Hartree/Bohr^3.
    """
    return system_setup.calculator.compute_stress()
