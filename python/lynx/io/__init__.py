"""File I/O and serialization — like torch.save/torch.load.

Usage:
    lynx.save(result, "si2.lynx")
    result = lynx.load("si2.lynx")

    atoms = lynx.io.read("POSCAR")
    lynx.io.write(atoms, "output.xyz")
"""

import pickle
import numpy as np
from pathlib import Path


def save(obj, path):
    """Save a LYNX object (DFTResult, density, etc.) to file.

    Uses numpy for arrays, pickle for complex objects.

    Args:
        obj: DFTResult, numpy array, or any picklable object
        path: output file path (.lynx, .npy, .npz)
    """
    path = Path(path)

    if isinstance(obj, np.ndarray):
        np.save(path, obj)
    elif path.suffix == '.npz':
        # Save DFTResult as npz
        data = _result_to_dict(obj) if hasattr(obj, 'energy') else {'data': obj}
        np.savez(path, **data)
    else:
        with open(path, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(path):
    """Load a LYNX object from file.

    Args:
        path: input file path

    Returns:
        Loaded object (DFTResult, array, etc.)
    """
    path = Path(path)

    if path.suffix == '.npy':
        return np.load(path)
    elif path.suffix == '.npz':
        return dict(np.load(path, allow_pickle=True))
    else:
        with open(path, 'rb') as f:
            return pickle.load(f)


def read(filename, format=None, **kwargs):
    """Read atomic structure from file.

    Supports: LYNX JSON, POSCAR, XYZ, CIF (via ASE).

    Args:
        filename: path to structure file
        format: file format (auto-detected if None)

    Returns:
        lynx.Atoms instance
    """
    from lynx.atoms import Atoms
    return Atoms.read(filename, format=format, **kwargs)


def write(atoms, filename, format=None):
    """Write atomic structure to file.

    Args:
        atoms: lynx.Atoms instance
        filename: output file path
        format: file format (auto-detected from extension if None)
    """
    atoms.write(filename, format=format)


def _result_to_dict(result):
    """Convert DFTResult to dict for npz serialization."""
    d = {
        'energy': np.array(result.energy),
        'fermi_energy': np.array(result.fermi_energy),
        'converged': np.array(result.converged),
        'n_iterations': np.array(result.n_iterations),
        'pressure': np.array(result.pressure),
    }
    if result.forces is not None:
        d['forces'] = result.forces
    if result.stress is not None:
        d['stress'] = result.stress
    if result.density is not None:
        d['density'] = result.density
    return d
