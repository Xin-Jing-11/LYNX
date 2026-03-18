"""
Example 7: Plug in a custom eigensolver via the ABC interface.

Shows how to replace CheFSI with SciPy's LOBPCG eigensolver while
reusing all other LYNX components (Poisson, XC, mixer, etc.).

This is the key use-case for the modular framework: algorithm research
on one component without touching the rest of the DFT pipeline.
"""
import numpy as np
import lynx
from lynx.abc import EigenSolver, HamiltonianOperator

lynx.init()


# ======================================================================
# Custom eigensolver: wraps scipy.sparse.linalg.lobpcg
# ======================================================================
class LOBPCGSolver(EigenSolver):
    """LOBPCG eigensolver that calls H.apply() through the ABC interface.

    This solver works with ANY HamiltonianOperator implementation —
    FD, plane-wave, or anything else that implements apply().
    """

    def __init__(self, dV: float, tol: float = 1e-6, max_iter: int = 200):
        """
        Args:
            dV: Volume element for inner products (grid spacing product).
            tol: LOBPCG convergence tolerance.
            max_iter: Maximum LOBPCG iterations.
        """
        self.dV = dV
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, H: HamiltonianOperator, psi, Veff, **kw):
        from scipy.sparse.linalg import lobpcg, LinearOperator

        Nd_d = H.ndof
        Nband = psi.shape[1]

        # Wrap H.apply as a scipy LinearOperator
        # LOBPCG passes column vectors; we reshape for H.apply
        def matvec(x):
            # x is (Nd_d,) — single vector
            x2d = x.reshape(Nd_d, 1)
            Hx = H.apply(x2d, Veff)
            return Hx.ravel()

        A = LinearOperator((Nd_d, Nd_d), matvec=matvec, dtype=np.float64)

        # LOBPCG needs an SPD inner-product matrix B for the generalized
        # eigenproblem.  For uniform grids, B = dV * I  (mass matrix).
        # We encode this as a preconditioner on the residual instead.
        # With B=None, LOBPCG uses the standard dot product, which is
        # equivalent to the dV-weighted inner product up to a constant.

        # Initial guess: use current psi (must be C-contiguous for LOBPCG)
        X0 = np.ascontiguousarray(psi)

        try:
            eigenvalues, eigenvectors = lobpcg(
                A, X0,
                largest=False,
                tol=self.tol,
                maxiter=self.max_iter,
                verbosityLevel=0,
            )
        except Exception as e:
            # LOBPCG can fail to converge; fall back to input
            print(f"  LOBPCG warning: {e}")
            eigenvalues = np.zeros(Nband)
            eigenvectors = psi.copy()

        # Return in Fortran order (column-major) to match CheFSI convention
        psi_out = np.asfortranarray(eigenvectors)
        return psi_out, eigenvalues


# ======================================================================
# Another custom eigensolver: power-iteration style (pedagogical)
# ======================================================================
class SubspaceIterationSolver(EigenSolver):
    """Simple subspace iteration — pedagogical, not production-quality.

    Applies H repeatedly to a subspace, then diagonalizes the projected
    Hamiltonian.  Very slow convergence but demonstrates the ABC interface.
    """

    def __init__(self, dV: float, n_inner: int = 5):
        self.dV = dV
        self.n_inner = n_inner

    def solve(self, H: HamiltonianOperator, psi, Veff, **kw):
        Nband = psi.shape[1]
        X = psi.copy()

        for _ in range(self.n_inner):
            # Apply H to all columns
            HX = H.apply(X, Veff)

            # QR orthogonalization
            Q, _ = np.linalg.qr(HX, mode="reduced")
            X = Q

        # Rayleigh-Ritz: project H onto subspace
        HX = H.apply(X, Veff)
        H_sub = X.T @ HX * self.dV       # (Nband, Nband)
        S_sub = X.T @ X * self.dV         # overlap
        from scipy.linalg import eigh as scipy_eigh
        eigvals, C = scipy_eigh(H_sub, S_sub)

        # Rotate subspace
        psi_out = X @ C
        return np.asfortranarray(psi_out), eigvals


# ======================================================================
# Run the examples
# ======================================================================
if __name__ == "__main__":
    import os

    psp = lynx.find_psp("Si", xc="PBE")
    if psp is None:
        raise FileNotFoundError("Si PBE pseudopotential not found")

    config = lynx.DFTConfig(
        cell=[[10.26, 0, 0], [0, 10.26, 0], [0, 0, 10.26]],
        fractional=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        symbols=["Si", "Si"],
        pseudo_files={"Si": psp},
        Nx=20, Ny=20, Nz=20,
        Nstates=14,
        xc="GGA_PBE",
    )

    system = lynx.SystemSetup(config)
    ops = system.default_operators(xc_type=lynx.XCType.GGA_PBE)
    params = system.scf_params()
    dV = system.dV

    psi_init = system.random_wavefunctions()
    Nd_d = system.Nd_d
    rho_init = np.full(Nd_d, params["Nelectron"] / (Nd_d * dV))

    # ---- A. Default CheFSI (baseline) ----
    print("=" * 60)
    print("A. Default CheFSI eigensolver")
    print("=" * 60)

    driver_chefsi = lynx.SCFDriver(
        **{k: v for k, v in ops.items() if k not in ("kinetic", "gradient")},
        **params,
        max_iter=30, tol=1e-3, rho_trigger=4,
        on_iteration=lambda it, err, E: print(f"  iter {it:3d}: err={err:.3e} E={E:.4f}"),
    )
    result_a = driver_chefsi.run(psi_init, rho_init)
    print(f"  => Converged in {result_a.n_iterations} iters, E={result_a.total_energy:.6f} Ha\n")

    # ---- B. LOBPCG eigensolver ----
    print("=" * 60)
    print("B. SciPy LOBPCG eigensolver (drop-in replacement)")
    print("=" * 60)

    lobpcg_solver = LOBPCGSolver(dV=dV, tol=1e-4, max_iter=100)

    driver_lobpcg = lynx.SCFDriver(
        hamiltonian=ops["hamiltonian"],
        eigensolver=lobpcg_solver,        # <-- swapped!
        poisson=ops["poisson"],
        xc=ops["xc"],
        mixer=ops["mixer"],
        occupation=ops["occupation"],
        **params,
        max_iter=30, tol=1e-3, rho_trigger=1,  # LOBPCG doesn't need warmup passes
        on_iteration=lambda it, err, E: print(f"  iter {it:3d}: err={err:.3e} E={E:.4f}"),
    )
    result_b = driver_lobpcg.run(psi_init, rho_init)
    print(f"  => Converged in {result_b.n_iterations} iters, E={result_b.total_energy:.6f} Ha\n")

    # ---- C. Subspace iteration eigensolver ----
    print("=" * 60)
    print("C. Simple subspace iteration (pedagogical)")
    print("=" * 60)

    subspace_solver = SubspaceIterationSolver(dV=dV, n_inner=10)

    driver_sub = lynx.SCFDriver(
        hamiltonian=ops["hamiltonian"],
        eigensolver=subspace_solver,      # <-- swapped!
        poisson=ops["poisson"],
        xc=ops["xc"],
        mixer=ops["mixer"],
        occupation=ops["occupation"],
        **params,
        max_iter=30, tol=1e-3, rho_trigger=1,
        on_iteration=lambda it, err, E: print(f"  iter {it:3d}: err={err:.3e} E={E:.4f}"),
    )
    result_c = driver_sub.run(psi_init, rho_init)
    print(f"  => Converged={result_c.converged} in {result_c.n_iterations} iters, "
          f"E={result_c.total_energy:.6f} Ha\n")

    # ---- Summary ----
    print("=" * 60)
    print("Summary: same DFT system, three different eigensolvers")
    print("=" * 60)
    print(f"  CheFSI (C++):       {result_a.n_iterations:3d} iters, E = {result_a.total_energy:.6f} Ha")
    print(f"  LOBPCG (SciPy):     {result_b.n_iterations:3d} iters, E = {result_b.total_energy:.6f} Ha")
    print(f"  Subspace iteration: {result_c.n_iterations:3d} iters, E = {result_c.total_energy:.6f} Ha")
