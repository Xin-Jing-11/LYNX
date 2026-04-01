#pragma once

#include "solvers/Preconditioner.hpp"
#include "operators/Laplacian.hpp"
#include "parallel/HaloExchange.hpp"
#include "core/FDGrid.hpp"

namespace lynx {

// Kerker preconditioner: solves -(Lap - kTF^2)*Pf = (Lap - idiemac*kTF^2)*f
// then Pf *= -mixing_param.
// Matches SPARC Kerker_precond exactly.
class KerkerPreconditioner : public Preconditioner {
public:
    KerkerPreconditioner(const Laplacian* lap, const HaloExchange* halo,
                         const FDGrid* grid, double kTF = 1.0, double idiemac = 0.1,
                         double precond_tol = -1.0);

    void apply(const double* r, double* z, int N, double mixing_param) override;

private:
    const Laplacian* laplacian_;
    const HaloExchange* halo_;
    const FDGrid* grid_;
    double kTF_;
    double idiemac_;
    double precond_tol_;
};

} // namespace lynx
