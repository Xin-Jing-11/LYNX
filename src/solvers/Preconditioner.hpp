#pragma once

namespace lynx {

// Abstract preconditioner interface for mixing.
// Transforms a residual vector: z = P(r), scaled by mixing_param.
class Preconditioner {
public:
    virtual ~Preconditioner() = default;

    // Apply preconditioner: z = P(r), where r is residual of length N.
    // mixing_param is the Pulay/simple mixing coefficient (beta).
    virtual void apply(const double* r, double* z, int N, double mixing_param) = 0;
};

// Identity preconditioner: z = mixing_param * r
class IdentityPreconditioner : public Preconditioner {
public:
    void apply(const double* r, double* z, int N, double mixing_param) override {
        for (int i = 0; i < N; ++i)
            z[i] = mixing_param * r[i];
    }
};

} // namespace lynx
