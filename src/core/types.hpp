#pragma once

#include <complex>
#include <cstddef>
#include <cstring>
#include <cmath>

namespace lynx {

using Real = double;
using Complex = std::complex<double>;

enum class CellType { Orthogonal = 0, NonOrthogonal = 1, Helical = 2 };
enum class BCType { Periodic = 0, Dirichlet = 1 };
enum class SpinType { None = 0, Collinear = 1, NonCollinear = 2 };
enum class MixingVariable { Density = 0, Potential = 1 };
enum class MixingPrecond { None = 0, Kerker = 1 };
enum class PoissonSolverType { AAR = 0, CG = 1 };
enum class SmearingType { GaussianSmearing = 0, FermiDirac = 1 };
enum class XCType { LDA_PZ, LDA_PW, GGA_PBE, GGA_PBEsol, GGA_RPBE };

// 3x3 matrix — row-major, trivially copyable (CUDA-friendly)
struct Mat3 {
    double data[9] = {};

    double& operator()(int i, int j) { return data[i * 3 + j]; }
    const double& operator()(int i, int j) const { return data[i * 3 + j]; }

    double determinant() const {
        return data[0] * (data[4] * data[8] - data[5] * data[7])
             - data[1] * (data[3] * data[8] - data[5] * data[6])
             + data[2] * (data[3] * data[7] - data[4] * data[6]);
    }

    Mat3 inverse() const {
        double det = determinant();
        Mat3 inv;
        inv.data[0] = (data[4] * data[8] - data[5] * data[7]) / det;
        inv.data[1] = (data[2] * data[7] - data[1] * data[8]) / det;
        inv.data[2] = (data[1] * data[5] - data[2] * data[4]) / det;
        inv.data[3] = (data[5] * data[6] - data[3] * data[8]) / det;
        inv.data[4] = (data[0] * data[8] - data[2] * data[6]) / det;
        inv.data[5] = (data[2] * data[3] - data[0] * data[5]) / det;
        inv.data[6] = (data[3] * data[7] - data[4] * data[6]) / det;
        inv.data[7] = (data[1] * data[6] - data[0] * data[7]) / det;
        inv.data[8] = (data[0] * data[4] - data[1] * data[3]) / det;
        return inv;
    }

    Mat3 transpose() const {
        Mat3 t;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                t(i, j) = (*this)(j, i);
        return t;
    }

    // Matrix-matrix multiply: C = this * B
    Mat3 operator*(const Mat3& B) const {
        Mat3 C;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double sum = 0.0;
                for (int k = 0; k < 3; ++k)
                    sum += (*this)(i, k) * B(k, j);
                C(i, j) = sum;
            }
        return C;
    }
};

struct Vec3 {
    double x = 0, y = 0, z = 0;

    Vec3() = default;
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    double norm() const { return std::sqrt(x * x + y * y + z * z); }
};

// Memory alignment for SIMD/CUDA
constexpr std::size_t MEMORY_ALIGNMENT = 64;

} // namespace lynx
