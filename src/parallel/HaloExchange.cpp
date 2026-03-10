#include "parallel/HaloExchange.hpp"
#include <cstring>
#include <vector>

namespace sparc {

HaloExchange::HaloExchange(const Domain& domain, int FDn, MPI_Comm cart_comm)
    : nx_(domain.Nx_d()), ny_(domain.Ny_d()), nz_(domain.Nz_d()),
      FDn_(FDn), cart_comm_(cart_comm) {
    nx_ex_ = nx_ + 2 * FDn;
    ny_ex_ = ny_ + 2 * FDn;
    nz_ex_ = nz_ + 2 * FDn;

    if (cart_comm_ != MPI_COMM_NULL) {
        MPI_Cart_get(cart_comm_, 3, dims_, reinterpret_cast<int*>(periods_), nullptr);
        // Fix: MPI_Cart_get uses int for periods
        int int_periods[3];
        int coords[3];
        MPI_Cart_get(cart_comm_, 3, dims_, int_periods, coords);
        periods_[0] = int_periods[0]; periods_[1] = int_periods[1]; periods_[2] = int_periods[2];

        MPI_Cart_shift(cart_comm_, 0, 1, &rank_xm_, &rank_xp_);
        MPI_Cart_shift(cart_comm_, 1, 1, &rank_ym_, &rank_yp_);
        MPI_Cart_shift(cart_comm_, 2, 1, &rank_zm_, &rank_zp_);
    } else {
        dims_[0] = dims_[1] = dims_[2] = 1;
    }
}

void HaloExchange::copy_to_interior(const double* x, double* x_ex, int ncol) const {
    for (int n = 0; n < ncol; ++n) {
        const double* xn = x + n * static_cast<std::ptrdiff_t>(nx_ * ny_ * nz_);
        double* xn_ex = x_ex + n * static_cast<std::ptrdiff_t>(nx_ex_ * ny_ex_ * nz_ex_);
        for (int k = 0; k < nz_; ++k) {
            for (int j = 0; j < ny_; ++j) {
                const double* src = xn + j * nx_ + k * nx_ * ny_;
                double* dst = xn_ex + (j + FDn_) * nx_ex_ + (k + FDn_) * nx_ex_ * ny_ex_ + FDn_;
                std::memcpy(dst, src, nx_ * sizeof(double));
            }
        }
    }
}

void HaloExchange::execute(const double* x, double* x_ex, int ncol) const {
    int nd_ex_total = nx_ex_ * ny_ex_ * nz_ex_;
    // Zero the extended array
    std::memset(x_ex, 0, ncol * nd_ex_total * sizeof(double));

    // Copy local data to interior
    copy_to_interior(x, x_ex, ncol);

    int nproc = dims_[0] * dims_[1] * dims_[2];
    if (nproc > 1) {
        exchange_faces(x_ex, ncol);
    } else {
        // Single process: apply periodic BCs by wrapping
        apply_periodic_bc(x_ex, ncol);
    }
}

void HaloExchange::exchange_faces(double* x_ex, int ncol) const {
    // Exchange in each of 3 directions using non-blocking sends/recvs
    // For each direction, send FDn slabs to neighbors, receive FDn slabs into ghost zones

    int nxny_ex = nx_ex_ * ny_ex_;

    for (int n = 0; n < ncol; ++n) {
        double* xn = x_ex + n * static_cast<std::ptrdiff_t>(nx_ex_ * ny_ex_ * nz_ex_);

        // --- Z direction (contiguous slabs) ---
        {
            int slab_size = nx_ex_ * ny_ex_ * FDn_;
            std::vector<double> send_lo(slab_size), send_hi(slab_size);
            std::vector<double> recv_lo(slab_size), recv_hi(slab_size);

            // Pack: send low slab (k = FDn..2*FDn-1) to rank_zm
            for (int k = 0; k < FDn_; ++k)
                std::memcpy(send_lo.data() + k * nx_ex_ * ny_ex_,
                           xn + (k + FDn_) * nxny_ex, nx_ex_ * ny_ex_ * sizeof(double));
            // Pack: send high slab (k = nz..nz+FDn-1) to rank_zp
            for (int k = 0; k < FDn_; ++k)
                std::memcpy(send_hi.data() + k * nx_ex_ * ny_ex_,
                           xn + (nz_ + k) * nxny_ex, nx_ex_ * ny_ex_ * sizeof(double));

            MPI_Request reqs[4];
            MPI_Isend(send_lo.data(), slab_size, MPI_DOUBLE, rank_zm_, 0, cart_comm_, &reqs[0]);
            MPI_Isend(send_hi.data(), slab_size, MPI_DOUBLE, rank_zp_, 1, cart_comm_, &reqs[1]);
            MPI_Irecv(recv_lo.data(), slab_size, MPI_DOUBLE, rank_zm_, 1, cart_comm_, &reqs[2]);
            MPI_Irecv(recv_hi.data(), slab_size, MPI_DOUBLE, rank_zp_, 0, cart_comm_, &reqs[3]);
            MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

            // Unpack: recv from zm goes to k = 0..FDn-1
            for (int k = 0; k < FDn_; ++k)
                std::memcpy(xn + k * nxny_ex,
                           recv_lo.data() + k * nx_ex_ * ny_ex_, nx_ex_ * ny_ex_ * sizeof(double));
            // Unpack: recv from zp goes to k = nz+FDn..nz+2*FDn-1
            for (int k = 0; k < FDn_; ++k)
                std::memcpy(xn + (nz_ + FDn_ + k) * nxny_ex,
                           recv_hi.data() + k * nx_ex_ * ny_ex_, nx_ex_ * ny_ex_ * sizeof(double));
        }

        // --- Y direction (strided slabs) ---
        {
            int slab_size = nx_ex_ * FDn_ * nz_ex_;
            std::vector<double> send_lo(slab_size), send_hi(slab_size);
            std::vector<double> recv_lo(slab_size, 0.0), recv_hi(slab_size, 0.0);

            int idx = 0;
            for (int k = 0; k < nz_ex_; ++k)
                for (int j = 0; j < FDn_; ++j) {
                    std::memcpy(send_lo.data() + idx, xn + (j + FDn_) * nx_ex_ + k * nxny_ex, nx_ex_ * sizeof(double));
                    idx += nx_ex_;
                }
            idx = 0;
            for (int k = 0; k < nz_ex_; ++k)
                for (int j = 0; j < FDn_; ++j) {
                    std::memcpy(send_hi.data() + idx, xn + (ny_ + j) * nx_ex_ + k * nxny_ex, nx_ex_ * sizeof(double));
                    idx += nx_ex_;
                }

            MPI_Request reqs[4];
            MPI_Isend(send_lo.data(), slab_size, MPI_DOUBLE, rank_ym_, 2, cart_comm_, &reqs[0]);
            MPI_Isend(send_hi.data(), slab_size, MPI_DOUBLE, rank_yp_, 3, cart_comm_, &reqs[1]);
            MPI_Irecv(recv_lo.data(), slab_size, MPI_DOUBLE, rank_ym_, 3, cart_comm_, &reqs[2]);
            MPI_Irecv(recv_hi.data(), slab_size, MPI_DOUBLE, rank_yp_, 2, cart_comm_, &reqs[3]);
            MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

            idx = 0;
            for (int k = 0; k < nz_ex_; ++k)
                for (int j = 0; j < FDn_; ++j) {
                    std::memcpy(xn + j * nx_ex_ + k * nxny_ex, recv_lo.data() + idx, nx_ex_ * sizeof(double));
                    idx += nx_ex_;
                }
            idx = 0;
            for (int k = 0; k < nz_ex_; ++k)
                for (int j = 0; j < FDn_; ++j) {
                    std::memcpy(xn + (ny_ + FDn_ + j) * nx_ex_ + k * nxny_ex, recv_hi.data() + idx, nx_ex_ * sizeof(double));
                    idx += nx_ex_;
                }
        }

        // --- X direction (most strided) ---
        {
            int slab_size = FDn_ * ny_ex_ * nz_ex_;
            std::vector<double> send_lo(slab_size), send_hi(slab_size);
            std::vector<double> recv_lo(slab_size, 0.0), recv_hi(slab_size, 0.0);

            int idx = 0;
            for (int k = 0; k < nz_ex_; ++k)
                for (int j = 0; j < ny_ex_; ++j) {
                    for (int i = 0; i < FDn_; ++i)
                        send_lo[idx++] = xn[(i + FDn_) + j * nx_ex_ + k * nxny_ex];
                }
            idx = 0;
            for (int k = 0; k < nz_ex_; ++k)
                for (int j = 0; j < ny_ex_; ++j) {
                    for (int i = 0; i < FDn_; ++i)
                        send_hi[idx++] = xn[(nx_ + i) + j * nx_ex_ + k * nxny_ex];
                }

            MPI_Request reqs[4];
            MPI_Isend(send_lo.data(), slab_size, MPI_DOUBLE, rank_xm_, 4, cart_comm_, &reqs[0]);
            MPI_Isend(send_hi.data(), slab_size, MPI_DOUBLE, rank_xp_, 5, cart_comm_, &reqs[1]);
            MPI_Irecv(recv_lo.data(), slab_size, MPI_DOUBLE, rank_xm_, 5, cart_comm_, &reqs[2]);
            MPI_Irecv(recv_hi.data(), slab_size, MPI_DOUBLE, rank_xp_, 4, cart_comm_, &reqs[3]);
            MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

            idx = 0;
            for (int k = 0; k < nz_ex_; ++k)
                for (int j = 0; j < ny_ex_; ++j) {
                    for (int i = 0; i < FDn_; ++i)
                        xn[i + j * nx_ex_ + k * nxny_ex] = recv_lo[idx++];
                }
            idx = 0;
            for (int k = 0; k < nz_ex_; ++k)
                for (int j = 0; j < ny_ex_; ++j) {
                    for (int i = 0; i < FDn_; ++i)
                        xn[(nx_ + FDn_ + i) + j * nx_ex_ + k * nxny_ex] = recv_hi[idx++];
                }
        }
    }
}

void HaloExchange::apply_periodic_bc(double* x_ex, int ncol) const {
    int nxny_ex = nx_ex_ * ny_ex_;

    for (int n = 0; n < ncol; ++n) {
        double* xn = x_ex + n * static_cast<std::ptrdiff_t>(nx_ex_ * ny_ex_ * nz_ex_);

        // Z-direction wrapping
        if (periods_[2]) {
            for (int k = 0; k < FDn_; ++k) {
                // Low ghost from high interior
                std::memcpy(xn + k * nxny_ex,
                           xn + (nz_ + k) * nxny_ex, nxny_ex * sizeof(double));
                // High ghost from low interior
                std::memcpy(xn + (nz_ + FDn_ + k) * nxny_ex,
                           xn + (FDn_ + k) * nxny_ex, nxny_ex * sizeof(double));
            }
        }

        // Y-direction wrapping
        if (periods_[1]) {
            for (int k = 0; k < nz_ex_; ++k) {
                for (int j = 0; j < FDn_; ++j) {
                    std::memcpy(xn + j * nx_ex_ + k * nxny_ex,
                               xn + (ny_ + j) * nx_ex_ + k * nxny_ex, nx_ex_ * sizeof(double));
                    std::memcpy(xn + (ny_ + FDn_ + j) * nx_ex_ + k * nxny_ex,
                               xn + (FDn_ + j) * nx_ex_ + k * nxny_ex, nx_ex_ * sizeof(double));
                }
            }
        }

        // X-direction wrapping
        if (periods_[0]) {
            for (int k = 0; k < nz_ex_; ++k) {
                for (int j = 0; j < ny_ex_; ++j) {
                    int base = j * nx_ex_ + k * nxny_ex;
                    for (int i = 0; i < FDn_; ++i) {
                        xn[base + i] = xn[base + nx_ + i];
                        xn[base + nx_ + FDn_ + i] = xn[base + FDn_ + i];
                    }
                }
            }
        }
    }
}

} // namespace sparc
