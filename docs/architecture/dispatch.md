---
title: CPU/GPU Dispatch
parent: Architecture
nav_order: 1
---

# CPU/GPU Dispatch

LYNX follows the same dispatch pattern as PyTorch and TensorFlow: **one algorithm, kernel-level dispatch.** The iteration loop, convergence check, and sub-steps are written once in `.cpp`. GPU-specific code is isolated to `.cu` files.

## The Pattern

Each operator declares three layers in its header:

```cpp
// Hamiltonian.hpp — three layers

// 1. Public method: the only entry point callers use
void apply(const double* psi, const double* Veff,
           double* y, int ncol, double c) const;

// 2. CPU implementation (in Hamiltonian.cpp)
void apply_cpu(const double* psi, const double* Veff,
               double* y, int ncol, double c) const;

// 3. GPU implementation (in Hamiltonian.cu — kernel launches only)
void apply_gpu(const double* psi, const double* Veff,
               double* y, int ncol, double c) const;
```

The dispatcher in `.cpp` is three lines:

```cpp
// src/operators/Hamiltonian.cpp
void Hamiltonian::apply(const double* psi, const double* Veff,
                        double* y, int ncol, double c) const {
#ifdef USE_CUDA
    if (dev_ == Device::GPU) { apply_gpu(psi, Veff, y, ncol, c); return; }
#endif
    apply_cpu(psi, Veff, y, ncol, c);
}
```

`dev_` is a member set once at construction or `setup()`. Public methods never take a `Device` parameter — they check `dev_` internally.

## What Lives Where

| File | Contains |
|------|---------|
| `Operator.hpp` | Declares `apply()`, `apply_cpu()`, `apply_gpu()` |
| `Operator.cpp` | Implements `apply()` (dispatcher) and `apply_cpu()` |
| `Operator.cu` | Implements `apply_gpu()` — kernel launches, cuBLAS/cuSOLVER calls only |

**No loops in `.cu`.** No algorithm logic. If a sub-operation needs context, it is a method on `this`, not a standalone GPU function.

## Analogy to PyTorch

PyTorch uses the same principle via `at::Tensor` dispatch keys: the user calls `torch::mm(a, b)` and the framework routes to a CPU or CUDA kernel. The calling code never changes based on device. LYNX applies the same idea at the operator level — callers always call `hamiltonian.apply(...)`, and the device choice is encapsulated inside.

## GPU Data Residency

The dispatch pattern pairs with a data residency rule: **wavefunctions never leave the GPU.** On GPU builds, `psi` is allocated and randomized on-device (cuRAND), lives there for the entire SCF loop, and is used directly in GPU force and stress kernels. No download, no re-upload.

Per-iteration allowed transfers (tiny, scalar):

| Direction | Data | Size |
|-----------|------|------|
| D→H | eigenvalues | ~KB |
| H→D | occupations | ~KB |
| D→H | rho, Veff (for energy on CPU) | once/iter |

Everything else — `psi`, `Veff`, `rho`, `phi`, `exc` — stays on device.

## Why This Matters

- **Testability:** the CPU path is always available, so algorithms can be validated on CPU before GPU implementation.
- **Auditability:** GPU code is isolated to `.cu` files — a CUDA expert can review the kernel implementations without reading algorithm logic.
- **Extensibility:** adding a new execution backend (e.g., HIP/ROCm) requires only new `_hip()` methods and a new dispatch branch — no algorithm code changes.
