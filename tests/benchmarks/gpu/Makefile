NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -O3 -std=c++17 --use_fast_math
CUDA_ARCH = -arch=sm_120
LIBS = -lcurand

BENCHMARKS = bench_laplacian bench_nonlocal bench_chebyshev bench_scan_xc bench_gradient bench_xc_lda_gga

all: $(BENCHMARKS)

bench_%: bench_%.cu
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) -o $@ $< $(LIBS)

clean:
	rm -f $(BENCHMARKS)

.PHONY: all clean
