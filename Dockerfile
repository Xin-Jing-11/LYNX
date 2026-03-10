FROM ubuntu:24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential cmake g++ git \
    libopenmpi-dev openmpi-bin \
    libopenblas-dev libscalapack-openmpi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# --- Build stage ---
FROM base AS build

COPY CMakeLists.txt .
COPY cmake/ cmake/
COPY src/ src/
COPY tests/ tests/
COPY examples/ examples/

RUN cmake -B build -DBUILD_TESTS=ON -DUSE_MKL=OFF \
    && cmake --build build -j$(nproc)

# --- Test stage (default) ---
FROM build AS test
CMD ["ctest", "--test-dir", "build", "--output-on-failure"]

# --- Runtime stage (minimal) ---
FROM base AS runtime
COPY --from=build /workspace/build/src/sparc /usr/local/bin/sparc
COPY examples/ /workspace/examples/
ENTRYPOINT ["sparc"]
