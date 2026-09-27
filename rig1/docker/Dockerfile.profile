# Profiling image for rig1 llama.cpp experiments.
# Extends the build image with Nsight Systems (nsys) and Nsight Compute (ncu)
# so we can capture per-GPU kernel timelines and occupancy metrics on the M10.
# Base matches the build image so the binaries stay compatible.
FROM llamacpp-exp-build:12.6.2

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-nsight-systems-12-6 \
        cuda-nsight-compute-12-6 \
        libnss3 \
        libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# nsys/ncu ship under /usr/local/cuda-12.6/bin (nsight-compute) and
# /opt/nvidia/nsight-systems (nsys). Put both on PATH.
ENV PATH="/opt/nvidia/nsight-systems/2024.4.2/target-linux-x64:/usr/local/cuda-12.6/bin:${PATH}"

WORKDIR /app
