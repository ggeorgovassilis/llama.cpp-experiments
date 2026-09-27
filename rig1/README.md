# rig1 llama.cpp experiment environment

Fresh, bespoke llama.cpp build environment for the rig1 test server. Builds and
tests everything inside Docker.

## Hardware targets

- GPU: 4x NVIDIA Tesla M10, 8 GiB each. Maxwell, compute capability **5.0** (`sm_50`).
- CPU: 2x Intel Xeon E5-2640 (Sandy Bridge). **AVX only** - no AVX2, FMA or F16C.

## Layout

```
src/                     fresh clone of ggeorgovassilis/llama.cpp-experiments (branch true-pipeline)
docker/Dockerfile.build  build image (nvidia/cuda:12.6.2-devel-ubuntu24.04 + deps)
docker/Dockerfile.runtime runtime image (bundles the built binaries)
scripts/build.sh         configure + build (CUDA sm_50 + AVX) inside Docker
scripts/build-runtime.sh build the runtime image
scripts/test.sh          run fast unit tests inside Docker
scripts/run.sh           smoke test: load a small model on GPU, generate tokens
scripts/llama-server     run llama-server from the runtime image
scripts/llama-cli        run llama-cli from the runtime image
build/                   build output (bind-mounted, persists across runs)
.ccache/                 ccache store (persists across runs)
```

## Build flags

```sh
cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=50 \   # bespoke: only sm_50, not the 9-arch default
    -DGGML_NATIVE=OFF \
    -DGGML_SSE42=ON \
    -DGGML_AVX=ON \                    # Sandy Bridge has AVX only
    -DGGML_AVX2=OFF \                  # must stay OFF: Sandy Bridge has no AVX2
    -DGGML_FMA=OFF \                   # must stay OFF: no FMA
    -DGGML_F16C=OFF \                  # must stay OFF: no F16C
    -DGGML_BMI2=OFF \                  # must stay OFF: no BMI2
    -DGGML_CUDA=ON \
    -DGGML_CUDA_CUB_3DOT2=ON \         # fetch CCCL 3.2 (needed by mean/sum/ssm-scan CUB usage)
    -DLLAMA_CURL=OFF
```

Why CUDA 12.6.2: it still compiles sm_50 (Maxwell). CUDA 13 dropped Maxwell.

Why `GGML_CUDA_CUB_3DOT2=ON`: the CUDA backend now uses CUB (`cub::BlockScan`,
`cub::BlockLoad`) that needs CCCL >= 3.2; CUDA 12.6 bundles an older CCCL, so the
flag fetches v3.2.0 during configure.

Why the AVX2/FMA/F16C/BMI2=OFF: with `GGML_NATIVE=OFF` the build defaults them
ON (they compile to `-mavx2 -mfma -mf16c -mbmi2`), which crashes with SIGILL on
Sandy Bridge. Only SSE4.2 + AVX are safe.

## Usage

```sh
./scripts/build.sh            # build everything (image + configure + build)
./scripts/test.sh             # run fast unit tests (ctest)
./scripts/run.sh              # smoke test with a small model (Qwen3.5-0.8B)
./scripts/run.sh gemma-4-E2B-it-Q6_K.gguf 32   # pick model + token count
./scripts/build-runtime.sh    # build the runtime image (bundles binaries)
./scripts/llama-server -m /models/gemma-4-E2B-it-Q6_K.gguf   # run the server
```

CUDA kernel validation (`test-backend-ops`) is built but not registered as a
ctest; run it manually:

```sh
ssh rig1.local 'docker run --rm --gpus all -v ~/llamacpp-experiments/build:/build llamacpp-exp-build:12.6.2 /build/bin/test-backend-ops -b CUDA0'
```

## Gotchas

- `llama-cli` defaults to interactive chat and loops on the `>` prompt forever
  against a closed stdin. Pass `--single-turn` (run.sh already does) or it
  spins at 100% CPU spamming `> `.

## Runtime image

The runtime image `llamacpp-exp:true-pipeline` bundles the built binaries so the
branch can run without the build toolchain. See `docs/experiments/runtime-image.md`.

```sh
./scripts/build-runtime.sh    # build it (runs build.sh first if binaries are missing)
./scripts/llama-server -m /models/gemma-4-E2B-it-Q6_K.gguf
./scripts/llama-cli -m /models/Qwen3.5-0.8B-Q8_0.gguf -p "hello" -n 8 --single-turn
```

## Notes

- Models are read-only mounted from `/mnt/ssd2/models` (the shared store).
- `~/llamacpp` (the `rig1` deployment repo) is separate and must not be touched.
- `src/` is a clean clone; re-clone or `git pull` to update, never edit in place
  unless you intend to commit there.
