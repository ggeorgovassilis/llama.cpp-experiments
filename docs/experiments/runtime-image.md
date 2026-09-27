# Runtime docker image

A runnable image that bundles the fork's binaries so the `true-pipeline` branch
can be deployed without the build toolchain. Contrast with the dev image
(`llamacpp-exp-build:12.6.2`), which only compiles and leaves the binaries in the
bind-mounted `build/` dir.

## Image

- Tags: `llamacpp-exp:true-pipeline` (alias `llamacpp-exp:12.6.2-true-pipeline`)
- Base: `nvidia/cuda:12.6.2-runtime-ubuntu24.04` (runtime, not devel)
- Contents: `llama-server`, `llama-cli`, `llama-pipeline`, `llama-chain` plus the
  `libggml*` / `libllama*` / `libmtmd` shared libs, all under `/app/bin`
- Extra apt packages: `libgomp1` (OpenMP runtime for `libggml-cpu.so`), `libnccl2`
  (NCCL for `libggml-cuda.so`)

The binaries link their own `.so` files via `RUNPATH=/build/bin`, so the image
copies them to `/app/bin` and sets `LD_LIBRARY_PATH=/app/bin`.

## Build

```sh
ssh rig1.local 'cd ~/llamacpp-experiments && ./scripts/build-runtime.sh'
```

The script builds the binaries first if `build/bin/llama-server` is missing (it
runs `build.sh`), then stages the four binaries and the shared libs into the
docker context and tags the image.

## Run

The wrappers `scripts/llama-server` and `scripts/llama-cli` run the image with the
GPU/shm/memlock flags, mount the shared model store at `/models`, and forward all
arguments to the binary.

Server:

```sh
./scripts/llama-server -m /models/gemma-4-E2B-it-Q6_K.gguf

# or, with a models preset (INI):
MODELS_INI="$HOME/llamacpp/models.ini" ./scripts/llama-server --models-preset /models.ini
```

CLI:

```sh
./scripts/llama-cli -m /models/Qwen3.5-0.8B-Q8_0.gguf -p "hello" -n 8 \
    --single-turn --no-display-prompt
```

Drivers run directly against the image (they are not server/cli):

```sh
docker run --rm --gpus all -v /mnt/ssd2/models:/models:ro \
    llamacpp-exp:true-pipeline \
    llama-pipeline -m /models/Qwen3.5-9B-Q4_K_M.gguf -sm layer -ngl all -ts 1,1,1,1 -k 4
```

## Wrapper env

- `IMAGE` - runtime image (default `llamacpp-exp:true-pipeline`)
- `MODELS_DIR` - host models dir, mounted at `/models` (default `/mnt/ssd2/models`)
- `MODELS_INI` - optional host INI, mounted at `/models.ini` (server wrapper only)
- `HOST_PORT` / `CONTAINER_PORT` - port mapping (server wrapper only, default 8080)
- `NETWORK` - optional docker network to attach
- `CONTAINER_NAME` - optional container name

## Fork env knobs

- `LLAMA_PIPELINE_PARALLEL=0` - force pipeline_parallel off, `n_copies = 1`
- `LLAMA_GRAPH_REUSE_DISABLE=1` - force graph re-allocation per decode

These are read by the binaries via `getenv`, not by the wrappers, so they pass
through to the container unchanged (e.g. `docker run -e LLAMA_GRAPH_REUSE_DISABLE=1 ...`).
