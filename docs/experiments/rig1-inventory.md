# rig1 inventory

Reference for working on the layer-parallelism experiments. Captured 2026-09-27.
Re-investigate only when something here is stale.

## This repository (workspace)

- Fork of [llama.cpp](https://github.com/ggml-org/llama.cpp), remote `ggeorgovassilis/llama.cpp-experiments`.
- Active branch: `true-pipeline` (only one commit ahead of `master`: "initial setup", which added `.github/copilot-instructions.md`). No experiment code exists yet.
- Goal: increase GPU utilisation on multi-GPU deployments via layer parallelism.
- Key docs:
  - `docs/multi-gpu.md` - split modes (`--split-mode layer|row|tensor`), `--tensor-split`, `--main-gpu`, `-ngl`, flash-attn/KV-cache constraints, troubleshooting.
  - `docs/build.md` - CMake build flags (incl. CUDA/NCCL/HIP options).
  - `docs/backend/` - per-backend notes (CUDA, OPENCL, SYCL, Vulkan, ROCm, ...).
  - `docs/development/HOWTO-add-model.md` - model architecture guide.
  - `docs/plan/` - empty; intended for experiment plans (see planning rule in `copilot-instructions.md`).
- The project is a fork; upstream changes land on `master`. Keep `true-pipeline` work rebased on it.

## rig1 server

- SSH: `ssh rig1.local` (user `george`). No sudo; do everything in Docker.
- CPU: 2x Intel Xeon E5-2640 v0 (12 physical / 24 logical, 2.5 GHz, 2 sockets).
- RAM: 125 GiB.
- GPU: 4x NVIDIA Tesla M10, 8 GiB VRAM each (GM107 / Maxwell gen 1). No tensor cores; old arch - FP16 flash-attn and tensor-split are limited/unsupported on some paths. This is why layer split is the default choice.
- Driver 580.173.02, CUDA 13.0.
- Disk: 63 GB root on `/dev/sdb3`. Models live on `/mnt/ssd2` (symlinked as `~/llamacpp/models`).
- Git authenticated as `George Georgovassilis <george@georgovassilis.com>`.
- `gh` CLI is NOT installed on rig1 (the instructions that claimed it was are wrong). Use `gh` from the local dev machine; transfer via git/ssh.

## rig1 `~/llamacpp` = the `rig1` deployment repo

`~/llamacpp` is NOT llama.cpp source. It is the `ggeorgovassilis/rig1` repo - a
single-host inference stack. Useful to know because the experiments run through it.

> CAUTION: `~/llamacpp` is an existing deployment used for something else. Read
> its models, `models.ini` and configs for reference, but do NOT work inside it.
> Create a separate directory (e.g. `~/experiments`) for all experiment work.

- Stack: `llama-server` (Router Mode) behind a LiteLLM gateway, plus HAProxy,
  VictoriaMetrics, Grafana, and node/gpu exporters.
- Everything is driven by one TOML topology file per host under `configs/`.
  `scripts/generate.py` turns it into a `docker-compose.yml`, a LiteLLM routing
  table, and per-instance `models.ini` presets under `generated/<name>/`.
- Lifecycle:
  - `./up.sh configs/rig1.toml -d` - regenerate + start.
  - `./down.sh configs/rig1.toml` - teardown (add `-v` to drop volumes).
  - `./scripts/generate.py configs/<name>.toml` - regenerate only.
- Topology files present: `rig1.toml` (4 sharded instances, one model per GPU),
  `m10-unified.toml` (one instance, layer-split across all 4 GPUs), `frankenrig.toml`.
- `models.ini` is the shared model catalogue (generated, do not hand-edit). Each
  `[name]` section maps a friendly name to a GGUF path under `/models/`.
  `servers[].models` in a topology file references these names.
- Models store: `/mnt/ssd2/models` (~400 GB used). Symlink `~/llamacpp/models -> /mnt/ssd2/models`.
  Notable models: gemma-4 (12b/26B/E2B/E4B), Qwen3.5/3.6/3.8 (3B..35B), granite-4.2,
  Nanbeige4.2, LFM2.5, NVIDIA-Nemotron-3.5-Lightning-30B, Ornith-1.5, plus
  `mmproj-*` and `mtp-*` sidecar files.
- Benchmarks: `benchmarks/` (platform / gpu-clpeak / llamacpp runners), results in
  `benchmarks/results/`, baseline writeup in `docs/benchmarks.md`.
- Secrets (`DEEPSEEK_API_KEY`) go in `.litellm.env` (gitignored). HF token in `hftoken`.

## Running containers (as of inventory)

`docker ps` showed: `llama-server-0` (CUDA, host 11435 -> 8080), LiteLLM gateway
(11434 -> 4000), `haproxy` (80), `victoriametrics`, `grafana`, `node_exporter`,
`nvidia_gpu_exporter`, `litellm-database` (postgres).

## Shortcuts

- Multi-GPU semantics: `docs/multi-gpu.md`.
- Build/backend flags: `docs/build.md`, `docs/backend/`.
- rig1 topology source of truth: `ssh rig1.local 'cat ~/llamacpp/configs/<name>.toml'`.
- Model catalogue: `ssh rig1.local 'cat ~/llamacpp/models.ini'`.
- Live GPU state: `ssh rig1.local nvidia-smi`.
- rig1 deployment repo README: `ssh rig1.local 'cat ~/llamacpp/README.md'`.
