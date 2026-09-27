# T8: NCCL for tensor split - blocked on sm_50

Status: complete (blocked, negative result). Executed on rig1, 2026-09-27.

Tracked by: issue #11. Follow-up to #10 (tensor split) and #9 (decode ceiling).

## 1. Summary

#11 set out to enable NCCL and re-benchmark the `-sm tensor` allreduce. The
premise was that NCCL is not compiled in (`GGML_CUDA_NCCL=OFF`) and enabling it
would replace the host-mediated butterfly allreduce with a 4-GPU NCCL allreduce.
Both halves of that premise are wrong.

1. NCCL is already compiled in. `GGML_CUDA_NCCL` defaults ON, the build image
   ships `libnccl-dev` 2.23.4, `find_package(NCCL)` finds it
   (`NCCL_LIBRARY=/usr/lib/x86_64-linux-gnu/libnccl.so`), and the built
   `libggml-cuda.so` links `ncclCommInitAll` / `ncclAllReduce`.
2. NCCL cannot initialise on the M10. `ncclCommInitAll` fails at runtime:
   `cudaMemPoolCreate` returns `cudaErrorNotSupported`. The M10 is Maxwell sm_50,
   which does not support CUDA VMM (a Pascal-era, compute-capability 6.0+
   feature); NCCL 2.23.4 calls `cudaMemPoolCreate` unconditionally in
   `ncclCommInitRank` (`src/init.cc:417`), and there is no env knob to skip it.

So the allreduce chain on rig1 is fixed at nccl (fails) -> internal (fails:
n_devices=4, sm_50) -> butterfly. NCCL is not reachable on this hardware.

## 2. Evidence

Diagnostic run (`llama-pipeline -sm tensor -fa on`, Qwen3.5-9B-Q4_K_M):

```
W NCCL init failed (unhandled cuda error ...); falling back to internal AllReduce
W internal AllReduce init failed (n_devices != 2?); falling back to meta-backend butterfly
```

With `NCCL_DEBUG=INFO`:

```
NCCL INFO NCCL version 2.23.4+cuda12.6
init.cc:417 NCCL WARN Cuda failure 'operation not supported'
```

NCCL 2.23.4 source, `src/init.cc` (GitHub tag v2.23.4-1):

```c
cudaMemPoolProps props = {};
props.allocType = cudaMemAllocationTypePinned;
props.handleTypes = cudaMemHandleTypeNone;
props.location.type = cudaMemLocationTypeDevice;
props.location.id = comm->cudaDev;
CUDACHECK(cudaMemPoolCreate(&comm->memPool, &props));   // line 417
```

`cudaMemPoolCreate` is the CUDA VMM API. The M10 (sm_50) returns
`cudaErrorNotSupported`. `NCCL_P2P_DISABLE=1` has no effect (the failure is not
peer access, it is VMM). NCCL 2.18 predates this call (checked `v2.18.3-1`), so
an old NCCL would avoid it, but installing an NCCL built for CUDA 11.x into a
CUDA 12.6 image is not a supported path and is not worth it given the finding
below.

## 3. K-scaling of `-sm tensor` (the remaining question from #10)

Even without NCCL, #10 left open whether `-sm tensor` scales with K. Measured
with the same driver/model (K contexts, overlap mode, n=32, 3 iterations):

| K | tok/s (avg) |
| --- | ---: |
| 1 | 15.83 |
| 2 | 16.30 |
| 4 | 16.58 |

Tensor mode is essentially flat over K (+4.7% from K=1 to K=4). This is expected
and is not an allreduce artefact: tensor parallelism already splits every layer
across all 4 GPUs, so a single token uses all 4 GPUs at once. Unlike layer mode
(where one token uses one GPU and K>1 overlaps decodes, +47% from #4), tensor
mode is already parallel at K=1 and K>1 only fills small pipeline gaps.

Because tensor mode is already ~2x layer-mode-K=4 at K=1 (15.83 vs 7.94 tok/s)
and its K-scaling is flat, there is no scenario where NCCL would unlock a
meaningful further gain on this hardware: the allreduce is not on the critical
path for the models tested (consistent with #10).

## 4. Conclusion

- NCCL cannot run on the M10: sm_50 lacks CUDA VMM, which NCCL 2.23.4 requires
  at init. The butterfly allreduce is the only path available on rig1.
- `-sm tensor` does not scale with K because tensor parallelism already uses all
  4 GPUs per token. The butterfly allreduce is not the scaling bottleneck.
- The ticket's success criteria are not met as stated (NCCL does not initialise),
  but the underlying question is answered: NCCL is a dead end on this hardware,
  and the K-scaling plateau is architectural, not an allreduce shortfall.

## Related

- #10 (tensor split), #9 (decode ceiling), #7 (control), #4 (overlap).
- `docs/experiments/pipeline-parallelism-tensor-split.md` (the sweep this
  follows up).
