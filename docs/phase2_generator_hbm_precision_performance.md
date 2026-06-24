# Phase 2: Generator HBM Precision and Performance Comparison

## Summary

**Date**: 2026-06-24  
**Target**: Horizon/Sunrise S600, `nash-p`  
**Runtime**: local S600 `hrt_model_exec`  
**Reference**: ws-wan RTX 4090 CUDA PyTorch generator output  
**Output compared**: `noise_pred`, shape `(20, 6)`

The best current generator-only deployment candidate is:

```text
pointnet_sa1_neural_gen_bpu.hbm
pointnet_sa2_neural_gen_conv1_cpu.hbm
pointnet_sa3_encoder_head_gen_all_cpu.hbm
graspgen_generator_head_temb_simplified_pred_cpu.hbm
```

This keeps FPS / ball query / grouping / timestep embedding on CPU, runs the BPU-friendly PointNet and generator-head portions on BPU, and keeps the final prediction MLP in float32 on CPU.

Result:

```text
HBM subgraph time: 262.389 ms
max_abs:           0.026734233
mean_abs:          0.007267100
```

Compared with the all-CPU/f32 split baseline:

```text
1406.748 ms -> 262.389 ms
speedup: 5.36x
```

Compared with the previous precision-preserving hybrid baseline:

```text
820.253 ms -> 262.389 ms
speedup: 3.13x
```

## Measurement Notes

- Timings are the sum of HBM subgraph `Infer time` values from local S600 `hrt_model_exec`.
- Timings do **not** include CPU-side geometry preprocessing:
  - FPS
  - ball query
  - grouping
  - timestep embedding
- Precision metrics compare local S600 HBM outputs against:

```text
test_data/s600_precision_refs_cuda/generator_pytorch_cuda_noise_pred_f32_20x6.bin
```

- Some HBM outputs are stride-padded to `(20, 8)`; precision metrics use the logical `[:, :6]` slice.
- Current results are from one deterministic reference input. Multi-input / multi-timestep validation should be run before treating thresholds as final deployment guarantees.

## End-to-End Generator Split Comparison

| Variant | SA1 | SA2 | SA3 | Generator head | HBM time | max_abs | mean_abs | Decision |
|---|---|---|---|---|---:|---:|---:|---|
| all CPU/f32 split | CPU/f32 | CPU/f32 | CPU/f32 | CPU/f32 simplified | `1406.748 ms` | `0.029335916` | `0.005908893` | Correctness baseline, too slow |
| SA1 BPU | BPU int | CPU/f32 | CPU/f32 | CPU/f32 simplified | `1200.724 ms` | `0.025236875` | `0.005509773` | Safe, limited speedup |
| SA1 BPU + SA2 hybrid | BPU int | conv1 CPU + rest BPU | CPU/f32 | CPU/f32 simplified | `820.253 ms` | `0.025388837` | `0.005507584` | Previous recommended precision-preserving path |
| fast int16 | BPU int | hybrid/BPU | BPU int16 | BPU int16 | `200.928 ms` | `0.081851095` | `0.023040915` | Fastest, precision loss too large |
| **pred_cpu candidate** | BPU int | conv1 CPU + rest BPU | CPU/f32 | BPU int16 + final prediction MLP CPU/f32 | **`262.389 ms`** | **`0.026734233`** | **`0.007267100`** | **Recommended tradeoff** |

## Speedup Summary

### Relative to all CPU/f32 split

| Variant | HBM time | Speedup |
|---|---:|---:|
| all CPU/f32 split | `1406.748 ms` | `1.00x` |
| SA1 BPU | `1200.724 ms` | `1.17x` |
| SA1 BPU + SA2 hybrid | `820.253 ms` | `1.71x` |
| **pred_cpu candidate** | **`262.389 ms`** | **`5.36x`** |
| fast int16 | `200.928 ms` | `7.00x` |

### Relative to previous hybrid baseline

| Variant | HBM time | Speedup |
|---|---:|---:|
| SA1 BPU + SA2 hybrid + head CPU/f32 | `820.253 ms` | `1.00x` |
| **pred_cpu candidate** | **`262.389 ms`** | **`3.13x`** |
| fast int16 | `200.928 ms` | `4.08x` |

## Generator Head Variant Comparison

The following experiments keep the front half fixed:

```text
SA1: BPU int
SA2: conv1 CPU + rest BPU
SA3: CPU/f32
```

Only the generator head HBM changes.

| Head variant | Head time | Estimated total HBM time | max_abs | mean_abs | Decision |
|---|---:|---:|---:|---:|---|
| CPU/f32 simplified head | `~599.6 ms` | `~820.3 ms` | `0.025388837` | `0.005507584` | Best precision, too slow |
| all int16 head | `~1.34 ms` | `~200.9 ms` | `0.081851095` | `0.023040915` | Too much precision loss |
| **pred_cpu** | **`41.798 ms`** | **`262.389 ms`** | **`0.026734233`** | **`0.007267100`** | **Best current tradeoff** |
| tail_cpu | `87.781 ms` | `308.372 ms` | `0.028551698` | `0.006189113` | Usable, slower than pred_cpu |
| ln_pred_cpu | `47.671 ms` | `268.262 ms` | `0.150984526` | `0.039804373` | Not usable |
| all float16 quant_config | `1779.460 ms` | `2000.051 ms` | `0.031506382` | `0.008089208` | Runs, but falls back heavily and is too slow |
| int16 + LayerNorm float16 | `1.392 ms` | `221.983 ms` | `0.080411941` | `0.024020454` | LayerNorm float16 alone does not fix precision |
| int16 + LayerNorm float16 + pred CPU/f32 | `47.716 ms` | `268.307 ms` | `0.026383117` | `0.007369281` | Usable backup, slightly slower than pred_cpu |
| int16 + prediction float16 | `129.653 ms` | `350.244 ms` | `0.026664078` | `0.007696058` | Usable, slower than pred_cpu |

## Error Source Isolation

| Variant | SA3 | Generator head | max_abs | mean_abs | Interpretation |
|---|---|---|---:|---:|---|
| SA3 int16 + head CPU/f32 | BPU int16 | CPU/f32 | `0.040669173` | `0.009370428` | SA3 int16 adds error, but is not the dominant issue |
| SA3 CPU/f32 + head int16 | CPU/f32 | BPU int16 | `0.083386838` | `0.023814589` | Head int16 is the dominant precision-loss source |
| fast int16 | BPU int16 | BPU int16 | `0.081851095` | `0.023040915` | Similar to head-int16 error, confirming head sensitivity |

Error-source ranking:

```text
head int16  >>>  SA3 int16  >  SA1/SA2 BPU
```

The final generator prediction MLP is the most sensitive section. Keeping only that MLP on CPU/f32 recovers most of the lost precision while retaining most of the head speedup.

## Float16 Toolchain Findings

The newer S100/S600 OpenExplorer documentation and the current ws-wan `hb_compile` environment both accept `quant_config` entries such as:

```yaml
quant_config:
  model_config:
    all_node_type: "float16"
```

and targeted configurations such as:

```yaml
quant_config:
  model_config:
    all_node_type: "int16"
  op_config:
    LayerNormalization:
      qtype: "float16"
```

However, the experiments show that full float16 is not a useful path for this generator head on the current runtime:

```text
all float16 head time: 1779.460 ms
all float16 max_abs:   0.031506382
```

The node tables show many heavy Conv / MatMul-equivalent nodes falling back to CPU when configured through float16/f32 paths. The practical acceleration path remains:

```text
int8/int16 BPU compute for heavy subgraphs
+ selective CPU/f32 protection for numerically sensitive output layers
```

## Recommended Deployment Candidate

Use:

```text
CPU side:
  FPS
  ball_query
  grouping
  timestep embedding

HBM side:
  pointnet_sa1_neural_gen_bpu.hbm
  pointnet_sa2_neural_gen_conv1_cpu.hbm
  pointnet_sa3_encoder_head_gen_all_cpu.hbm
  graspgen_generator_head_temb_simplified_pred_cpu.hbm
```

Expected HBM-only metrics on the tested input:

```text
HBM time: 262.389 ms
max_abs:  0.026734233
mean_abs: 0.007267100
```

This is the best current precision/performance tradeoff:

- Much faster than the all-CPU/f32 split (`5.36x` HBM-subgraph speedup).
- Much more accurate than the all-int16 fast path.
- Simpler and slightly faster than the LayerNorm-float16 + prediction-CPU variant.

## Next Steps

1. Promote `graspgen_generator_head_temb_simplified_pred_cpu` to the primary generator-head HBM candidate.
2. Add a reproducible generator-only local smoke/deployment script for the four-HBM chain.
3. Validate the selected candidate on multiple point clouds and diffusion timesteps.
4. Decide final acceptance thresholds for `max_abs`, `mean_abs`, and downstream grasp quality.
