# ONNX Export Guide

## Overview

This guide explains how to export GraspGen models to ONNX format for Horizon S600 deployment.

## Prerequisites

### On ws-wan (or x86_64 machine with GPU access)

```bash
# Install dependencies
pip install torch torchvision onnx onnxruntime pyyaml

# Install upstream GraspGen
cd third_party/GraspGen
pip install -e .
cd ../..
```

## Step 1: Download Pretrained Weights

Weights are already downloaded on ws-wan:
```
~/GraspGen-s600/models/upstream/
├── graspgen_franka_panda_gen.pth (907 MB)
├── graspgen_franka_panda_dis.pth (166 MB)
└── graspgen_franka_panda.yml (4.8 KB)
```

## Step 2: Export to ONNX

### Basic Export

```bash
cd /path/to/GraspGen-s600

python src/python/scripts/export_to_onnx.py \
  --gen-checkpoint models/upstream/graspgen_franka_panda_gen.pth \
  --dis-checkpoint models/upstream/graspgen_franka_panda_dis.pth \
  --config models/upstream/graspgen_franka_panda.yml
```

### Custom Configuration

```bash
python src/python/scripts/export_to_onnx.py \
  --gen-checkpoint models/upstream/graspgen_franka_panda_gen.pth \
  --dis-checkpoint models/upstream/graspgen_franka_panda_dis.pth \
  --config models/upstream/graspgen_franka_panda.yml \
  --num-points 2048 \
  --num-grasps 20 \
  --grasp-dim 12 \
  --gen-output models/onnx/generator.onnx \
  --dis-output models/onnx/discriminator.onnx
```

### Export Only Generator

```bash
python src/python/scripts/export_to_onnx.py \
  --gen-checkpoint models/upstream/graspgen_franka_panda_gen.pth \
  --dis-checkpoint models/upstream/graspgen_franka_panda_dis.pth \
  --skip-discriminator
```

## Step 3: Validate ONNX Export

### Check Model Structure

```bash
python -c "
import onnx
model = onnx.load('models/onnx/graspgen_generator_pointnet.onnx')
onnx.checker.check_model(model)
print('✓ ONNX model is valid')
print(f'  Inputs: {[i.name for i in model.graph.input]}')
print(f'  Outputs: {[o.name for o in model.graph.output]}')
"
```

### Run ONNXRuntime Inference

```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession('models/onnx/graspgen_generator_pointnet.onnx')

# Prepare inputs
pc = np.random.randn(1, 2048, 3).astype(np.float32)
noisy_grasps = np.random.randn(20, 9).astype(np.float32)
timestep = np.array([0], dtype=np.int64)

# Run inference
outputs = session.run(
    None,
    {
        'pc': pc,
        'noisy_grasps': noisy_grasps,
        'timestep': timestep,
    }
)

print(f"Output shape: {outputs[0].shape}")
print(f"✓ ONNXRuntime inference successful")
```

## Architecture Notes

### Generator Export Strategy

The original Generator uses a diffusion loop (10-20 steps) that's difficult to export directly to ONNX. Our strategy:

1. **Export single denoising step**: The ONNX model contains only one denoising iteration
2. **Python-side loop**: The full diffusion process runs in Python, calling the ONNX model repeatedly
3. **Fixed shapes**: All inputs are fixed size for BPU compatibility

### Discriminator Export

The Discriminator is simpler - a single forward pass that scores grasp candidates. Direct ONNX export works well.

## Grasp Representations

The original model uses `r3_so3` (position + SO(3) rotation):
- **r3_so3**: 12 dimensions (3 position + 9 rotation matrix)
- **r3_6d**: 9 dimensions (3 position + 6D rotation)

For S600 adaptation, we may convert to `r3_6d` for efficiency.

## Troubleshooting

### Import Error: No module named 'grasp_gen'

```bash
cd third_party/GraspGen
pip install -e .
```

### ONNX Export Fails: Unsupported operator

Some operators (like custom CUDA kernels) may not be ONNX-compatible:
- Check if PointNet++ uses custom ops
- Consider using standard PyTorch PointNet
- See `generator.py` for fallback strategies

### Shape Mismatch

Ensure contract shapes match export shapes:
```bash
# Check contract
cat configs/manifests/graspgen_generator.json

# Verify ONNX shapes match
python -c "import onnx; m=onnx.load('models/onnx/graspgen_generator_pointnet.onnx'); print([i.type.tensor_type.shape for i in m.graph.input])"
```

## Next Steps

After successful ONNX export:

1. **Precision validation**: Compare PyTorch vs ONNX outputs (see `validate_onnx.py`)
2. **Calibration data**: Collect diverse point clouds for quantization
3. **HBM compilation**: Use `hb_compile` to generate BPU binaries
4. **Board-side testing**: Deploy and validate on S600

## Files Generated

```
models/onnx/
├── graspgen_generator_pointnet.onnx    # Generator single-step denoiser
└── graspgen_discriminator_pointnet.onnx # Discriminator scorer
```

## References

- [ONNX Opset Documentation](https://github.com/onnx/onnx/blob/main/docs/Operators.md)
- [PyTorch ONNX Export](https://pytorch.org/docs/stable/onnx.html)
- Horizon OpenExplorer Toolchain Guide
