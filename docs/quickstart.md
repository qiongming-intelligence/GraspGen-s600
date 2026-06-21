# Quick Start Guide - GraspGen-s600

## Prerequisites

### Hardware
- x86_64 host machine (for compilation)
- Horizon Sunrise 6 (S600) development board

### Software
- Conda / Miniconda
- Git
- Horizon OpenExplorer Toolchain 3.7.0+
- Horizon Runtime 4.7.5+

## Setup

### 1. Clone Repository

```bash
git clone https://github.com/qiongming-intelligence/GraspGen-s600.git
cd GraspGen-s600
```

### 2. Clone Upstream GraspGen

```bash
git clone https://github.com/NVlabs/GraspGen.git third_party/GraspGen
```

### 3. Create Conda Environments

#### Export Environment (PyTorch → ONNX)

```bash
conda create -n graspgen-export python=3.11
conda activate graspgen-export
conda install pytorch torchvision onnx onnxruntime -c pytorch
pip install diffusers omegaconf hydra-core transformers trimesh scipy

# Install upstream GraspGen
cd third_party/GraspGen
pip install -e .
cd ../..
```

#### Compile Environment (ONNX → HBM)

```bash
conda create -n graspgen-compile python=3.10
conda activate graspgen-compile
conda install numpy=1.23.0

# Install Horizon Toolchain
pip install horizon_tc_ui-<version>.whl
```

### 4. Download Pretrained Weights

```bash
# Option 1: Using huggingface-cli
pip install huggingface_hub
huggingface-cli download adithyamurali/GraspGenModels \
  franka_panda_pointnet_generator.pth \
  franka_panda_pointnet_discriminator.pth \
  --local-dir models/upstream/

# Option 2: Manual download from
# https://huggingface.co/adithyamurali/GraspGenModels
# Place files in models/upstream/
```

## Current Status (Phase 1)

### What Works ✅

- Directory structure created
- Contract generation module
- Generator and Discriminator contracts generated
- Upstream GraspGen integrated

### In Progress 🚧

- ONNX export implementation
- Model loading utilities
- Export validation tools

### Not Yet Started ⏳

- HBM compilation
- BPU runtime adapters
- Board-side validation

## Generate Contracts

```bash
# Activate export environment
conda activate graspgen-export

# Generate contracts
PYTHONPATH=src/python python3 src/python/graspgen_s600_tools/export/contract.py

# Check generated files
ls -l configs/manifests/
# Output:
#   graspgen_generator.json
#   graspgen_discriminator.json
```

## Project Structure

```
GraspGen-s600/
├── src/python/graspgen_s600_tools/
│   ├── export/          # ✅ Contract generation done
│   ├── convert/         # ⏳ TODO: Precision conversion
│   ├── runtime/         # ⏳ TODO: BPU adapters
│   └── debug/           # ⏳ TODO: Validation tools
├── models/
│   ├── onnx/            # Empty (will contain .onnx files)
│   ├── hbm/             # Empty (will contain .hbm files)
│   └── upstream/        # Put pretrained weights here
├── configs/
│   └── manifests/       # ✅ Contracts generated
├── third_party/
│   └── GraspGen/        # ✅ Cloned
└── docs/                # ✅ Documentation
```

## Next Steps

See [Phase 1 Summary](phase1_summary.md) for detailed roadmap.

### For Developers

1. Implement ONNX export wrapper
2. Create model loading utilities
3. Export Generator and Discriminator to ONNX
4. Validate ONNX outputs vs PyTorch

### For Users

Wait for Phase 1 completion, then follow the full pipeline:
1. Export ONNX
2. Compile HBM
3. Deploy to S600
4. Run inference

## Troubleshooting

### Import Errors

```bash
# Make sure PYTHONPATH is set
export PYTHONPATH=$PWD/src/python:$PYTHONPATH

# Or use conda develop
cd src/python
conda develop .
```

### Missing Dependencies

```bash
# Upstream GraspGen dependencies
cd third_party/GraspGen
pip install -r requirements.txt
```

### Git Submodules

```bash
# If using submodules (future)
git submodule update --init --recursive
```

## Documentation

- [Phase 1 Progress](phase1_progress.md) - Current task list
- [Phase 1 Summary](phase1_summary.md) - Detailed analysis and plan
- [README.md](../README.md) - Project overview

## Contact

- **Author**: lvyufeng
- **Email**: lvyufeng@cqu.edu.cn
- **Organization**: Qiongming Intelligence

## License

This adaptation follows the original GraspGen project license.
