#!/bin/bash
# Install dependencies for ONNX export on ws-wan

set -e

echo "======================================"
echo "Installing GraspGen-s600 Dependencies"
echo "======================================"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version)
echo "Python version: $PYTHON_VERSION"
echo ""

# Install Python packages
echo "[1/3] Installing core dependencies..."
pip3 install --user torch torchvision onnx onnxruntime pyyaml numpy scipy

echo ""
echo "[2/3] Installing upstream GraspGen..."
cd third_party/GraspGen
pip3 install --user -e .
cd ../..

echo ""
echo "[3/3] Verifying installations..."
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python3 -c "import onnx; print(f'✓ ONNX {onnx.__version__}')"
python3 -c "import onnxruntime; print(f'✓ ONNXRuntime {onnxruntime.__version__}')"
python3 -c "import grasp_gen; print(f'✓ GraspGen imported')"

echo ""
echo "======================================"
echo "✅ All dependencies installed!"
echo "======================================"
echo ""
echo "Next step: Run ONNX export"
echo "  python3 src/python/scripts/export_to_onnx.py \\"
echo "    --gen-checkpoint models/upstream/graspgen_franka_panda_gen.pth \\"
echo "    --dis-checkpoint models/upstream/graspgen_franka_panda_dis.pth \\"
echo "    --config models/upstream/graspgen_franka_panda.yml"
