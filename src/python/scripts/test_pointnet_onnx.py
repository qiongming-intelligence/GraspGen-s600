#!/usr/bin/env python3
"""
Test ONNX export compatibility of PointNet encoder.

This script verifies that the PointNet encoder can be exported to ONNX
and that the exported model produces the same outputs as PyTorch.
"""

import sys
from pathlib import Path
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "python"))

from graspgen_s600_tools.models.pointnet_encoder import PointNetEncoder


def test_onnx_export():
    """Test ONNX export and numerical consistency."""
    print("=" * 70)
    print("Testing PointNet ONNX Export")
    print("=" * 70)
    print()

    # Create model
    print("[1/5] Creating PointNet encoder...")
    encoder = PointNetEncoder(num_classes=512, normal_channel=False)
    encoder.eval()
    print("✓ Model created")
    print()

    # Test input
    batch_size = 1
    num_points = 2048
    xyz = torch.randn(batch_size, num_points, 3)

    # PyTorch inference
    print("[2/5] Running PyTorch inference...")
    with torch.no_grad():
        pt_output = encoder(xyz)
    print(f"✓ PyTorch output shape: {pt_output.shape}")
    print()

    # Export to ONNX
    print("[3/5] Exporting to ONNX...")
    import tempfile
    onnx_path = tempfile.mktemp(suffix=".onnx")

    try:
        torch.onnx.export(
            encoder,
            xyz,
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['point_cloud'],
            output_names=['features'],
            dynamic_axes=None,  # Fixed shapes for BPU
            verbose=False,
        )
        print(f"✓ ONNX export successful: {onnx_path}")
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        return False

    print()

    # Verify ONNX model
    print("[4/5] Verifying ONNX model...")
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid")
    except Exception as e:
        print(f"✗ ONNX verification failed: {e}")
        return False

    print()

    # ONNXRuntime inference
    print("[5/5] Testing ONNXRuntime inference...")
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path)

        # Run inference
        onnx_output = session.run(
            None,
            {'point_cloud': xyz.numpy()}
        )[0]

        print(f"✓ ONNXRuntime output shape: {onnx_output.shape}")
        print()

        # Compare outputs
        print("Comparing PyTorch vs ONNX outputs...")
        error = np.abs(pt_output.numpy() - onnx_output).max()
        mean_error = np.abs(pt_output.numpy() - onnx_output).mean()

        print(f"  Max error: {error:.6f}")
        print(f"  Mean error: {mean_error:.6f}")
        print()

        # Precision gate
        threshold = 1e-3
        if error < threshold:
            print(f"✅ PRECISION GATE PASSED (error < {threshold})")
            return True
        else:
            print(f"❌ PRECISION GATE FAILED (error = {error} >= {threshold})")
            return False

    except Exception as e:
        print(f"✗ ONNXRuntime inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Clean up
        Path(onnx_path).unlink(missing_ok=True)


def main():
    success = test_onnx_export()

    print()
    print("=" * 70)
    if success:
        print("✅ All tests passed!")
        print("PointNet encoder is ONNX-compatible and ready for deployment.")
        return 0
    else:
        print("❌ Tests failed!")
        print("Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
