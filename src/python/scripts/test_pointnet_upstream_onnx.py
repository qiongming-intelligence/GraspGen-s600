#!/usr/bin/env python3
"""
Test ONNX export compatibility of the weight-compatible PointNetUpstream encoder.

This verifies that the faithful pure-PyTorch FPS implementation can be exported
and that ONNXRuntime matches PyTorch numerically.

Usage (on ws-wan):
  python src/python/scripts/test_pointnet_upstream_onnx.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "python"))

from graspgen_s600_tools.models.pointnet_upstream import PointNetUpstream


def count_ops(onnx_model) -> dict[str, int]:
    """Count ONNX node types for quick export inspection."""
    counts: dict[str, int] = {}
    for node in onnx_model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


def test_onnx_export() -> bool:
    """Test PointNetUpstream(sampling='fps') ONNX export and numerical consistency."""
    print("=" * 70)
    print("Testing PointNetUpstream ONNX Export (faithful FPS)")
    print("=" * 70)

    print("\n[1/5] Creating PointNetUpstream encoder...")
    encoder = PointNetUpstream(output_embedding_dim=512, feature_dim=-1, sampling="fps")
    encoder.eval()
    print("✓ Model created")

    print("\n[2/5] Running PyTorch inference...")
    torch.manual_seed(42)
    xyz = torch.randn(1, 2048, 3)
    with torch.no_grad():
        pt_output = encoder(xyz)
    print(f"✓ PyTorch output shape: {pt_output.shape}")

    print("\n[3/5] Exporting to ONNX...")
    onnx_path = tempfile.mktemp(suffix=".onnx")
    try:
        torch.onnx.export(
            encoder,
            xyz,
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["point_cloud"],
            output_names=["features"],
            dynamic_axes=None,
            verbose=False,
        )
        print(f"✓ ONNX export successful: {onnx_path}")
    except Exception as exc:
        print(f"✗ ONNX export failed: {exc}")
        return False

    print("\n[4/5] Verifying ONNX model...")
    try:
        import onnx

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        op_counts = count_ops(onnx_model)
        print("✓ ONNX model is valid")
        print(f"  Nodes: {len(onnx_model.graph.node)}")
        for op_name in ["ArgMax", "Gather", "ReduceSum", "Min", "TopK", "Conv", "Gemm"]:
            if op_name in op_counts:
                print(f"  {op_name}: {op_counts[op_name]}")
    except Exception as exc:
        print(f"✗ ONNX verification failed: {exc}")
        return False

    print("\n[5/5] Testing ONNXRuntime inference...")
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(onnx_path)
        onnx_output = session.run(None, {"point_cloud": xyz.numpy()})[0]
        print(f"✓ ONNXRuntime output shape: {onnx_output.shape}")

        abs_diff = np.abs(pt_output.numpy() - onnx_output)
        max_error = abs_diff.max()
        mean_error = abs_diff.mean()
        print("\nComparing PyTorch vs ONNX outputs...")
        print(f"  Max error: {max_error:.6e}")
        print(f"  Mean error: {mean_error:.6e}")

        threshold = 1e-3
        if max_error < threshold:
            print(f"\n✅ PRECISION GATE PASSED (error < {threshold})")
            return True

        print(f"\n❌ PRECISION GATE FAILED (error = {max_error} >= {threshold})")
        return False
    except Exception as exc:
        print(f"✗ ONNXRuntime inference failed: {exc}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        Path(onnx_path).unlink(missing_ok=True)


def main() -> int:
    success = test_onnx_export()
    print("\n" + "=" * 70)
    if success:
        print("✅ All tests passed!")
        print("PointNetUpstream faithful-FPS encoder is ONNX-compatible.")
        return 0

    print("❌ Tests failed!")
    return 1


if __name__ == "__main__":
    sys.exit(main())
