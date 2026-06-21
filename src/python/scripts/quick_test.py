#!/usr/bin/env python3
"""
Quick test script to verify ONNX export is working.

This runs a basic sanity check without full validation.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "python"))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ PyTorch import failed: {e}")
        return False

    try:
        import onnx
        print(f"  ✓ ONNX {onnx.__version__}")
    except ImportError as e:
        print(f"  ✗ ONNX import failed: {e}")
        return False

    try:
        import onnxruntime as ort
        print(f"  ✓ ONNXRuntime {ort.__version__}")
    except ImportError as e:
        print(f"  ✗ ONNXRuntime import failed: {e}")
        return False

    try:
        import grasp_gen
        print(f"  ✓ GraspGen imported")
    except ImportError as e:
        print(f"  ✗ GraspGen import failed: {e}")
        print(f"     Make sure to install: cd third_party/GraspGen && pip install -e .")
        return False

    return True


def test_model_loading():
    """Test that models can be loaded."""
    print("\nTesting model loading...")

    from graspgen_s600_tools.export.factories import load_generator

    gen_path = PROJECT_ROOT / "models" / "upstream" / "graspgen_franka_panda_gen.pth"

    if not gen_path.exists():
        print(f"  ✗ Generator checkpoint not found: {gen_path}")
        return False

    try:
        print(f"  Loading generator from {gen_path.name}...")
        generator = load_generator(str(gen_path), device="cpu")
        print(f"  ✓ Generator loaded successfully")
        return True
    except Exception as e:
        print(f"  ✗ Generator loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_onnx_export():
    """Test that ONNX export runs without errors."""
    print("\nTesting ONNX export...")

    from graspgen_s600_tools.export.factories import load_generator
    from graspgen_s600_tools.export.generator import export_generator_onnx
    import tempfile

    gen_path = PROJECT_ROOT / "models" / "upstream" / "graspgen_franka_panda_gen.pth"

    try:
        print(f"  Loading generator...")
        generator = load_generator(str(gen_path), device="cpu")

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = f.name

        print(f"  Exporting to ONNX...")
        export_generator_onnx(
            generator,
            onnx_path,
            num_points=2048,
            num_grasps=20,
            output_dim=9,
            verbose=False,
        )

        print(f"  ✓ ONNX export successful")

        # Clean up
        Path(onnx_path).unlink()

        return True

    except Exception as e:
        print(f"  ✗ ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("GraspGen-s600 ONNX Export Quick Test")
    print("=" * 60)
    print()

    results = []

    # Test 1: Imports
    results.append(("Imports", test_imports()))

    # Test 2: Model loading (only if imports passed)
    if results[0][1]:
        results.append(("Model Loading", test_model_loading()))
    else:
        print("\nSkipping model loading test (imports failed)")
        results.append(("Model Loading", False))

    # Test 3: ONNX export (only if loading passed)
    if results[1][1]:
        results.append(("ONNX Export", test_onnx_export()))
    else:
        print("\nSkipping ONNX export test (model loading failed)")
        results.append(("ONNX Export", False))

    # Summary
    print()
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:<20} {status}")

    all_passed = all(r[1] for r in results)

    print()
    if all_passed:
        print("✅ All tests passed! Ready for full export.")
        return 0
    else:
        print("❌ Some tests failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
