#!/usr/bin/env python3
"""
Validate strict loading of upstream GraspGen head weights into ONNX models.

This checks the parts not covered by test_weight_loading.py:
  - generator diffusion_head
  - discriminator sample_encoder + prediction_head

Usage (on ws-wan or another environment with torch installed):
  python src/python/scripts/test_upstream_head_loading.py \
    --generator-ckpt models/upstream/graspgen_robotiq_2f_140_gen.pth

  python src/python/scripts/test_upstream_head_loading.py \
    --generator-ckpt models/upstream/graspgen_robotiq_2f_140_gen.pth \
    --discriminator-ckpt models/upstream/graspgen_robotiq_2f_140_dis.pth
"""

import argparse
import sys
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "python"))

if torch is not None:
    from graspgen_s600_tools.models.graspgen_onnx import (  # noqa: E402
        GraspGenGeneratorONNX,
        GraspGenDiscriminatorONNX,
        load_upstream_generator_weights,
        load_upstream_discriminator_weights,
    )

NUM_POINTS = 2048
NUM_GRASPS = 20
SAMPLE_DIM = 6


def _validate_checkpoint_path(path: Path) -> bool:
    if not path.exists():
        print(f"  ⚠ Missing checkpoint: {path}")
        return False
    if path.stat().st_size == 0:
        print(f"  ⚠ Empty checkpoint: {path}")
        return False
    return True


def check_generator(checkpoint_path: Path) -> bool:
    print("=" * 70)
    print("Generator full upstream weight loading")
    print("=" * 70)
    print(f"  Checkpoint: {checkpoint_path}")
    if not _validate_checkpoint_path(checkpoint_path):
        return False

    model = GraspGenGeneratorONNX(num_grasps=NUM_GRASPS, grasp_repr="r3_so3")
    load_upstream_generator_weights(model, checkpoint_path, strict=True)
    model.eval()
    print("  ✓ Loaded object_encoder + diffusion_head (strict=True)")

    pc = torch.randn(1, NUM_POINTS, 3)
    noisy_grasps = torch.randn(NUM_GRASPS, SAMPLE_DIM)
    timestep = torch.tensor([5], dtype=torch.long)
    with torch.no_grad():
        output = model(pc, noisy_grasps, timestep)

    ok = output.shape == (NUM_GRASPS, SAMPLE_DIM) and bool(torch.isfinite(output).all())
    print(f"  Forward output: {tuple(output.shape)} finite={bool(torch.isfinite(output).all())}")
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}\n")
    return ok


def check_discriminator(checkpoint_path: Path) -> bool:
    print("=" * 70)
    print("Discriminator full upstream weight loading")
    print("=" * 70)
    print(f"  Checkpoint: {checkpoint_path}")
    if not _validate_checkpoint_path(checkpoint_path):
        return False

    model = GraspGenDiscriminatorONNX(num_grasps=NUM_GRASPS, grasp_repr="r3_so3")
    load_upstream_discriminator_weights(model, checkpoint_path, strict=True)
    model.eval()
    print("  ✓ Loaded object_encoder + sample_encoder + prediction_head (strict=True)")

    pc = torch.randn(1, NUM_POINTS, 3)
    grasps = torch.randn(1, NUM_GRASPS, SAMPLE_DIM)
    with torch.no_grad():
        output = model(pc, grasps)

    ok = output.shape == (1, NUM_GRASPS) and bool(torch.isfinite(output).all())
    print(f"  Forward output: {tuple(output.shape)} finite={bool(torch.isfinite(output).all())}")
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}\n")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate upstream head weight loading into ONNX models"
    )
    parser.add_argument(
        "--generator-ckpt",
        type=Path,
        default=PROJECT_ROOT / "models/upstream/graspgen_robotiq_2f_140_gen.pth",
    )
    parser.add_argument("--discriminator-ckpt", type=Path, default=None)
    args = parser.parse_args()

    if torch is None:
        print("❌ PyTorch is not installed in this Python environment.")
        print("   Run this script from the ws-wan GraspGen virtualenv or another torch environment.")
        return 1

    results = {"generator": check_generator(args.generator_ckpt)}
    if args.discriminator_ckpt is not None:
        results["discriminator"] = check_discriminator(args.discriminator_ckpt)
    else:
        print("⚠ No discriminator checkpoint provided; skipped discriminator loading.\n")

    print("=" * 70)
    for name, ok in results.items():
        print(f"  {name:15s}: {'✅' if ok else '❌'}")
    print("=" * 70)

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
