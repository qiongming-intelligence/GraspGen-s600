#!/usr/bin/env python3
"""
Export GraspGen models to ONNX format.

This script exports both Generator and Discriminator models to ONNX,
preparing them for compilation to Horizon S600 HBM format.

Usage:
    python export_to_onnx.py --gen-checkpoint models/upstream/graspgen_franka_panda_gen.pth \
                             --dis-checkpoint models/upstream/graspgen_franka_panda_dis.pth \
                             --config models/upstream/graspgen_franka_panda.yml
"""

import sys
from pathlib import Path
import argparse
import torch

# Add src/python to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "python"))

from graspgen_s600_tools.export.factories import load_generator, load_discriminator, get_model_info
from graspgen_s600_tools.export.generator import export_generator_onnx
from graspgen_s600_tools.export.discriminator import export_discriminator_onnx


def main():
    parser = argparse.ArgumentParser(description="Export GraspGen models to ONNX")

    # Checkpoint paths
    parser.add_argument("--gen-checkpoint", type=str, required=True,
                       help="Path to generator checkpoint (.pth)")
    parser.add_argument("--dis-checkpoint", type=str, required=True,
                       help="Path to discriminator checkpoint (.pth)")
    parser.add_argument("--config", type=str,
                       help="Path to config YAML (optional)")

    # Output paths
    parser.add_argument("--gen-output", type=str,
                       default="models/onnx/graspgen_generator_pointnet.onnx",
                       help="Output path for generator ONNX")
    parser.add_argument("--dis-output", type=str,
                       default="models/onnx/graspgen_discriminator_pointnet.onnx",
                       help="Output path for discriminator ONNX")

    # Model configuration
    parser.add_argument("--num-points", type=int, default=2048,
                       help="Number of input points")
    parser.add_argument("--num-grasps", type=int, default=20,
                       help="Number of grasps to generate/score")
    parser.add_argument("--grasp-dim", type=int, default=6,
                       help="Grasp representation dimension (9 for r3_6d, 6 for r3_so3)")

    # Export options
    parser.add_argument("--opset-version", type=int, default=17,
                       help="ONNX opset version")
    parser.add_argument("--skip-generator", action="store_true",
                       help="Skip generator export")
    parser.add_argument("--skip-discriminator", action="store_true",
                       help="Skip discriminator export")

    args = parser.parse_args()

    print("=" * 80)
    print("GraspGen ONNX Export")
    print("=" * 80)
    print()

    # Create output directory
    Path(args.gen_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.dis_output).parent.mkdir(parents=True, exist_ok=True)

    # Export Generator
    if not args.skip_generator:
        print("[1/2] Exporting Generator...")
        print("-" * 80)

        try:
            # Load model
            generator = load_generator(args.gen_checkpoint, args.config, device="cpu")
            info = get_model_info(generator)
            print(f"Model info: {info['total_parameters']:,} parameters")
            print()

            # Export to ONNX
            export_generator_onnx(
                generator,
                args.gen_output,
                num_points=args.num_points,
                num_grasps=args.num_grasps,
                output_dim=args.grasp_dim,
                opset_version=args.opset_version,
                verbose=True,
            )
            print(f"✓ Generator ONNX saved to: {args.gen_output}")

        except Exception as e:
            print(f"✗ Generator export failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

        print()

    # Export Discriminator
    if not args.skip_discriminator:
        print("[2/2] Exporting Discriminator...")
        print("-" * 80)

        try:
            # Load model
            discriminator = load_discriminator(args.dis_checkpoint, args.config, device="cpu")
            info = get_model_info(discriminator)
            print(f"Model info: {info['total_parameters']:,} parameters")
            print()

            # Export to ONNX
            export_discriminator_onnx(
                discriminator,
                args.dis_output,
                num_points=args.num_points,
                num_candidates=args.num_grasps,
                grasp_dim=args.grasp_dim,
                opset_version=args.opset_version,
                verbose=True,
            )
            print(f"✓ Discriminator ONNX saved to: {args.dis_output}")

        except Exception as e:
            print(f"✗ Discriminator export failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

        print()

    # Summary
    print("=" * 80)
    print("✅ Export Complete!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Validate ONNX models with ONNXRuntime")
    print("  2. Compare PyTorch vs ONNX outputs (precision < 1e-3)")
    print("  3. Compile to HBM using hb_compile")
    print()
    print("Generated files:")
    if not args.skip_generator:
        print(f"  - {args.gen_output}")
    if not args.skip_discriminator:
        print(f"  - {args.dis_output}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
