#!/usr/bin/env python3
"""
Generate default contracts for GraspGen models.

This script creates the JSON contract files that define the model
input/output shapes and compilation hints for Horizon S600.
"""

import sys
from pathlib import Path

# Add src/python to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src" / "python"))

from graspgen_s600_tools.export import (
    generate_generator_contract,
    generate_discriminator_contract,
    save_contract,
)


def main():
    """Generate and save contracts."""
    contracts_dir = project_root / "configs" / "manifests"
    contracts_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GraspGen-s600 Contract Generation")
    print("=" * 60)

    # Generator contract
    print("\n[1/2] Generating Generator contract...")
    gen_contract = generate_generator_contract(
        batch_size=1,
        num_points=2048,
        num_grasps=20,
        grasp_repr="r3_6d",
        obs_backbone="pointnet",
        num_diffusion_steps=20,
    )
    save_contract(gen_contract, contracts_dir / "graspgen_generator.json")

    # Discriminator contract
    print("\n[2/2] Generating Discriminator contract...")
    disc_contract = generate_discriminator_contract(
        batch_size=1,
        num_points=2048,
        num_candidates=20,
        grasp_repr="r3_6d",
        obs_backbone="pointnet",
    )
    save_contract(disc_contract, contracts_dir / "graspgen_discriminator.json")

    print("\n" + "=" * 60)
    print("✅ Contracts generated successfully!")
    print("=" * 60)
    print(f"\nOutput directory: {contracts_dir}")
    print(f"  - graspgen_generator.json")
    print(f"  - graspgen_discriminator.json")
    print("\nNext steps:")
    print("  1. Clone upstream GraspGen: git clone https://github.com/NVlabs/GraspGen third_party/GraspGen")
    print("  2. Download pretrained weights")
    print("  3. Implement ONNX export in src/python/graspgen_s600_tools/export/")


if __name__ == "__main__":
    main()
