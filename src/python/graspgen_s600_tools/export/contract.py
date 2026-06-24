"""
Contract generation for GraspGen models.

Contracts define the fixed input/output shapes and configurations
for ONNX export and HBM compilation.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Literal


def generate_generator_contract(
    batch_size: int = 1,
    num_points: int = 2048,
    num_grasps: int = 20,
    grasp_repr: Literal["r3_6d", "r3_so3", "euler"] = "r3_so3",
    obs_backbone: Literal["pointnet", "pointnet2", "vit"] = "pointnet",
    num_diffusion_steps: int = 20,
) -> Dict:
    """
    Generate contract for GraspGen Generator model.

    Args:
        batch_size: Batch size (fixed for BPU)
        num_points: Number of points in input point cloud
        num_grasps: Number of grasps to generate
        grasp_repr: Grasp representation ("r3_6d" = position + 6D rotation, "r3_so3"/"euler" = position + 3D rotation)
        obs_backbone: Observation encoder backbone
        num_diffusion_steps: Number of diffusion denoising steps (reduced from 100 for speed)

    Returns:
        Contract dictionary
    """
    output_dim = 9 if grasp_repr == "r3_6d" else 6

    contract = {
        "name": "graspgen_generator",
        "description": f"GraspGen Generator with {obs_backbone} backbone",
        "onnx_path": f"models/onnx/graspgen_generator_{obs_backbone}.onnx",
        "hbm_name": f"graspgen_generator_{obs_backbone}.hbm",
        "inputs": [
            {
                "name": "pc",
                "description": "Input point cloud (normalized)",
                "shape": ["B", "N", "3"],
                "concrete_shape": [batch_size, num_points, 3],
                "dtype": "float32",
                "range": [-1.0, 1.0],  # Expected after normalization
            },
            {
                "name": "noisy_grasps",
                "description": "Current noisy grasp samples",
                "shape": ["B*K", "D"],
                "concrete_shape": [batch_size * num_grasps, output_dim],
                "dtype": "float32",
            },
            {
                "name": "timestep",
                "description": "Current diffusion timestep",
                "shape": ["1"],
                "concrete_shape": [1],
                "dtype": "int64",
            },
        ],
        "outputs": [
            {
                "name": "noise_pred",
                "description": "Predicted noise for the denoising step",
                "shape": ["B*K", "D"],
                "concrete_shape": [batch_size * num_grasps, output_dim],
                "dtype": "float32",
            }
        ],
        "config": {
            "num_diffusion_iters_eval": num_diffusion_steps,
            "obs_backbone": obs_backbone,
            "grasp_repr": grasp_repr,
            "batch_size": batch_size,
        },
        "metadata": {
            "target_platform": "horizon_s600",
            "optimization_hints": {
                "core_num": 2,
                "optimize_level": "O2",
                "quantization": "int16",
                "calibration_type": "max",
            }
        }
    }
    return contract


def generate_discriminator_contract(
    batch_size: int = 1,
    num_points: int = 2048,
    num_candidates: int = 20,
    grasp_repr: Literal["r3_6d", "r3_so3", "euler"] = "r3_so3",
    obs_backbone: Literal["pointnet", "pointnet2"] = "pointnet",
) -> Dict:
    """
    Generate contract for GraspGen Discriminator model.

    Args:
        batch_size: Batch size (fixed for BPU)
        num_points: Number of points in input point cloud
        num_candidates: Number of grasp candidates to score
        grasp_repr: Grasp representation
        obs_backbone: Observation encoder backbone

    Returns:
        Contract dictionary
    """
    grasp_dim = 9 if grasp_repr == "r3_6d" else 6

    contract = {
        "name": "graspgen_discriminator",
        "description": f"GraspGen Discriminator with {obs_backbone} backbone",
        "onnx_path": f"models/onnx/graspgen_discriminator_{obs_backbone}.onnx",
        "hbm_name": f"graspgen_discriminator_{obs_backbone}.hbm",
        "inputs": [
            {
                "name": "pc",
                "description": "Input point cloud (normalized)",
                "concrete_shape": [batch_size, num_points, 3],
                "dtype": "float32",
                "range": [-1.0, 1.0],
            },
            {
                "name": "grasps",
                "description": "Grasp candidates to score",
                "concrete_shape": [batch_size, num_candidates, grasp_dim],
                "dtype": "float32",
            }
        ],
        "outputs": [
            {
                "name": "scores",
                "description": "Quality scores for each grasp",
                "concrete_shape": [batch_size, num_candidates],
                "dtype": "float32",
                "range": [0.0, 1.0],  # Sigmoid output
            }
        ],
        "config": {
            "obs_backbone": obs_backbone,
            "grasp_repr": grasp_repr,
            "batch_size": batch_size,
        },
        "metadata": {
            "target_platform": "horizon_s600",
            "optimization_hints": {
                "core_num": 1,  # Lighter model, single core sufficient
                "optimize_level": "O2",
                "quantization": "int16",
                "calibration_type": "max",
            }
        }
    }
    return contract


def save_contract(contract: Dict, output_path: Path) -> None:
    """Save contract to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(contract, f, indent=2)
    print(f"Contract saved to: {output_path}")


def load_contract(contract_path: Path) -> Dict:
    """Load contract from JSON file."""
    with open(contract_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    # Generate default contracts
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent.parent.parent
    contracts_dir = project_root / "configs" / "manifests"

    # Generator contract
    gen_contract = generate_generator_contract()
    save_contract(gen_contract, contracts_dir / "graspgen_generator.json")

    # Discriminator contract
    disc_contract = generate_discriminator_contract()
    save_contract(disc_contract, contracts_dir / "graspgen_discriminator.json")

    print("\n✅ Contracts generated successfully!")
    print(f"   Generator: {contracts_dir / 'graspgen_generator.json'}")
    print(f"   Discriminator: {contracts_dir / 'graspgen_discriminator.json'}")
