"""
Model loading utilities for GraspGen models.

This module provides factory functions to load pretrained GraspGen models
from checkpoints, supporting both Generator and Discriminator.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import torch

# Add upstream GraspGen to path
GRASPGEN_PATH = Path(__file__).parent.parent.parent.parent.parent / "third_party" / "GraspGen"
if str(GRASPGEN_PATH) not in sys.path:
    sys.path.insert(0, str(GRASPGEN_PATH))


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration YAML file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_generator(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    device: str = "cpu",
) -> torch.nn.Module:
    """
    Load GraspGen Generator model from checkpoint.

    Args:
        checkpoint_path: Path to generator checkpoint (.pth)
        config_path: Optional path to config YAML. If None, uses default config.
        device: Device to load model on ("cpu" or "cuda")

    Returns:
        Generator model in eval mode

    Example:
        >>> generator = load_generator("models/upstream/graspgen_franka_panda_gen.pth")
        >>> generator.eval()
    """
    from grasp_gen.models.generator import GraspGenGenerator

    # Load config if provided
    if config_path:
        config = load_config(config_path)
        model_config = config['diffusion']
    else:
        # Default config based on franka_panda checkpoint
        model_config = {
            'num_embed_dim': 256,
            'num_obs_dim': 512,
            'diffusion_embed_dim': 512,
            'image_size': 256,
            'num_diffusion_iters': 10,  # Training value
            'num_diffusion_iters_eval': 10,  # Can be reduced to 5-20 for speed
            'obs_backbone': 'ptv3',  # Original uses ptv3, we'll adapt to pointnet
            'compositional_schedular': True,
            'loss_pointmatching': False,
            'loss_l1_pos': True,
            'loss_l1_rot': True,
            'grasp_repr': 'r3_so3',  # Original uses r3_so3, we may use r3_6d
            'kappa': 3.27,
            'clip_sample': True,
            'beta_schedule': 'squaredcos_cap_v2',
            'attention': 'cat_attn',
            'gripper_name': 'franka_panda',
            'pose_repr': 'mlp',
            'num_grasps_per_object': 500,
        }

    # Create model
    model = GraspGenGenerator(**model_config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Load weights
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    print(f"✓ Generator loaded from {checkpoint_path}")
    print(f"  - Backbone: {model_config['obs_backbone']}")
    print(f"  - Grasp repr: {model_config['grasp_repr']}")
    print(f"  - Diffusion steps (eval): {model_config['num_diffusion_iters_eval']}")

    return model


def load_discriminator(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    device: str = "cpu",
) -> torch.nn.Module:
    """
    Load GraspGen Discriminator model from checkpoint.

    Args:
        checkpoint_path: Path to discriminator checkpoint (.pth)
        config_path: Optional path to config YAML
        device: Device to load model on

    Returns:
        Discriminator model in eval mode
    """
    from grasp_gen.models.discriminator import GraspGenDiscriminator

    # Load config if provided
    if config_path:
        config = load_config(config_path)
        model_config = config['discriminator']
    else:
        # Default config
        model_config = {
            'obs_backbone': 'ptv3',
            'num_embed_dim': 256,
            'num_obs_dim': 512,
            'grasp_repr': 'r3_so3',
            'pose_repr': 'mlp',
            'topk_ratio': 0.75,
            'kappa': 3.27,
            'gripper_name': 'franka_panda',
        }

    # Create model
    model = GraspGenDiscriminator(**model_config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Load weights
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    print(f"✓ Discriminator loaded from {checkpoint_path}")
    print(f"  - Backbone: {model_config['obs_backbone']}")
    print(f"  - Grasp repr: {model_config['grasp_repr']}")

    return model


def get_model_info(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Get information about a loaded model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_type': type(model).__name__,
    }


if __name__ == "__main__":
    # Test model loading
    import argparse

    parser = argparse.ArgumentParser(description="Test GraspGen model loading")
    parser.add_argument("--generator", type=str, help="Path to generator checkpoint")
    parser.add_argument("--discriminator", type=str, help="Path to discriminator checkpoint")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    args = parser.parse_args()

    if args.generator:
        print("Loading Generator...")
        gen = load_generator(args.generator, args.config)
        info = get_model_info(gen)
        print(f"Generator info: {info}")

    if args.discriminator:
        print("\nLoading Discriminator...")
        disc = load_discriminator(args.discriminator, args.config)
        info = get_model_info(disc)
        print(f"Discriminator info: {info}")
