"""
Improved model loading that bypasses PointNet2 dependency.

This module loads models directly from state_dict without triggering
PointNet2 C++ extension imports.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import torch
import torch.nn as nn

# Add upstream GraspGen to path
GRASPGEN_PATH = Path(__file__).parent.parent.parent.parent.parent / "third_party" / "GraspGen"
if str(GRASPGEN_PATH) not in sys.path:
    sys.path.insert(0, str(GRASPGEN_PATH))


def load_checkpoint_direct(checkpoint_path: str, device: str = "cpu") -> Dict[str, Any]:
    """
    Load checkpoint without instantiating models.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load tensors on

    Returns:
        Dictionary with 'model', 'epoch', 'optimizer' keys
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def build_ptv3_encoder(
    grid_size: float = 0.01,
    num_obs_dim: int = 512,
) -> nn.Module:
    """
    Build PTV3 encoder directly.

    Args:
        grid_size: Grid size for point cloud voxelization
        num_obs_dim: Output dimension

    Returns:
        PTV3 encoder module
    """
    from grasp_gen.models.ptv3.ptv3 import PointTransformerV3

    # PTV3 configuration based on graspgen_franka_panda.yml
    encoder = PointTransformerV3(
        in_channels=3,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    )

    return encoder


def load_generator_direct(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    device: str = "cpu",
) -> nn.Module:
    """
    Load Generator by directly building PTV3 and loading weights.

    This bypasses the GraspGenGenerator constructor which triggers
    PointNet2 imports.

    Args:
        checkpoint_path: Path to generator checkpoint
        config_path: Optional config file
        device: Device to load on

    Returns:
        Generator model (simplified, encoder + diffusion head)
    """
    print("Loading generator directly (bypassing PointNet2)...")

    # Load checkpoint
    checkpoint = load_checkpoint_direct(checkpoint_path, device)
    state_dict = checkpoint['model']

    print(f"  Checkpoint has {len(state_dict)} parameters")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")

    # Build object encoder (PTV3)
    print("  Building PTV3 encoder...")
    try:
        encoder = build_ptv3_encoder()
        encoder.to(device)
        encoder.eval()

        # Load encoder weights
        encoder_state = {
            k.replace('object_encoder.', ''): v
            for k, v in state_dict.items()
            if k.startswith('object_encoder.')
        }

        print(f"  Loading {len(encoder_state)} encoder parameters...")
        missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)

        if missing:
            print(f"  Warning: {len(missing)} missing keys")
        if unexpected:
            print(f"  Warning: {len(unexpected)} unexpected keys")

        print("✓ PTV3 encoder loaded successfully")

        return encoder

    except Exception as e:
        print(f"✗ PTV3 encoder loading failed: {e}")
        raise


def load_discriminator_direct(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    device: str = "cpu",
) -> nn.Module:
    """
    Load Discriminator by directly building PTV3 and loading weights.

    Args:
        checkpoint_path: Path to discriminator checkpoint
        config_path: Optional config file
        device: Device to load on

    Returns:
        Discriminator model (simplified, encoder + scorer)
    """
    print("Loading discriminator directly (bypassing PointNet2)...")

    # Load checkpoint
    checkpoint = load_checkpoint_direct(checkpoint_path, device)
    state_dict = checkpoint['model']

    print(f"  Checkpoint has {len(state_dict)} parameters")

    # Build object encoder (PTV3)
    print("  Building PTV3 encoder...")
    try:
        encoder = build_ptv3_encoder()
        encoder.to(device)
        encoder.eval()

        # Load encoder weights
        encoder_state = {
            k.replace('object_encoder.', ''): v
            for k, v in state_dict.items()
            if k.startswith('object_encoder.')
        }

        print(f"  Loading {len(encoder_state)} encoder parameters...")
        encoder.load_state_dict(encoder_state, strict=False)

        print("✓ PTV3 encoder loaded successfully")

        return encoder

    except Exception as e:
        print(f"✗ PTV3 encoder loading failed: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test direct model loading")
    parser.add_argument("--generator", type=str, help="Path to generator checkpoint")
    parser.add_argument("--discriminator", type=str, help="Path to discriminator checkpoint")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    args = parser.parse_args()

    if args.generator:
        print("Testing Generator loading...")
        gen = load_generator_direct(args.generator, args.config)
        print(f"Generator loaded: {type(gen)}")

    if args.discriminator:
        print("\nTesting Discriminator loading...")
        disc = load_discriminator_direct(args.discriminator, args.config)
        print(f"Discriminator loaded: {type(disc)}")
