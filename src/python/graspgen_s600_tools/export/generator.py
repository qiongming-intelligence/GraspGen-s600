"""
ONNX export wrapper for GraspGen Generator.

This module provides a wrapper that makes the Generator model compatible
with ONNX export, handling the diffusion loop and ensuring fixed shapes.

Key challenge: The original Generator has a diffusion loop that's hard to export.
Strategy: Export the single-step denoising function, run the loop in Python.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn as nn

# Add upstream GraspGen to path
GRASPGEN_PATH = Path(__file__).parent.parent.parent.parent.parent / "third_party" / "GraspGen"
if str(GRASPGEN_PATH) not in sys.path:
    sys.path.insert(0, str(GRASPGEN_PATH))


class GraspGenGeneratorONNXWrapper(nn.Module):
    """
    ONNX-compatible wrapper for GraspGen Generator.

    This wrapper exports only the core inference computation with fixed shapes,
    avoiding dynamic control flow (diffusion loop).

    Strategy: Export the single denoising step. The full diffusion loop will be
    implemented in Python at runtime.

    Args:
        generator: Original GraspGenGenerator model
        num_points: Fixed number of input points
        num_grasps: Fixed number of grasps to generate
        output_dim: Grasp representation dimension (9 for r3_6d, 6 for r3_so3)
    """

    def __init__(
        self,
        generator: nn.Module,
        num_points: int = 2048,
        num_grasps: int = 20,
        output_dim: int = 6,
    ):
        super().__init__()
        self.generator = generator
        self.num_points = num_points
        self.num_grasps = num_grasps
        self.output_dim = output_dim

        # Extract components
        self.object_encoder = generator.object_encoder
        self.diffusion_head = generator.diffusion_head

        # Store essential attributes
        self.pose_repr = generator.pose_repr
        self.grasp_repr = generator.grasp_repr
        self.kappa = generator.kappa

    def forward(
        self,
        pc: torch.Tensor,
        noisy_grasps: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single-step denoising forward pass.

        This is the core computation that will be exported to ONNX.
        The diffusion loop will be handled externally in Python.

        Args:
            pc: Input point cloud (B, N, 3) - normalized to [-1, 1]
            noisy_grasps: Current noisy grasp predictions (B*K, D)
                where K=num_grasps, D=output_dim
            timestep: Current diffusion timestep (1,) - scalar tensor

        Returns:
            noise_pred: Predicted noise to remove (B*K, D)
        """
        batch_size = pc.shape[0]

        # Encode point cloud to get object features
        # Shape: (B, num_obs_dim)
        object_feat = self.object_encoder(pc)

        # Replicate object features for each grasp
        # Shape: (B*K, num_obs_dim)
        object_feat_expanded = object_feat.repeat_interleave(self.num_grasps, dim=0)

        # Predict noise using diffusion head. Signature matches upstream:
        # forward(observation_embedding, timesteps, sample)
        noise_pred = self.diffusion_head(
            object_feat_expanded,
            timestep,
            noisy_grasps,
        )

        return noise_pred


def export_generator_onnx(
    generator: nn.Module,
    output_path: str,
    num_points: int = 2048,
    num_grasps: int = 20,
    output_dim: int = 6,
    opset_version: int = 17,
    verbose: bool = True,
) -> None:
    """
    Export Generator to ONNX format.

    Args:
        generator: Loaded GraspGenGenerator model
        output_path: Path to save ONNX model
        num_points: Number of input points (fixed)
        num_grasps: Number of grasps per object (fixed)
        output_dim: Grasp representation dimension
        opset_version: ONNX opset version
        verbose: Print export information

    Example:
        >>> from graspgen_s600_tools.export.factories import load_generator
        >>> generator = load_generator("models/upstream/graspgen_franka_panda_gen.pth")
        >>> export_generator_onnx(generator, "models/onnx/graspgen_generator.onnx")
    """

    # Create wrapper
    wrapper = GraspGenGeneratorONNXWrapper(
        generator,
        num_points=num_points,
        num_grasps=num_grasps,
        output_dim=output_dim,
    )
    wrapper.eval()

    # Create dummy inputs
    batch_size = 1
    dummy_pc = torch.randn(batch_size, num_points, 3)
    dummy_noisy_grasps = torch.randn(batch_size * num_grasps, output_dim)
    dummy_timestep = torch.tensor([0], dtype=torch.long)

    if verbose:
        print(f"Exporting Generator to ONNX...")
        print(f"  Input shapes:")
        print(f"    - pc: {dummy_pc.shape}")
        print(f"    - noisy_grasps: {dummy_noisy_grasps.shape}")
        print(f"    - timestep: {dummy_timestep.shape}")

    # Export to ONNX
    torch.onnx.export(
        wrapper,
        (dummy_pc, dummy_noisy_grasps, dummy_timestep),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['pc', 'noisy_grasps', 'timestep'],
        output_names=['noise_pred'],
        dynamic_axes=None,  # Fixed shapes for BPU compatibility
        verbose=verbose,
    )

    if verbose:
        print(f"✓ Generator exported to {output_path}")

        # Verify ONNX model
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX model verified")


class SimplifiedGeneratorWrapper(nn.Module):
    """
    Simplified Generator wrapper that exports the full inference pipeline.

    This version exports the complete forward pass including object encoding
    and multiple denoising steps unrolled. Use this if the single-step approach
    doesn't work with hb_compile.

    NOTE: This will be larger and less flexible than the single-step version.
    """

    def __init__(
        self,
        generator: nn.Module,
        num_diffusion_steps: int = 20,
        num_points: int = 2048,
        num_grasps: int = 20,
    ):
        super().__init__()
        self.generator = generator
        self.num_diffusion_steps = num_diffusion_steps
        self.num_points = num_points
        self.num_grasps = num_grasps

    def forward(self, pc: torch.Tensor) -> torch.Tensor:
        """
        Full inference forward pass with unrolled diffusion.

        Args:
            pc: Input point cloud (1, N, 3)

        Returns:
            grasps_pred: Predicted grasps (1, K, D)
        """
        # This would require unrolling the full diffusion loop
        # which makes the ONNX model very large (20+ steps)
        # Keeping this as a fallback option
        raise NotImplementedError("Full unrolled export not yet implemented")


if __name__ == "__main__":
    import argparse
    from .factories import load_generator

    parser = argparse.ArgumentParser(description="Export Generator to ONNX")
    parser.add_argument("checkpoint", type=str, help="Path to generator checkpoint")
    parser.add_argument("--output", type=str, default="models/onnx/graspgen_generator.onnx",
                       help="Output ONNX path")
    parser.add_argument("--config", type=str, help="Config YAML path")
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--num-grasps", type=int, default=20)
    parser.add_argument("--output-dim", type=int, default=6,
                       help="9 for r3_6d, 6 for r3_so3")

    args = parser.parse_args()

    # Load model
    print(f"Loading generator from {args.checkpoint}...")
    generator = load_generator(args.checkpoint, args.config)

    # Export
    export_generator_onnx(
        generator,
        args.output,
        num_points=args.num_points,
        num_grasps=args.num_grasps,
        output_dim=args.output_dim,
    )
