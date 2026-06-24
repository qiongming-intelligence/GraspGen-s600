"""
ONNX export wrapper for GraspGen Discriminator.

The Discriminator is simpler than the Generator - it's a single forward pass
that scores grasp candidates.
"""

import sys
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn

# Add upstream GraspGen to path
GRASPGEN_PATH = Path(__file__).parent.parent.parent.parent.parent / "third_party" / "GraspGen"
if str(GRASPGEN_PATH) not in sys.path:
    sys.path.insert(0, str(GRASPGEN_PATH))


class GraspGenDiscriminatorONNXWrapper(nn.Module):
    """
    ONNX-compatible wrapper for GraspGen Discriminator.

    The Discriminator evaluates the quality of grasp candidates.
    It's a straightforward forward pass without loops, making it easier to export.

    Args:
        discriminator: Original GraspGenDiscriminator model
        num_points: Fixed number of input points
        num_candidates: Fixed number of grasp candidates to evaluate
        grasp_dim: Grasp representation dimension (9 for r3_6d, 6 for r3_so3)
    """

    def __init__(
        self,
        discriminator: nn.Module,
        num_points: int = 2048,
        num_candidates: int = 20,
        grasp_dim: int = 6,
    ):
        super().__init__()
        self.discriminator = discriminator
        self.num_points = num_points
        self.num_candidates = num_candidates
        self.grasp_dim = grasp_dim

        # Extract core components
        self.object_encoder = discriminator.object_encoder
        self.grasp_scorer = discriminator.grasp_scorer  # or whatever the scoring head is called

        # Store attributes
        self.pose_repr = discriminator.pose_repr
        self.grasp_repr = discriminator.grasp_repr
        self.kappa = discriminator.kappa

    def forward(
        self,
        pc: torch.Tensor,
        grasps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass to score grasp candidates.

        Args:
            pc: Input point cloud (B, N, 3) - normalized to [-1, 1]
            grasps: Grasp candidates to score (B, K, D)
                where K=num_candidates, D=grasp_dim

        Returns:
            scores: Quality scores for each grasp (B, K)
                Range [0, 1] where higher is better
        """
        batch_size = pc.shape[0]

        # Encode point cloud
        # Shape: (B, num_obs_dim)
        object_feat = self.object_encoder(pc)

        # Flatten grasps for processing
        # Shape: (B*K, D)
        grasps_flat = grasps.reshape(-1, self.grasp_dim)

        # Replicate object features for each grasp
        # Shape: (B*K, num_obs_dim)
        object_feat_expanded = object_feat.repeat_interleave(self.num_candidates, dim=0)

        # Score each grasp given the object context
        # Shape: (B*K, 1)
        scores_flat = self.grasp_scorer(grasps_flat, object_feat_expanded)

        # Reshape back to (B, K)
        scores = scores_flat.reshape(batch_size, self.num_candidates)

        return scores


def export_discriminator_onnx(
    discriminator: nn.Module,
    output_path: str,
    num_points: int = 2048,
    num_candidates: int = 20,
    grasp_dim: int = 6,
    opset_version: int = 17,
    verbose: bool = True,
) -> None:
    """
    Export Discriminator to ONNX format.

    Args:
        discriminator: Loaded GraspGenDiscriminator model
        output_path: Path to save ONNX model
        num_points: Number of input points (fixed)
        num_candidates: Number of grasp candidates to score (fixed)
        grasp_dim: Grasp representation dimension
        opset_version: ONNX opset version
        verbose: Print export information

    Example:
        >>> from graspgen_s600_tools.export.factories import load_discriminator
        >>> disc = load_discriminator("models/upstream/graspgen_franka_panda_dis.pth")
        >>> export_discriminator_onnx(disc, "models/onnx/graspgen_discriminator.onnx")
    """

    # Create wrapper
    wrapper = GraspGenDiscriminatorONNXWrapper(
        discriminator,
        num_points=num_points,
        num_candidates=num_candidates,
        grasp_dim=grasp_dim,
    )
    wrapper.eval()

    # Create dummy inputs
    batch_size = 1
    dummy_pc = torch.randn(batch_size, num_points, 3)
    dummy_grasps = torch.randn(batch_size, num_candidates, grasp_dim)

    if verbose:
        print(f"Exporting Discriminator to ONNX...")
        print(f"  Input shapes:")
        print(f"    - pc: {dummy_pc.shape}")
        print(f"    - grasps: {dummy_grasps.shape}")

    # Export to ONNX
    torch.onnx.export(
        wrapper,
        (dummy_pc, dummy_grasps),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['pc', 'grasps'],
        output_names=['scores'],
        dynamic_axes=None,  # Fixed shapes for BPU compatibility
        verbose=verbose,
    )

    if verbose:
        print(f"✓ Discriminator exported to {output_path}")

        # Verify ONNX model
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX model verified")


if __name__ == "__main__":
    import argparse
    from .factories import load_discriminator

    parser = argparse.ArgumentParser(description="Export Discriminator to ONNX")
    parser.add_argument("checkpoint", type=str, help="Path to discriminator checkpoint")
    parser.add_argument("--output", type=str, default="models/onnx/graspgen_discriminator.onnx",
                       help="Output ONNX path")
    parser.add_argument("--config", type=str, help="Config YAML path")
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--num-candidates", type=int, default=20)
    parser.add_argument("--grasp-dim", type=int, default=6,
                       help="9 for r3_6d, 6 for r3_so3")

    args = parser.parse_args()

    # Load model
    print(f"Loading discriminator from {args.checkpoint}...")
    discriminator = load_discriminator(args.checkpoint, args.config)

    # Export
    export_discriminator_onnx(
        discriminator,
        args.output,
        num_points=args.num_points,
        num_candidates=args.num_candidates,
        grasp_dim=args.grasp_dim,
    )
