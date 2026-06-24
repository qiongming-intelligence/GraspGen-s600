"""
ONNX-compatible GraspGen Generator and Discriminator.

This module assembles the full S600-deployable GraspGen models from the
ONNX-friendly building blocks implemented in this package:

  - PointNetUpstream (models/pointnet_upstream.py)  -- weight-compatible PointNet++
  - DiffusionHead    (models/diffusion_head.py)     -- noise prediction net

Two deployable graphs are provided:

  GraspGenGeneratorONNX
      Single-step denoiser. Inputs (pc, noisy_grasps, timestep) -> noise_pred.
      The iterative DDPM sampling loop runs in Python at runtime (see
      scripts/test_graspgen_onnx.py), calling this graph once per timestep.
      This keeps the exported graph free of dynamic control flow, which is
      required for the Horizon S600 BPU.

  GraspGenDiscriminatorONNX
      Grasp quality scorer. Inputs (pc, grasps) -> per-grasp scores.

Design choices for ONNX/BPU compatibility:
  - Fixed batch size (B=1 object) and fixed num_grasps (K).
  - The point cloud is encoded once; object features are repeated per grasp.
  - grasp_repr = "r3_so3" => sample_dim = 6 (matches upstream Robotiq).
"""

from pathlib import Path

import torch
import torch.nn as nn

from .pointnet_upstream import PointNetUpstream
from .diffusion_head import DiffusionHead


# Grasp representation -> sample dimension
GRASP_REPR_DIM = {
    "r3_6d": 9,    # 3 translation + 6 rotation (two columns of rotation matrix)
    "r3_so3": 6,   # 3 translation + 3 axis-angle
    "r3_euler": 6,
}


def _load_checkpoint_state_dict(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    """Load a PyTorch checkpoint and return its model state dict."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint["model"] if "model" in checkpoint else checkpoint


def _load_prefixed_state_dict(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    checkpoint_path: str | Path,
    strict: bool,
) -> None:
    """Load a prefixed subset of an upstream state dict into a module."""
    module_state = {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not module_state:
        raise ValueError(f"No {prefix!r} weights found in {checkpoint_path}")
    module.load_state_dict(module_state, strict=strict)


def load_upstream_object_encoder_weights(
    model: nn.Module,
    checkpoint_path: str | Path,
    strict: bool = True,
) -> None:
    """Load upstream Robotiq object_encoder weights into a model's PointNetUpstream."""
    state_dict = _load_checkpoint_state_dict(checkpoint_path)
    if not hasattr(model, "object_encoder"):
        raise AttributeError("model has no object_encoder attribute")
    _load_prefixed_state_dict(
        model.object_encoder,
        state_dict,
        "object_encoder.",
        checkpoint_path,
        strict,
    )


def load_upstream_generator_weights(
    model: nn.Module,
    checkpoint_path: str | Path,
    strict: bool = True,
) -> None:
    """Load upstream PointNet GraspGen generator weights into the ONNX model."""
    state_dict = _load_checkpoint_state_dict(checkpoint_path)
    if not hasattr(model, "object_encoder"):
        raise AttributeError("model has no object_encoder attribute")
    if not hasattr(model, "diffusion_head"):
        raise AttributeError("model has no diffusion_head attribute")

    _load_prefixed_state_dict(
        model.object_encoder,
        state_dict,
        "object_encoder.",
        checkpoint_path,
        strict,
    )
    _load_prefixed_state_dict(
        model.diffusion_head,
        state_dict,
        "diffusion_head.",
        checkpoint_path,
        strict,
    )


def load_upstream_discriminator_weights(
    model: nn.Module,
    checkpoint_path: str | Path,
    strict: bool = True,
) -> None:
    """Load upstream PointNet GraspGen discriminator weights into the ONNX model."""
    state_dict = _load_checkpoint_state_dict(checkpoint_path)
    for module_name in ("object_encoder", "sample_encoder", "prediction_head"):
        if not hasattr(model, module_name):
            raise AttributeError(f"model has no {module_name} attribute")
        _load_prefixed_state_dict(
            getattr(model, module_name),
            state_dict,
            f"{module_name}.",
            checkpoint_path,
            strict,
        )


class GraspGenGeneratorONNX(nn.Module):
    """
    ONNX-exportable single-step grasp denoiser.

    Combines a PointNet++ encoder with the diffusion noise-prediction head.
    The point cloud is encoded once and broadcast across the K grasps so a
    single object embedding drives all grasp denoising in one forward pass.

    Args:
        num_obs_dim: object embedding dimension. Default: 512
        diffusion_embed_dim: timestep/sample embedding dim. Default: 512
        grasp_repr: grasp representation. Default: "r3_so3" (sample_dim=6)
        num_grasps: number of grasps generated per object (K). Default: 20
        encoder_sampling: PointNetUpstream centroid sampling strategy. Default: "fps"
    """

    def __init__(
        self,
        num_obs_dim: int = 512,
        diffusion_embed_dim: int = 512,
        grasp_repr: str = "r3_so3",
        num_grasps: int = 20,
        encoder_sampling: str = "fps",
        attention: str = "cat_attn",
    ):
        super().__init__()
        if grasp_repr not in GRASP_REPR_DIM:
            raise NotImplementedError(f"grasp_repr {grasp_repr} not supported")

        self.num_obs_dim = num_obs_dim
        self.grasp_repr = grasp_repr
        self.sample_dim = GRASP_REPR_DIM[grasp_repr]
        self.num_grasps = num_grasps

        self.object_encoder = PointNetUpstream(
            output_embedding_dim=num_obs_dim,
            feature_dim=-1,
            sampling=encoder_sampling,
        )
        self.diffusion_head = DiffusionHead(
            diffusion_step_embed_dim=diffusion_embed_dim,
            observation_embed_dim=num_obs_dim,
            sample_embed_dim=diffusion_embed_dim,
            sample_dim=self.sample_dim,
            attention=attention,
        )

    def forward(
        self,
        pc: torch.Tensor,
        noisy_grasps: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        One reverse-diffusion step: predict the noise to remove.

        Args:
            pc: (B, N, 3) normalized point cloud (B=1 for deployment)
            noisy_grasps: (B*K, sample_dim) current noisy grasps
            timestep: (1,) current diffusion timestep

        Returns:
            noise_pred: (B*K, sample_dim) predicted noise
        """
        # Encode point cloud once: (B, num_obs_dim)
        object_feat = self.object_encoder(pc)

        # Broadcast object features across the K grasps: (B*K, num_obs_dim)
        object_feat = object_feat.repeat_interleave(self.num_grasps, dim=0)

        # Predict noise for every grasp in parallel.
        noise_pred = self.diffusion_head(object_feat, timestep, noisy_grasps)
        return noise_pred


class GraspGenDiscriminatorONNX(nn.Module):
    """
    ONNX-exportable grasp quality discriminator.

    Scores each candidate grasp against the observed point cloud. Mirrors the
    upstream GraspGenDiscriminator structure (pose_repr="mlp"): encode the point
    cloud, encode each grasp, concatenate, and regress a per-grasp logit.

    Args:
        num_obs_dim: object embedding dimension. Default: 512
        sample_embed_dim: grasp embedding dimension. Default: 512
        grasp_repr: grasp representation. Default: "r3_so3" (sample_dim=6)
        num_grasps: number of candidate grasps (K). Default: 20
        apply_sigmoid: if True, output probabilities in [0, 1]. Default: True
        encoder_sampling: PointNetUpstream centroid sampling strategy. Default: "fps"
    """

    def __init__(
        self,
        num_obs_dim: int = 512,
        sample_embed_dim: int = 512,
        grasp_repr: str = "r3_so3",
        num_grasps: int = 20,
        apply_sigmoid: bool = True,
        encoder_sampling: str = "fps",
    ):
        super().__init__()
        if grasp_repr not in GRASP_REPR_DIM:
            raise NotImplementedError(f"grasp_repr {grasp_repr} not supported")

        self.num_obs_dim = num_obs_dim
        self.grasp_repr = grasp_repr
        self.sample_dim = GRASP_REPR_DIM[grasp_repr]
        self.num_grasps = num_grasps
        self.apply_sigmoid = apply_sigmoid

        self.object_encoder = PointNetUpstream(
            output_embedding_dim=num_obs_dim,
            feature_dim=-1,
            sampling=encoder_sampling,
        )

        self.sample_encoder = nn.Sequential(
            nn.Linear(self.sample_dim, sample_embed_dim),
            nn.ReLU(),
            nn.Linear(sample_embed_dim, sample_embed_dim),
        )

        total_input_dim = sample_embed_dim + num_obs_dim
        self.prediction_head = nn.Sequential(
            nn.Linear(total_input_dim, total_input_dim // 2),
            nn.ReLU(),
            nn.Linear(total_input_dim // 2, total_input_dim // 4),
            nn.ReLU(),
            nn.Linear(total_input_dim // 4, 1),
        )

    def forward(self, pc: torch.Tensor, grasps: torch.Tensor) -> torch.Tensor:
        """
        Score each candidate grasp.

        Args:
            pc: (B, N, 3) normalized point cloud (B=1 for deployment)
            grasps: (B, K, sample_dim) candidate grasps

        Returns:
            scores: (B, K) quality score per grasp (probabilities if apply_sigmoid)
        """
        B, K, D = grasps.shape

        # Encode point cloud once: (B, num_obs_dim)
        object_feat = self.object_encoder(pc)
        # Broadcast across K grasps: (B*K, num_obs_dim)
        object_feat = object_feat.repeat_interleave(K, dim=0)

        # Encode grasps: (B*K, sample_embed_dim)
        grasps_flat = grasps.reshape(B * K, D)
        sample_feat = self.sample_encoder(grasps_flat)

        # Concatenate and regress logits: (B*K, 1)
        embed = torch.cat([sample_feat, object_feat], dim=-1)
        logits = self.prediction_head(embed)

        # Reshape to (B, K)
        logits = logits.reshape(B, K)
        if self.apply_sigmoid:
            return torch.sigmoid(logits)
        return logits


if __name__ == "__main__":
    print("Testing GraspGen ONNX models...")

    num_points = 2048
    num_grasps = 20
    sample_dim = GRASP_REPR_DIM["r3_so3"]

    # Generator
    gen = GraspGenGeneratorONNX(num_grasps=num_grasps, attention="cat_attn")
    gen.eval()
    pc = torch.randn(1, num_points, 3)
    noisy = torch.randn(num_grasps, sample_dim)
    t = torch.tensor([5], dtype=torch.long)
    with torch.no_grad():
        noise_pred = gen(pc, noisy, t)
    print(f"✓ Generator noise_pred: {noise_pred.shape}")
    assert noise_pred.shape == (num_grasps, sample_dim)

    # Discriminator
    disc = GraspGenDiscriminatorONNX(num_grasps=num_grasps)
    disc.eval()
    grasps = torch.randn(1, num_grasps, sample_dim)
    with torch.no_grad():
        scores = disc(pc, grasps)
    print(f"✓ Discriminator scores: {scores.shape}")
    assert scores.shape == (1, num_grasps)
    assert (scores >= 0).all() and (scores <= 1).all()

    print("\n✓ GraspGen ONNX models test passed!")
