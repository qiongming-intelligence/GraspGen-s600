"""
ONNX-compatible Diffusion Noise Prediction Network for GraspGen.

This module ports the DiffusionNoisePredictionNet from the upstream GraspGen
(grasp_gen/models/generator.py) into a self-contained, ONNX-exportable form.

The diffusion head predicts the noise present in a noisy grasp sample, given:
  - an object observation embedding (from the PointNet encoder)
  - the current diffusion timestep
  - the current noisy grasp sample

It is backbone-agnostic: it only consumes the observation embedding vector, so
it works identically with the original PTV3 encoder or our PointNet++ encoder.

ONNX compatibility notes:
- Uses only standard ops (Linear, ReLU, Mish, sin/cos, cat).
- The "cat" pose representation (concatenate embeddings + MLP) is fully static.
- The optional transformer attention path from upstream is intentionally NOT
  ported here; the pretrained Franka model uses `attention=cat_attn`, but for the
  S600 deployment we target the static MLP head ("cat") which is BPU-friendly.
"""

import math
import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embedding for diffusion timesteps.

    Ported verbatim (numerically) from upstream grasp_gen.models.model_utils.
    Fully ONNX-compatible (uses exp/sin/cos on a static arange).

    Args:
        dim: embedding dimension
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        batch_size = x.shape[0]
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        if len(x.shape) == 1:
            emb = x[:, None] * emb[None, :]
        else:
            emb = x[:, :, None] * emb[None, None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        emb = emb.reshape([batch_size, -1])
        return emb


class DiffusionHead(nn.Module):
    """
    ONNX-compatible diffusion noise prediction network (MLP / "cat" variant).

    Architecture (matches upstream DiffusionNoisePredictionNet with pose_repr="mlp"
    and the non-attention "cat" path):

        timestep --> SinusoidalPosEmb --> Linear --> Mish --> Linear  (step embed)
        sample   --> Linear --> ReLU --> Linear                       (sample embed)
        embed = cat([sample_embed, step_embed, obs_embed])
        noise_pred = prediction_head(embed)

    Args:
        diffusion_step_embed_dim: dim of timestep embedding. Default: 512
        observation_embed_dim: dim of object observation embedding. Default: 512
        sample_embed_dim: dim of the sample embedding. Default: 512
        sample_dim: dim of grasp representation (9 for r3_6d, 6 for r3_so3). Default: 9
    """

    def __init__(
        self,
        diffusion_step_embed_dim: int = 512,
        observation_embed_dim: int = 512,
        sample_embed_dim: int = 512,
        sample_dim: int = 9,
    ):
        super().__init__()
        self.sample_dim = sample_dim

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        self.sample_encoder = nn.Sequential(
            nn.Linear(sample_dim, sample_embed_dim),
            nn.ReLU(),
            nn.Linear(sample_embed_dim, sample_embed_dim),
        )

        total_input_dim = (
            sample_embed_dim + diffusion_step_embed_dim + observation_embed_dim
        )

        self.prediction_head = nn.Sequential(
            nn.Linear(total_input_dim, total_input_dim // 2),
            nn.ReLU(),
            nn.Linear(total_input_dim // 2, total_input_dim // 4),
            nn.ReLU(),
            nn.Linear(total_input_dim // 4, sample_dim),
        )

    def forward(
        self,
        observation_embedding: torch.Tensor,
        timesteps: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict the noise in a noisy grasp sample.

        Args:
            observation_embedding: (B, observation_embed_dim) object features
            timesteps: (B,) diffusion timesteps (float or long)
            sample: (B, sample_dim) current noisy grasp

        Returns:
            noise_pred: (B, sample_dim) predicted noise
        """
        device = observation_embedding.device

        # Broadcast a scalar timestep to the batch (ONNX-friendly, static at export).
        if torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(device)
            timesteps = timesteps.expand(observation_embedding.shape[0])

        timestep_embedding = self.diffusion_step_encoder(timesteps)
        sample_embedding = self.sample_encoder(sample)

        embed = torch.cat(
            [sample_embedding, timestep_embedding, observation_embedding],
            dim=-1,
        )

        return self.prediction_head(embed)


if __name__ == "__main__":
    print("Testing DiffusionHead...")

    B = 20  # batch = num_grasps for a single object
    obs_dim = 512
    sample_dim = 9

    head = DiffusionHead(
        observation_embed_dim=obs_dim,
        sample_dim=sample_dim,
    )
    head.eval()

    obs = torch.randn(B, obs_dim)
    timestep = torch.tensor([5], dtype=torch.long)  # scalar-ish
    sample = torch.randn(B, sample_dim)

    with torch.no_grad():
        noise_pred = head(obs, timestep, sample)

    print(f"✓ obs:        {obs.shape}")
    print(f"✓ timestep:   {timestep.shape}")
    print(f"✓ sample:     {sample.shape}")
    print(f"✓ noise_pred: {noise_pred.shape}")
    assert noise_pred.shape == (B, sample_dim), "Output shape mismatch!"
    print("\n✓ DiffusionHead test passed!")
