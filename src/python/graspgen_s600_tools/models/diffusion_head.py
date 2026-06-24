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
- Uses only standard ops (Linear, ReLU, Mish, GELU, LayerNorm, sin/cos, cat).
- The "cat" pose representation (concatenate embeddings + MLP) is fully static.
- The upstream "cat_attn" path is supported. Because it attends over a single
  query token, self-attention is exactly equivalent to the value projection plus
  output projection, followed by the upstream residual LayerNorm.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

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


class AttentionLayer(nn.Module):
    """Single-token attention block matching upstream structure."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value, query_pos_enc, key_pos_enc, attn_mask=None):
        output, _ = self.attn(
            query + query_pos_enc,
            key + key_pos_enc,
            value,
            attn_mask=attn_mask,
        )
        return self.norm(query + output)


class FFNLayer(nn.Module):
    """Feed-forward residual block matching upstream structure."""

    def __init__(self, embed_dim: int, hidden_dim: int, activation: str = "ReLU"):
        super().__init__()
        if activation == "ReLU":
            act = nn.ReLU()
        elif activation == "GELU":
            act = nn.GELU()
        else:
            raise NotImplementedError(f"Unsupported activation: {activation}")
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(x + self.ff(x))


class DiffusionHead(nn.Module):
    """ONNX-compatible diffusion noise prediction network."""

    def __init__(
        self,
        diffusion_step_embed_dim: int = 512,
        observation_embed_dim: int = 512,
        sample_embed_dim: int = 512,
        sample_dim: int = 6,
        attention: str = "cat_attn",
    ):
        super().__init__()
        self.sample_dim = sample_dim
        self.attention = attention

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

        if self.attention == "cat_attn":
            self.query_pos_enc = nn.Embedding(1, diffusion_step_embed_dim + observation_embed_dim + sample_embed_dim)
            self.self_attention_layers = nn.ModuleList([
                AttentionLayer(diffusion_step_embed_dim + observation_embed_dim + sample_embed_dim, 8)
                for _ in range(3)
            ])
            self.ffn_layers = nn.ModuleList([
                FFNLayer(diffusion_step_embed_dim + observation_embed_dim + sample_embed_dim, 512, "GELU")
                for _ in range(3)
            ])
            total_input_dim = diffusion_step_embed_dim + observation_embed_dim + sample_embed_dim
        elif self.attention == "cat":
            self.query_pos_enc = None
            self.self_attention_layers = None
            self.ffn_layers = None
            total_input_dim = sample_embed_dim + diffusion_step_embed_dim + observation_embed_dim
        else:
            raise NotImplementedError(f"Unsupported attention mode: {attention}")

        self.prediction_head = nn.Sequential(
            nn.Linear(total_input_dim, total_input_dim // 2),
            nn.ReLU(),
            nn.Linear(total_input_dim // 2, total_input_dim // 4),
            nn.ReLU(),
            nn.Linear(total_input_dim // 4, sample_dim),
        )

    def _broadcast_timesteps(self, observation_embedding: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        device = observation_embedding.device
        batch = observation_embedding.shape[0]
        if torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(device)
        timesteps = timesteps.to(device)
        if timesteps.shape[0] != batch:
            timesteps = timesteps.reshape(-1)[:1].expand(batch)
        return timesteps

    def forward(self, observation_embedding: torch.Tensor, timesteps: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        timesteps = self._broadcast_timesteps(observation_embedding, timesteps)
        timestep_embedding = self.diffusion_step_encoder(timesteps)
        sample_embedding = self.sample_encoder(sample)

        if self.attention == "cat":
            embed = torch.cat([sample_embedding, timestep_embedding, observation_embedding], dim=-1)
            return self.prediction_head(embed)

        embed = torch.cat([sample_embedding, timestep_embedding, observation_embedding], dim=-1).unsqueeze(0)
        batch_size = embed.shape[1]
        query_pos_enc = self.query_pos_enc.weight.repeat(1, batch_size, 1)
        for i in range(3):
            embed = self.self_attention_layers[i](
                embed,
                embed,
                embed,
                query_pos_enc,
                query_pos_enc,
            )
            embed = self.ffn_layers[i](embed)
        embed = embed.squeeze(0)
        return self.prediction_head(embed)


if __name__ == "__main__":
    print("Testing DiffusionHead...")

    B = 20
    obs_dim = 512
    sample_dim = 6

    head = DiffusionHead(observation_embed_dim=obs_dim, sample_dim=sample_dim, attention="cat_attn")
    head.eval()

    obs = torch.randn(B, obs_dim)
    timestep = torch.tensor([5], dtype=torch.long)
    sample = torch.randn(B, sample_dim)

    with torch.no_grad():
        noise_pred = head(obs, timestep, sample)

    print(f"✓ obs:        {obs.shape}")
    print(f"✓ timestep:   {timestep.shape}")
    print(f"✓ sample:     {sample.shape}")
    print(f"✓ noise_pred: {noise_pred.shape}")
    assert noise_pred.shape == (B, sample_dim), "Output shape mismatch!"
    print("\n✓ DiffusionHead test passed!")
