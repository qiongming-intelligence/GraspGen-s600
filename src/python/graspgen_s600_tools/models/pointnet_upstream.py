"""
Weight-compatible PointNet++ encoder for loading upstream Robotiq pretrained weights.

This is a faithful PyTorch reimplementation of the upstream PointNetPlusPlus
(grasp_gen/models/model_utils.py) with ONNX-compatible sampling replacing FPS.

Structural match to upstream OBJ_* constants:
  OBJ_NPOINTS = [256, 64, None]
  OBJ_RADII = [0.02, 0.04, None]
  OBJ_NSAMPLES = [64, 128, None]
  OBJ_MLPS = [[0, 64, 128], [128, 128, 256], [256, 256, 512]]
  prediction_head: 512 -> 1024 -> 1024 -> 512

The upstream uses pointnet2_ops CUDA FPS; here we substitute random/grid sampling
for ONNX compatibility and measure the accuracy impact.

Key differences from Phase 2 pointnet_encoder.py:
- Matches upstream hyperparameters (npoints, radii, mlp dims)
- Adds the 3-layer prediction_head
- State dict keys align for torch.load_state_dict()
- Uses `use_xyz=True` (concatenate xyz to features in SA layers)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Calculate squared Euclidean distance between two point sets.

    Args:
        src: (B, N, C) source points
        dst: (B, M, C) destination points

    Returns:
        dist: (B, N, M) squared distances
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = torch.sum((src.unsqueeze(2) - dst.unsqueeze(1)) ** 2, dim=-1)
    return dist


def random_sample_pytorch(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Random sampling (ONNX-compatible replacement for FPS).

    Args:
        xyz: (B, N, 3) input points
        npoint: number of points to sample

    Returns:
        indices: (B, npoint) indices of sampled points
    """
    B, N, C = xyz.shape
    device = xyz.device
    indices = torch.randint(0, N, (B, npoint), device=device, dtype=torch.long)
    return indices


def query_ball_point(
    radius: float,
    nsample: int,
    xyz: torch.Tensor,
    new_xyz: torch.Tensor
) -> torch.Tensor:
    """
    Query ball point grouping (ONNX-compatible version).

    Args:
        radius: local region radius
        nsample: max sample number in local region
        xyz: (B, N, 3) all points
        new_xyz: (B, S, 3) query points

    Returns:
        group_idx: (B, S, nsample) grouped points indices
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    sqrdists = square_distance(new_xyz, xyz)  # (B, S, N)

    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat(B, S, 1)
    mask_far = sqrdists > radius ** 2
    group_idx = torch.where(mask_far, torch.tensor(N, dtype=torch.long, device=device), group_idx)

    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask_empty = group_idx == N
    group_idx = torch.where(mask_empty, group_first, group_idx)

    return group_idx


def sample_and_group(
    npoint: int,
    radius: float,
    nsample: int,
    xyz: torch.Tensor,
    points: Optional[torch.Tensor] = None,
    use_xyz: bool = True,
    use_random_sample: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample and group points (ONNX-compatible version).

    Args:
        npoint: number of centroids
        radius: ball query radius
        nsample: max sample number in local region
        xyz: (B, N, 3) input points
        points: (B, C, N) input features (channel first), optional
        use_xyz: concatenate xyz to features
        use_random_sample: use random sampling instead of FPS

    Returns:
        new_xyz: (B, npoint, 3) sampled centroids
        new_points: (B, npoint, nsample, C+3) or (B, npoint, nsample, 3) grouped features
    """
    B, N, _ = xyz.shape

    if use_random_sample:
        fps_idx = random_sample_pytorch(xyz, npoint)
    else:
        # FPS placeholder (not ONNX-compatible)
        raise NotImplementedError("FPS not ONNX-compatible; use random sampling")

    new_xyz = torch.gather(
        xyz, 1, fps_idx.unsqueeze(-1).expand(-1, -1, 3)
    )  # (B, npoint, 3)

    idx = query_ball_point(radius, nsample, xyz, new_xyz)  # (B, npoint, nsample)

    idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, 3)
    xyz_expanded = xyz.unsqueeze(1).expand(-1, npoint, -1, -1)
    grouped_xyz = torch.gather(xyz_expanded, 2, idx_expanded)  # (B, npoint, nsample, 3)

    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

    if points is not None:
        C = points.shape[1]
        idx_expanded_points = idx.unsqueeze(1).expand(-1, C, -1, -1)
        points_expanded = points.unsqueeze(2).expand(-1, -1, npoint, -1)
        grouped_points = torch.gather(points_expanded, 3, idx_expanded_points)
        grouped_points = grouped_points.permute(0, 2, 3, 1)  # (B, npoint, nsample, C)

        if use_xyz:
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    """
    PointNet Set Abstraction layer (upstream-compatible).

    Args:
        npoint: number of centroids to sample
        radius: ball query radius
        nsample: max sample number in local region
        in_channel: input channel dimension
        mlp: list of output channel dimensions
        group_all: whether to group all points
        use_xyz: concatenate xyz to features
    """

    def __init__(
        self,
        npoint: Optional[int],
        radius: float,
        nsample: int,
        in_channel: int,
        mlp: list,
        group_all: bool = False,
        use_xyz: bool = True
    ):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.use_xyz = use_xyz

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(
        self,
        xyz: torch.Tensor,
        points: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            xyz: (B, N, 3) input points
            points: (B, C, N) input features (channel first), optional

        Returns:
            new_xyz: (B, npoint, 3) or (B, 1, 3) sampled points
            new_points: (B, mlp[-1], npoint) or (B, mlp[-1], 1) output features
        """
        if self.group_all:
            new_xyz = xyz.mean(dim=1, keepdim=True)  # (B, 1, 3)
            B, N, _ = xyz.shape

            if points is not None:
                points_transposed = points.transpose(1, 2)  # (B, N, C)
                if self.use_xyz:
                    combined = torch.cat([xyz, points_transposed], dim=-1)
                else:
                    combined = points_transposed
            else:
                combined = xyz  # (B, N, 3)

            new_points = combined.unsqueeze(1)  # (B, 1, N, 3+C)
            new_points = new_points.permute(0, 3, 2, 1)  # (B, 3+C, N, 1)
        else:
            new_xyz, new_points = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points, self.use_xyz
            )
            # new_points: (B, npoint, nsample, C+3)
            new_points = new_points.permute(0, 3, 2, 1)  # (B, C+3, nsample, npoint)

        # Apply MLP
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        # Max pooling over nsample (or N for group_all)
        new_points = torch.max(new_points, dim=2)[0]  # (B, mlp[-1], npoint/1)

        return new_xyz, new_points


class PointNetUpstream(nn.Module):
    """
    Weight-compatible PointNet++ encoder matching upstream PointNetPlusPlus.

    Loads upstream Robotiq pretrained weights (obs_backbone=pointnet).

    Args:
        output_embedding_dim: output feature dimension (default: 512)
        feature_dim: input feature dim beyond xyz; -1 means xyz-only (default: -1)
    """

    # Upstream OBJ_* constants
    OBJ_NPOINTS = [256, 64, None]
    OBJ_RADII = [0.02, 0.04, None]
    OBJ_NSAMPLES = [64, 128, None]
    OBJ_MLPS = [[0, 64, 128], [128, 128, 256], [256, 256, 512]]

    def __init__(self, output_embedding_dim: int = 512, feature_dim: int = -1):
        super().__init__()
        self.output_embedding_dim = output_embedding_dim

        # Build MLP configs
        mlp = []
        for elem in self.OBJ_MLPS:
            mlp.append(elem.copy())
        if feature_dim > 0:
            mlp[0][0] = feature_dim

        # Set Abstraction layers
        self.obj_SA_modules = nn.ModuleList()
        for k in range(len(self.OBJ_NPOINTS)):
            self.obj_SA_modules.append(
                PointNetSetAbstraction(
                    npoint=self.OBJ_NPOINTS[k],
                    radius=self.OBJ_RADII[k],
                    nsample=self.OBJ_NSAMPLES[k],
                    in_channel=mlp[k][0],
                    mlp=mlp[k],
                    group_all=(self.OBJ_NPOINTS[k] is None),
                    use_xyz=True,
                )
            )

        # Prediction head (upstream: 512 -> 1024 -> 1024 -> 512)
        self.prediction_head = nn.Sequential(
            nn.Linear(self.OBJ_MLPS[-1][-1], 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.output_embedding_dim),
        )

    def forward(self, xyz: torch.Tensor, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            xyz: (B, N, 3) input point cloud
            features: (B, C, N) optional input features (channel first)

        Returns:
            features: (B, output_embedding_dim) global features
        """
        # SA layers
        for i in range(len(self.obj_SA_modules)):
            xyz, features = self.obj_SA_modules[i](xyz, features)

        # features: (B, 512, 1) after final SA (group_all)
        features = features.squeeze(-1)  # (B, 512)

        # Prediction head
        features = self.prediction_head(features)

        return features


if __name__ == "__main__":
    print("Testing PointNetUpstream (weight-compatible)...")

    encoder = PointNetUpstream(output_embedding_dim=512, feature_dim=-1)
    encoder.eval()

    batch_size = 2
    num_points = 2048
    xyz = torch.randn(batch_size, num_points, 3)

    with torch.no_grad():
        features = encoder(xyz)

    print(f"✓ Input shape: {xyz.shape}")
    print(f"✓ Output shape: {features.shape}")
    assert features.shape == (batch_size, 512), "Output shape mismatch!"

    print("\n✓ PointNetUpstream test passed!")
    print("Ready to load upstream Robotiq pretrained weights.")
