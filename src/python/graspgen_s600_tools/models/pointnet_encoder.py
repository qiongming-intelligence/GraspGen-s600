"""
Pure PyTorch PointNet++ implementation for ONNX compatibility.

This module provides a ONNX-exportable PointNet++ encoder without
any CUDA extensions or custom operators.

Based on the original PointNet++ architecture but using only standard
PyTorch operations.

Key features:
- No CUDA extensions (no farthest point sampling C++/CUDA)
- No spconv or other sparse convolution libraries
- Full ONNX compatibility
- Differentiable and trainable

Architecture:
- Set Abstraction layers with multi-scale grouping
- Feature propagation for decoding (if needed)
- Global feature aggregation
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

    # Expand dimensions for broadcasting
    # src: (B, N, 1, C)
    # dst: (B, 1, M, C)
    dist = torch.sum((src.unsqueeze(2) - dst.unsqueeze(1)) ** 2, dim=-1)
    return dist


def farthest_point_sample_pytorch(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Pure PyTorch farthest point sampling (no CUDA extension).

    This is slower than CUDA version but fully ONNX-exportable.

    Args:
        xyz: (B, N, 3) input points
        npoint: number of points to sample

    Returns:
        centroids: (B, npoint) indices of sampled points
    """
    device = xyz.device
    B, N, C = xyz.shape

    # Initialize centroids
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10

    # Randomly select first centroid
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


def query_ball_point(
    radius: float,
    nsample: int,
    xyz: torch.Tensor,
    new_xyz: torch.Tensor
) -> torch.Tensor:
    """
    Query ball point grouping.

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

    # Compute squared distances
    sqrdists = square_distance(new_xyz, xyz)  # (B, S, N)

    # Find points within radius
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat(B, S, 1)
    group_idx[sqrdists > radius ** 2] = N

    # Take first nsample points
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    # Handle cases where less than nsample points are found
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


def sample_and_group(
    npoint: int,
    radius: float,
    nsample: int,
    xyz: torch.Tensor,
    points: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample and group points.

    Args:
        npoint: number of centroids
        radius: ball query radius
        nsample: max sample number in local region
        xyz: (B, N, 3) input points
        points: (B, C, N) input features (channel first)

    Returns:
        new_xyz: (B, npoint, 3) sampled centroids
        new_points: (B, npoint, nsample, C+3) grouped features
    """
    B, N, _ = xyz.shape

    # Sample centroids
    fps_idx = farthest_point_sample_pytorch(xyz, npoint)  # (B, npoint)

    # Gather sampled points
    new_xyz = torch.gather(
        xyz, 1, fps_idx.unsqueeze(-1).expand(-1, -1, 3)
    )  # (B, npoint, 3)

    # Query ball point
    idx = query_ball_point(radius, nsample, xyz, new_xyz)  # (B, npoint, nsample)

    # Group xyz coordinates
    # Use advanced indexing instead of gather
    batch_indices = torch.arange(B, device=xyz.device).view(B, 1, 1)
    grouped_xyz = xyz[batch_indices, idx]  # (B, npoint, nsample, 3)

    # Translate to relative coordinates
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

    if points is not None:
        # points: (B, C, N) -> need to gather along N dimension
        # Reshape for gathering: (B, N, C)
        points_transposed = points.transpose(1, 2)  # (B, N, C)
        grouped_points = points_transposed[batch_indices, idx]  # (B, npoint, nsample, C)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    """
    PointNet Set Abstraction layer.

    Args:
        npoint: number of centroids to sample
        radius: ball query radius
        nsample: max sample number in local region
        in_channel: input channel dimension
        mlp: list of output channel dimensions
        group_all: whether to group all points
    """

    def __init__(
        self,
        npoint: Optional[int],
        radius: float,
        nsample: int,
        in_channel: int,
        mlp: list,
        group_all: bool = False
    ):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        # MLP layers
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
            points: (B, C, N) input features (channel first)

        Returns:
            new_xyz: (B, npoint, 3) sampled points
            new_points: (B, mlp[-1], npoint) output features
        """
        if self.group_all:
            # Global pooling: use all points
            new_xyz = xyz.mean(dim=1, keepdim=True)  # (B, 1, 3)

            if points is not None:
                # points: (B, C, N) -> (B, N, C)
                points_transposed = points.transpose(1, 2)
                # Concatenate xyz and features
                new_points = torch.cat([xyz, points_transposed], dim=-1)  # (B, N, 3+C)
                new_points = new_points.unsqueeze(2)  # (B, N, 1, 3+C)
            else:
                new_points = xyz.unsqueeze(2)  # (B, N, 1, 3)

            # Permute for conv: (B, 3+C, 1, N)
            new_points = new_points.permute(0, 3, 2, 1)
        else:
            new_xyz, new_points = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points
            )
            # new_points: (B, npoint, nsample, C+3)
            # Permute for conv: (B, C+3, nsample, npoint)
            new_points = new_points.permute(0, 3, 2, 1)

        # Apply MLP
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        # Max pooling over nsample
        new_points = torch.max(new_points, dim=2)[0]  # (B, mlp[-1], npoint)

        return new_xyz, new_points


class PointNetEncoder(nn.Module):
    """
    Pure PyTorch PointNet++ encoder for point cloud feature extraction.

    This encoder is fully ONNX-compatible and doesn't require any
    CUDA extensions.

    Args:
        num_classes: output feature dimension (default: 512)
        normal_channel: whether input has normals (default: False)
    """

    def __init__(self, num_classes: int = 512, normal_channel: bool = False):
        super().__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel

        # Set Abstraction layers
        self.sa1 = PointNetSetAbstraction(
            npoint=512,
            radius=0.2,
            nsample=32,
            in_channel=in_channel,
            mlp=[64, 64, 128],
            group_all=False
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.4,
            nsample=64,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
            group_all=False
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256 + 3,
            mlp=[256, 512, num_classes],
            group_all=True
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            xyz: (B, N, 3) or (B, N, 6) input point cloud

        Returns:
            features: (B, num_classes) global features
        """
        B, N, C = xyz.shape

        if self.normal_channel:
            norm = xyz[:, :, 3:]
            xyz = xyz[:, :, :3]
        else:
            norm = None

        # Set Abstraction layers
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Global feature: (B, num_classes, 1) -> (B, num_classes)
        x = l3_points.squeeze(-1)

        return x


if __name__ == "__main__":
    # Test the encoder
    print("Testing PointNet encoder...")

    encoder = PointNetEncoder(num_classes=512, normal_channel=False)
    encoder.eval()

    # Test input
    batch_size = 2
    num_points = 2048
    xyz = torch.randn(batch_size, num_points, 3)

    with torch.no_grad():
        features = encoder(xyz)

    print(f"✓ Input shape: {xyz.shape}")
    print(f"✓ Output shape: {features.shape}")
    assert features.shape == (batch_size, 512), "Output shape mismatch!"

    print("\n✓ PointNet encoder test passed!")
