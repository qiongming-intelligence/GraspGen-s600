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
    Random sampling (ONNX-compatible but non-deterministic).

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


def fps_faithful(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Pure-PyTorch Farthest Point Sampling, 1:1 replicating the upstream CUDA kernel.

    Faithfully reproduces all quirks of ``furthest_point_sampling_kernel`` in
    ``pointnet2_ops/_ext-src/src/sampling_gpu.cu`` and ``sampling.cpp``:

    1. Distance buffer initialised to **1e10**
       (``sampling.cpp``: ``torch::full({..., 1e10})``).
    2. First centroid is always **index 0**
       (kernel: ``int old = 0; idxs[0] = old``).
    3. Points with ``||p||² ≤ 1e-3`` are **never selected** as centroids
       and their distance buffer entries are **never updated**
       (kernel: ``if (mag <= 1e-3) continue`` before the distance update).
    4. Update rule: ``temp[k] = min(temp[k], dist_to_new_centroid)``
       (kernel: ``float d2 = min(d, temp[k]); temp[k] = d2``).
    5. Tie-break: on equal distance, the **lowest global index** wins
       (kernel per-thread: ``besti = d2 > best ? k : besti`` keeps lower k;
       block reduction: ``dists_i[idx1] = v2 > v1 ? i2 : i1`` keeps lower tid).
       ``torch.argmax`` returns the first (lowest-index) maximum → matches. ✓

    Every operation inside the fixed-length loop (sub / mul / ReduceSum /
    minimum / argmax / gather) is a standard ONNX opset-17 operator, so the
    unrolled graph exports cleanly.

    Args:
        xyz: (B, N, 3) input point coordinates
        npoint: number of centroids to sample

    Returns:
        indices: (B, npoint) sampled point indices
    """
    B, N, _ = xyz.shape
    device = xyz.device

    # 1. Initialise distance buffer to 1e10
    dist = xyz.new_full((B, N), 1e10)

    # 3. Near-origin mask: ‖p‖² ≤ 1e-3 → never selected, temp never updated
    near_origin = (xyz ** 2).sum(dim=-1) <= 1e-3  # (B, N)

    # 2. First centroid is always index 0
    last = xyz.new_zeros(B, dtype=torch.long, device=device)

    indices_list: list[torch.Tensor] = []
    for _ in range(npoint):
        indices_list.append(last)
        # Gather the coordinates of the last-selected centroid: (B, 1, 3)
        centroid = torch.gather(
            xyz, 1, last.view(B, 1, 1).expand(-1, -1, 3)
        )  # (B, 1, 3)
        # Squared distance from every point to the centroid: (B, N)
        d = ((xyz - centroid) ** 2).sum(dim=-1)
        # 4. temp[k] = min(temp[k], d[k])
        dist = torch.minimum(dist, d)
        # 3. Near-origin points are never selected: set their candidate dist
        #    to -1 so they can never win argmax (their real distances are kept
        #    in <dist> for the next round, only the selection mask is affected).
        masked = dist.masked_fill(near_origin, -1.0)
        # 5. argmax → lowest-index tie (torch default)
        last = masked.argmax(dim=-1)  # (B,)

    return torch.stack(indices_list, dim=1)  # (B, npoint)


def uniform_sample_pytorch(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Deterministic uniform stride sampling (ONNX-compatible).

    Picks every (N // npoint)-th point.  Less geometrically optimal than
    FPS but fully deterministic and reproducible.

    Args:
        xyz: (B, N, 3) input points
        npoint: number of points to sample

    Returns:
        indices: (B, npoint) indices of sampled points
    """
    B, N, _ = xyz.shape
    device = xyz.device
    stride = max(N // npoint, 1)
    indices = torch.arange(0, npoint, dtype=torch.long, device=device) * stride
    indices = indices.clamp(max=N - 1)
    indices = indices.unsqueeze(0).expand(B, -1)  # (B, npoint)
    return indices


def query_ball_point(
    radius: float,
    nsample: int,
    xyz: torch.Tensor,
    new_xyz: torch.Tensor
) -> torch.Tensor:
    """
    Ball query grouping — pure-PyTorch, 1:1 replicating the upstream CUDA
    ``ball_query`` kernel (pointnet2_ops/_ext-src/src/ball_query_gpu.cu).

    The CUDA kernel semantics reproduced here:
    - For each query center, scan all points in **original index order**.
    - A point is included if its squared distance to the center < radius².
    - Collect up to ``nsample`` points in index order.
    - **Empty-slot fill**: if fewer than ``nsample`` points fall in the ball,
      the remaining slots are filled with the **first in-ball point** index
      (kernel: ``if (cnt == 0) for (l) idx[j*nsample+l] = k;``).

    This implementation reproduces that exactly via sort + mask: out-of-radius
    points are set to index ``N`` (a sentinel larger than any valid index),
    sorting brings in-range points first in ascending index order, we take the
    first ``nsample``, and any leftover sentinel slots are replaced by the
    first valid index (the ``group_first`` fill). Uses only square_distance,
    where, sort, slice — all ONNX opset-17 operators.

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
    # CUDA ball_query includes points only when d2 < radius2 (strictly inside).
    mask_far = sqrdists >= radius ** 2
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
    sampling: str = "fps"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample and group points.

    Args:
        npoint: number of centroids
        radius: ball query radius
        nsample: max sample number in local region
        xyz: (B, N, 3) input points
        points: (B, C, N) input features (channel first), optional
        use_xyz: concatenate xyz to features
        sampling: centroid selection strategy — "fps" (faithful CUDA replica),
                  "random" (uniform random), or "uniform" (fixed stride)

    Returns:
        new_xyz: (B, npoint, 3) sampled centroids
        new_points: (B, npoint, nsample, C+3) or (B, npoint, nsample, 3) grouped features
    """
    B, N, _ = xyz.shape

    if sampling == "fps":
        fps_idx = fps_faithful(xyz, npoint)
    elif sampling == "random":
        fps_idx = random_sample_pytorch(xyz, npoint)
    elif sampling == "uniform":
        fps_idx = uniform_sample_pytorch(xyz, npoint)
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling!r}")

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
        sampling: centroid selection strategy — "fps", "random", or "uniform"
    """

    def __init__(
        self,
        npoint: Optional[int],
        radius: float,
        nsample: int,
        in_channel: int,
        mlp: list,
        group_all: bool = False,
        use_xyz: bool = True,
        sampling: str = "fps"
    ):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.use_xyz = use_xyz
        self.sampling = sampling

        # Match upstream structure: self.mlps is a ModuleList with one Sequential
        # The Sequential contains Conv2d + BN + ReLU for each MLP layer
        layers = []
        last_channel = in_channel
        for out_channel in mlp:
            layers.append(nn.Conv2d(last_channel, out_channel, 1, bias=False))
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU(True))
            last_channel = out_channel

        # Wrap in ModuleList to match upstream key structure
        self.mlps = nn.ModuleList([nn.Sequential(*layers)])

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
            # Match upstream GroupAll exactly:
            # grouped_xyz = xyz.transpose(1, 2).unsqueeze(2) -> (B, 3, 1, N)
            # grouped_features = features.unsqueeze(2)        -> (B, C, 1, N)
            # new_features = cat([grouped_xyz, grouped_features], dim=1)
            new_xyz = torch.zeros_like(xyz[:, :1])
            B, N, _ = xyz.shape

            grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)  # (B, 3, 1, N)
            if points is not None:
                grouped_features = points.unsqueeze(2)  # (B, C, 1, N)
                if self.use_xyz:
                    new_points = torch.cat([grouped_xyz, grouped_features], dim=1)
                else:
                    new_points = grouped_features
            else:
                new_points = grouped_xyz
        else:
            new_xyz, new_points = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points, self.use_xyz,
                sampling=self.sampling
            )
            # Match upstream QueryAndGroup layout: (B, C+3, npoint, nsample)
            new_points = new_points.permute(0, 3, 1, 2)

        # Apply MLP (using the Sequential in self.mlps[0])
        new_points = self.mlps[0](new_points)

        # Max pooling over nsample (or N for group_all), matching upstream dim=-1
        new_points = torch.max(new_points, dim=-1)[0]  # (B, mlp[-1], npoint/1)

        return new_xyz, new_points


class PointNetUpstream(nn.Module):
    """
    Weight-compatible PointNet++ encoder matching upstream PointNetPlusPlus.

    Loads upstream Robotiq pretrained weights (obs_backbone=pointnet).

    Args:
        output_embedding_dim: output feature dimension (default: 512)
        feature_dim: input feature dim beyond xyz; -1 means xyz-only (default: -1)
        sampling: centroid selection strategy for SA layers —
                  "fps" (faithful CUDA-replica FPS, default),
                  "random" (uniform random),
                  "uniform" (fixed stride)
    """

    # Upstream OBJ_* constants
    OBJ_NPOINTS = [256, 64, None]
    OBJ_RADII = [0.02, 0.04, None]
    OBJ_NSAMPLES = [64, 128, None]
    OBJ_MLPS = [[0, 64, 128], [128, 128, 256], [256, 256, 512]]

    def __init__(
        self,
        output_embedding_dim: int = 512,
        feature_dim: int = -1,
        sampling: str = "fps"
    ):
        super().__init__()
        self.output_embedding_dim = output_embedding_dim

        # Build MLP configs (upstream logic from PointNetPlusPlus.__init__)
        mlp = []
        for elem in self.OBJ_MLPS:
            mlp.append(elem.copy())

        # The first SA layer's in_channel depends on feature_dim:
        # - If feature_dim > 0: use features beyond xyz, in_channel = feature_dim
        # - If feature_dim == -1 (xyz-only): in_channel = 0, but use_xyz=True adds 3
        # The mlp[0][0] = 0 is a placeholder; the actual in_channel to the MLP conv
        # is determined by sample_and_group's output (xyz + features if use_xyz=True).
        # Since use_xyz=True, the first SA gets 3 (xyz) + feature_dim if >0, else just 3.
        if feature_dim > 0:
            mlp[0][0] = feature_dim

        # Set Abstraction layers
        self.obj_SA_modules = nn.ModuleList()
        for k in range(len(self.OBJ_NPOINTS)):
            # Compute actual in_channel for this SA layer
            if k == 0:
                # First SA: use_xyz=True adds 3, mlp[0][0] is feature_dim or 0
                actual_in = 3 + (mlp[k][0] if mlp[k][0] > 0 else 0)
            else:
                # Subsequent SAs: use_xyz=True adds 3, input features from prev SA
                actual_in = 3 + mlp[k-1][-1]

            self.obj_SA_modules.append(
                PointNetSetAbstraction(
                    npoint=self.OBJ_NPOINTS[k],
                    radius=self.OBJ_RADII[k],
                    nsample=self.OBJ_NSAMPLES[k],
                    in_channel=actual_in,
                    mlp=mlp[k][1:],  # skip the placeholder first elem
                    group_all=(self.OBJ_NPOINTS[k] is None),
                    use_xyz=True,
                    sampling=sampling,
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

    def forward(self, pc: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            pc: (B, N, 3) input point cloud (xyz only, no features)

        Returns:
            features: (B, output_embedding_dim) global features
        """
        # Break up into xyz and features (upstream convention)
        xyz = pc
        features = None

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
    pc = torch.randn(batch_size, num_points, 3)

    with torch.no_grad():
        features = encoder(pc)

    print(f"✓ Input shape: {pc.shape}")
    print(f"✓ Output shape: {features.shape}")
    assert features.shape == (batch_size, 512), "Output shape mismatch!"

    print("\n✓ PointNetUpstream test passed!")
    print("Ready to load upstream Robotiq pretrained weights.")
