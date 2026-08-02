"""Pure-NumPy point cloud registration for markerless hand-eye calibration.

This module estimates the rigid camera self-motion between two RGBD captures by
registering their point clouds with point-to-plane ICP.  It has no dependency on
open3d/scipy/sklearn and never commands robot motion or contacts inference.

The registration result is the transform ``source_T_target`` such that applying
it to the target points aligns them onto the source points, i.e. it maps points
expressed in the target camera frame into the source camera frame.  For hand-eye
calibration with captures ``i`` (source) and ``j`` (target) this yields
``camera_i_T_camera_j``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ICPResult:
    """Result of a point-to-plane ICP registration."""

    source_T_target: np.ndarray
    fitness: float
    inlier_rmse_m: float
    correspondence_count: int
    iterations: int
    converged: bool
    reasons: list[str]


def voxel_downsample(points_m: np.ndarray, voxel_size_m: float) -> np.ndarray:
    """Downsample a point cloud by averaging points within each voxel."""

    pts = np.asarray(points_m, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if voxel_size_m <= 0.0:
        return np.ascontiguousarray(pts, dtype=np.float64)
    keys = np.floor(pts / float(voxel_size_m)).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    count = int(inverse.max()) + 1
    sums = np.zeros((count, 3), dtype=np.float64)
    np.add.at(sums, inverse, pts)
    totals = np.bincount(inverse, minlength=count).astype(np.float64)
    return np.ascontiguousarray(sums / totals[:, None], dtype=np.float64)


def _chunked_nearest(source: np.ndarray, target: np.ndarray, *, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest target index and squared distance for each source point."""

    n = source.shape[0]
    idx = np.empty(n, dtype=np.int64)
    dist_sq = np.empty(n, dtype=np.float64)
    target_sq = np.einsum("ij,ij->i", target, target)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        block = source[start:end]
        # ||s||^2 - 2 s.d + ||d||^2, dropping ||s||^2 (constant per row for argmin)
        cross = block @ target.T
        d2 = target_sq[None, :] - 2.0 * cross
        nearest = np.argmin(d2, axis=1)
        idx[start:end] = nearest
        block_sq = np.einsum("ij,ij->i", block, block)
        dist_sq[start:end] = block_sq + d2[np.arange(end - start), nearest]
    return idx, np.maximum(dist_sq, 0.0)


def estimate_normals(points_m: np.ndarray, *, k: int = 16, chunk: int = 512) -> np.ndarray:
    """Estimate per-point unit normals via local PCA over k nearest neighbors."""

    pts = np.asarray(points_m, dtype=np.float64).reshape(-1, 3)
    n = pts.shape[0]
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)
    k_eff = int(min(max(k, 3), n))
    normals = np.zeros((n, 3), dtype=np.float64)
    pts_sq = np.einsum("ij,ij->i", pts, pts)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        block = pts[start:end]
        cross = block @ pts.T
        d2 = pts_sq[None, :] - 2.0 * cross + np.einsum("ij,ij->i", block, block)[:, None]
        nn_idx = np.argpartition(d2, k_eff - 1, axis=1)[:, :k_eff]
        for row in range(end - start):
            neighbors = pts[nn_idx[row]]
            centered = neighbors - neighbors.mean(axis=0, keepdims=True)
            cov = centered.T @ centered
            eigvals, eigvecs = np.linalg.eigh(cov)
            normals[start + row] = eigvecs[:, 0]
    # Orient normals toward the camera origin (points face the sensor).
    view = -pts
    flip = np.einsum("ij,ij->i", normals, view) < 0.0
    normals[flip] = -normals[flip]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return np.ascontiguousarray(normals / norms, dtype=np.float64)


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _exp_se3(twist: np.ndarray) -> np.ndarray:
    """Exponential map of a 6-vector twist [omega; upsilon] to SE(3)."""

    omega = np.asarray(twist[:3], dtype=np.float64)
    upsilon = np.asarray(twist[3:], dtype=np.float64)
    theta = float(np.linalg.norm(omega))
    transform = np.eye(4, dtype=np.float64)
    if theta < 1e-12:
        transform[:3, 3] = upsilon
        return transform
    axis = omega / theta
    k = _skew(axis)
    rot = np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)
    v = np.eye(3) + (1.0 - np.cos(theta)) / theta * k + (theta - np.sin(theta)) / theta * (k @ k)
    transform[:3, :3] = rot
    transform[:3, 3] = v @ upsilon
    return transform


def icp_point_to_plane(
    source_points_m: np.ndarray,
    target_points_m: np.ndarray,
    *,
    init_source_T_target: np.ndarray | None = None,
    target_normals: np.ndarray | None = None,
    max_iterations: int = 50,
    max_correspondence_distance_m: float = 0.02,
    tolerance: float = 1e-6,
) -> ICPResult:
    """Register target onto source, returning ``source_T_target``.

    Points are expected already downsampled.  ``target_normals`` may be provided
    to skip re-estimation; otherwise per-point normals are computed on target.
    """

    source = np.asarray(source_points_m, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target_points_m, dtype=np.float64).reshape(-1, 3)
    reasons: list[str] = []
    if source.shape[0] < 10 or target.shape[0] < 10:
        return ICPResult(
            source_T_target=np.eye(4, dtype=np.float64),
            fitness=0.0,
            inlier_rmse_m=float("inf"),
            correspondence_count=0,
            iterations=0,
            converged=False,
            reasons=["too_few_points"],
        )
    # Point-to-plane uses normals on the source (fixed reference) cloud.
    source_normals = estimate_normals(source) if target_normals is None else np.asarray(target_normals, dtype=np.float64)
    transform = np.eye(4, dtype=np.float64) if init_source_T_target is None else np.asarray(init_source_T_target, dtype=np.float64).copy()
    max_dist_sq = float(max_correspondence_distance_m) ** 2
    prev_rmse = float("inf")
    fitness = 0.0
    rmse = float("inf")
    corr_count = 0
    iteration = 0
    converged = False
    for iteration in range(1, max_iterations + 1):
        moved = (transform[:3, :3] @ target.T).T + transform[:3, 3]
        idx, dist_sq = _chunked_nearest(moved, source)
        inliers = dist_sq <= max_dist_sq
        corr_count = int(np.count_nonzero(inliers))
        if corr_count < 6:
            reasons.append("too_few_correspondences")
            break
        s = source[idx[inliers]]
        n = source_normals[idx[inliers]]
        p = moved[inliers]
        residual = np.einsum("ij,ij->i", p - s, n)
        a = np.concatenate([np.cross(p, n), n], axis=1)
        lhs = a.T @ a
        rhs = -a.T @ residual
        try:
            delta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            delta, *_ = np.linalg.lstsq(a, -residual, rcond=None)
        transform = _exp_se3(delta) @ transform
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        fitness = corr_count / float(moved.shape[0])
        if abs(prev_rmse - rmse) < tolerance:
            converged = True
            break
        prev_rmse = rmse
    if not reasons:
        reasons.append("ok")
    return ICPResult(
        source_T_target=np.ascontiguousarray(transform, dtype=np.float64),
        fitness=float(fitness),
        inlier_rmse_m=float(rmse),
        correspondence_count=int(corr_count),
        iterations=int(iteration),
        converged=bool(converged),
        reasons=reasons,
    )


def register_pair(
    source_points_m: np.ndarray,
    target_points_m: np.ndarray,
    *,
    voxel_size_m: float = 0.005,
    init_source_T_target: np.ndarray | None = None,
    max_correspondence_distance_m: float = 0.02,
    max_iterations: int = 50,
) -> ICPResult:
    """Downsample both clouds then run point-to-plane ICP for ``source_T_target``."""

    src = voxel_downsample(source_points_m, voxel_size_m)
    tgt = voxel_downsample(target_points_m, voxel_size_m)
    return icp_point_to_plane(
        src,
        tgt,
        init_source_T_target=init_source_T_target,
        max_iterations=max_iterations,
        max_correspondence_distance_m=max_correspondence_distance_m,
    )


def icp_result_summary(result: ICPResult) -> dict[str, Any]:
    """Return a JSON-friendly ICP result summary."""

    return {
        "source_T_target": result.source_T_target.astype(float).tolist(),
        "fitness": result.fitness,
        "inlier_rmse_m": result.inlier_rmse_m,
        "correspondence_count": result.correspondence_count,
        "iterations": result.iterations,
        "converged": result.converged,
        "reasons": result.reasons,
    }
