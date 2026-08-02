"""Pose-scout scoring helpers for eye-in-hand Dofbot camera views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .hbm_graspgen_runtime import get_nested


@dataclass(frozen=True)
class TableViewScore:
    """JSON-friendly score for whether a camera view looks table-facing."""

    ok: bool
    score: float
    grade: str
    reasons: list[str]
    metrics: dict[str, Any]


def score_table_view(
    point_cloud: np.ndarray,
    *,
    config: dict[str, Any] | None = None,
    rgb_image: np.ndarray | None = None,
) -> TableViewScore:
    """Score a camera view using depth geometry and optional RGB color evidence.

    This is deliberately a scouting heuristic, not calibration authority.  A good
    score means the view contains a broad, central, table-like plane with enough
    metric depth points, plus optional colorful calibration-mat evidence if an RGB
    image is provided.  It should be used to rank candidate viewpoints before a
    human promotes one into `calibration.observation_pose`.
    """

    cfg = get_nested(config or {}, ("camera", "pose_scout"), {}) or {}
    max_abs = float(get_nested(config or {}, ("safety", "max_abs_coord_m"), 5.0))
    min_points = int(cfg.get("min_points", get_nested(config or {}, ("safety", "min_points_before_resample"), 256)))
    accept_score = float(cfg.get("accept_score", 0.65))

    pc = _clean_point_cloud(point_cloud, max_abs=max_abs)
    metrics: dict[str, Any] = {
        "point_count": int(pc.shape[0]),
        "min_points": min_points,
        "rgb_available": rgb_image is not None,
    }
    reasons: list[str] = []
    if pc.shape[0] == 0:
        return TableViewScore(False, 0.0, "empty", ["no_valid_depth_points"], metrics)

    metrics.update(_basic_depth_metrics(pc))
    if pc.shape[0] < min_points:
        reasons.append("too_few_depth_points")

    plane = estimate_dominant_plane(
        pc,
        threshold_m=float(cfg.get("plane_distance_threshold_m", 0.012)),
        iterations=int(cfg.get("ransac_iterations", 128)),
        sample_points=int(cfg.get("ransac_sample_points", 4096)),
        evaluation_points=int(cfg.get("ransac_evaluation_points", 50000)),
        seed=int(cfg.get("seed", get_nested(config or {}, ("runtime", "seed"), 0))),
    )
    if plane is None:
        plane_metrics: dict[str, Any] = {
            "plane_found": False,
            "plane_fraction": 0.0,
            "plane_extent_min_m": 0.0,
            "plane_normal_abs_z": 0.0,
        }
        reasons.append("no_dominant_plane")
    else:
        plane_metrics = plane
        min_plane_fraction = float(cfg.get("min_plane_fraction", 0.15))
        min_plane_extent = float(cfg.get("min_plane_extent_m", 0.12))
        min_normal_abs_z = float(cfg.get("min_plane_normal_abs_z", 0.20))
        if float(plane_metrics["plane_fraction"]) < min_plane_fraction:
            reasons.append("dominant_plane_fraction_too_small")
        if float(plane_metrics["plane_extent_min_m"]) < min_plane_extent:
            reasons.append("dominant_plane_extent_too_small")
        if float(plane_metrics["plane_normal_abs_z"]) < min_normal_abs_z:
            reasons.append("dominant_plane_not_camera_facing")
    metrics.update(plane_metrics)

    center_metrics = _central_coverage_metrics(pc, cfg)
    metrics.update(center_metrics)

    rgb_metrics = score_rgb_image(rgb_image, config=config) if rgb_image is not None else None
    if rgb_metrics is not None:
        metrics.update(rgb_metrics)
        rgb_cfg = cfg.get("rgb") or {}
        if bool(rgb_cfg.get("require_color", False)):
            min_colored = float(rgb_cfg.get("min_colored_fraction", 0.02))
            if float(rgb_metrics.get("rgb_colored_fraction", 0.0)) < min_colored:
                reasons.append("rgb_color_evidence_too_weak")

    components = _score_components(metrics, cfg, rgb_metrics is not None)
    metrics["score_components"] = components
    score = _weighted_score(components, cfg, rgb_metrics is not None)
    if score < accept_score:
        reasons.append("score_below_accept_threshold")
    ok = not reasons
    grade = _grade(score, ok)
    return TableViewScore(ok=ok, score=score, grade=grade, reasons=reasons or ["ok"], metrics=metrics)


def estimate_dominant_plane(
    points: np.ndarray,
    *,
    threshold_m: float,
    iterations: int,
    sample_points: int,
    evaluation_points: int,
    seed: int,
) -> dict[str, Any] | None:
    """Estimate a dominant plane with deterministic RANSAC and SVD refinement."""

    if points.shape[0] < 3:
        return None
    if threshold_m <= 0:
        raise ValueError("threshold_m must be positive")
    sample = _sample_points(points, sample_points, seed=seed)
    if sample.shape[0] < 3:
        return None

    rng = np.random.default_rng(seed)
    best_normal: np.ndarray | None = None
    best_offset = 0.0
    best_count = 0
    for _ in range(max(1, iterations)):
        ids = rng.choice(sample.shape[0], size=3, replace=False)
        p0, p1, p2 = sample[ids]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-8:
            continue
        normal = normal / norm
        offset = -float(np.dot(normal, p0))
        distances = np.abs(sample @ normal + offset)
        count = int(np.count_nonzero(distances <= threshold_m))
        if count > best_count:
            best_count = count
            best_normal = normal
            best_offset = offset
    if best_normal is None or best_count < 3:
        return None

    initial_distances = np.abs(sample @ best_normal + best_offset)
    initial_inliers = sample[initial_distances <= threshold_m]
    if initial_inliers.shape[0] < 3:
        return None
    refined = _refine_plane(initial_inliers)
    if refined is None:
        return None
    normal, offset, basis = refined
    if normal[2] < 0:
        normal = -normal
        offset = -offset

    evaluation = _sample_points(points, evaluation_points, seed=seed + 1)
    distances = np.abs(evaluation @ normal + offset)
    inlier_mask = distances <= threshold_m
    inliers = evaluation[inlier_mask]
    if inliers.shape[0] < 3:
        return None
    projected = (inliers - inliers.mean(axis=0, keepdims=True)) @ basis.T
    extents = projected.max(axis=0) - projected.min(axis=0)
    residuals = distances[inlier_mask]
    centroid = inliers.mean(axis=0)
    return {
        "plane_found": True,
        "plane_inlier_count": int(inliers.shape[0]),
        "plane_evaluation_count": int(evaluation.shape[0]),
        "plane_fraction": float(inliers.shape[0] / max(evaluation.shape[0], 1)),
        "plane_normal": normal.astype(float).tolist(),
        "plane_normal_abs_z": float(abs(normal[2])),
        "plane_offset_m": float(offset),
        "plane_centroid_m": centroid.astype(float).tolist(),
        "plane_extent_m": extents.astype(float).tolist(),
        "plane_extent_min_m": float(np.min(extents)),
        "plane_extent_area_m2": float(np.prod(extents)),
        "plane_residual_median_m": float(np.median(residuals)),
        "plane_residual_p95_m": float(np.percentile(residuals, 95.0)),
        "plane_distance_threshold_m": float(threshold_m),
    }


def score_rgb_image(rgb_image: np.ndarray | None, *, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return simple color-evidence metrics for a calibration mat image."""

    if rgb_image is None:
        return {"rgb_available": False}
    arr = np.asarray(rgb_image)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"rgb_image must be HxWx3 or HxWx4, got {arr.shape}")
    arr = arr[:, :, :3].astype(np.float32)
    if arr.max(initial=0.0) > 1.5:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)

    cfg = get_nested(config or {}, ("camera", "pose_scout", "rgb"), {}) or {}
    central_fraction = float(cfg.get("central_fraction", 0.70))
    if central_fraction <= 0 or central_fraction > 1:
        raise ValueError("camera.pose_scout.rgb.central_fraction must be in (0, 1]")
    h, w = arr.shape[:2]
    y0 = int(round((1.0 - central_fraction) * 0.5 * h))
    y1 = int(round((1.0 + central_fraction) * 0.5 * h))
    x0 = int(round((1.0 - central_fraction) * 0.5 * w))
    x1 = int(round((1.0 + central_fraction) * 0.5 * w))
    roi = arr[y0:y1, x0:x1]
    mx = roi.max(axis=2)
    mn = roi.min(axis=2)
    saturation = (mx - mn) / np.maximum(mx, 1e-6)
    brightness = mx
    min_saturation = float(cfg.get("min_saturation", 0.25))
    min_brightness = float(cfg.get("min_brightness", 0.12))
    colored = (saturation >= min_saturation) & (brightness >= min_brightness)

    r, g, b = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    colorfulness = float(np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2) + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2))
    return {
        "rgb_available": True,
        "rgb_shape": [int(h), int(w), 3],
        "rgb_central_fraction": central_fraction,
        "rgb_colored_fraction": float(np.mean(colored)),
        "rgb_saturation_mean": float(np.mean(saturation)),
        "rgb_brightness_mean": float(np.mean(brightness)),
        "rgb_colorfulness": colorfulness,
    }


def _clean_point_cloud(points: np.ndarray, *, max_abs: float) -> np.ndarray:
    pc = np.asarray(points, dtype=np.float32)
    if pc.ndim != 2 or pc.shape[1] < 3:
        raise ValueError(f"point cloud must be (N, >=3), got {pc.shape}")
    pc = pc[:, :3]
    finite = np.isfinite(pc).all(axis=1)
    pc = pc[finite]
    if max_abs > 0:
        in_range = np.all(np.abs(pc) <= max_abs, axis=1)
        pc = pc[in_range]
    return np.ascontiguousarray(pc, dtype=np.float32)


def _sample_points(points: np.ndarray, max_points: int, *, seed: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return np.ascontiguousarray(points, dtype=np.float32)
    rng = np.random.default_rng(seed)
    indices = rng.choice(points.shape[0], size=max_points, replace=False)
    return np.ascontiguousarray(points[indices], dtype=np.float32)


def _refine_plane(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray] | None:
    centroid = points.mean(axis=0)
    centered = points - centroid
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if vh.shape[0] < 3:
        return None
    normal = vh[-1]
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-8:
        return None
    normal = normal / norm
    offset = -float(np.dot(normal, centroid))
    basis = vh[:2]
    return normal.astype(np.float32), offset, basis.astype(np.float32)


def _basic_depth_metrics(points: np.ndarray) -> dict[str, Any]:
    z = points[:, 2]
    return {
        "bounds_min_m": points.min(axis=0).astype(float).tolist(),
        "bounds_max_m": points.max(axis=0).astype(float).tolist(),
        "extent_m": (points.max(axis=0) - points.min(axis=0)).astype(float).tolist(),
        "centroid_m": points.mean(axis=0).astype(float).tolist(),
        "depth_min_m": float(np.min(z)),
        "depth_median_m": float(np.median(z)),
        "depth_p95_m": float(np.percentile(z, 95.0)),
        "depth_max_m": float(np.max(z)),
    }


def _central_coverage_metrics(points: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    bounds = cfg.get("central_bounds_m") or {"x": [-0.35, 0.35], "y": [-0.30, 0.30]}
    if not isinstance(bounds, dict):
        raise ValueError("camera.pose_scout.central_bounds_m must be a mapping")
    mask = np.ones(points.shape[0], dtype=bool)
    applied: dict[str, list[float | None]] = {}
    for axis_name, axis in (("x", 0), ("y", 1)):
        raw = bounds.get(axis_name)
        if raw is None:
            continue
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ValueError(f"central bound {axis_name!r} must be [min, max]")
        lo = None if raw[0] is None else float(raw[0])
        hi = None if raw[1] is None else float(raw[1])
        if lo is not None:
            mask &= points[:, axis] >= lo
        if hi is not None:
            mask &= points[:, axis] <= hi
        applied[axis_name] = [lo, hi]
    return {
        "central_bounds_m": applied,
        "central_point_count": int(np.count_nonzero(mask)),
        "central_fraction": float(np.mean(mask)) if mask.size else 0.0,
    }


def _score_components(metrics: dict[str, Any], cfg: dict[str, Any], has_rgb: bool) -> dict[str, float]:
    min_points = float(metrics.get("min_points", 256))
    point_score = _clamp01(float(metrics.get("point_count", 0)) / max(min_points, 1.0))

    min_plane_fraction = float(cfg.get("min_plane_fraction", 0.15))
    target_plane_fraction = float(cfg.get("target_plane_fraction", 0.45))
    plane_fraction_score = _ramp(
        float(metrics.get("plane_fraction", 0.0)),
        lo=min_plane_fraction,
        hi=target_plane_fraction,
    )

    min_plane_extent = float(cfg.get("min_plane_extent_m", 0.12))
    target_plane_extent = float(cfg.get("target_plane_extent_m", 0.30))
    plane_extent_score = _ramp(
        float(metrics.get("plane_extent_min_m", 0.0)),
        lo=min_plane_extent,
        hi=target_plane_extent,
    )

    min_normal_abs_z = float(cfg.get("min_plane_normal_abs_z", 0.20))
    normal_score = _ramp(float(metrics.get("plane_normal_abs_z", 0.0)), lo=min_normal_abs_z, hi=1.0)

    target_central_fraction = float(cfg.get("target_central_fraction", 0.25))
    central_score = _clamp01(float(metrics.get("central_fraction", 0.0)) / max(target_central_fraction, 1e-6))

    preferred_depth = cfg.get("preferred_depth_m", [0.30, 1.80])
    if not isinstance(preferred_depth, (list, tuple)) or len(preferred_depth) != 2:
        raise ValueError("camera.pose_scout.preferred_depth_m must be [min, max]")
    depth_score = _window_score(float(metrics.get("depth_median_m", 0.0)), float(preferred_depth[0]), float(preferred_depth[1]))

    components = {
        "points": point_score,
        "plane_fraction": plane_fraction_score,
        "plane_extent": plane_extent_score,
        "normal": normal_score,
        "central": central_score,
        "depth": depth_score,
    }
    if has_rgb:
        rgb_cfg = cfg.get("rgb") or {}
        target_colored = float(rgb_cfg.get("target_colored_fraction", 0.08))
        components["rgb"] = _clamp01(float(metrics.get("rgb_colored_fraction", 0.0)) / max(target_colored, 1e-6))
    return components


def _weighted_score(components: dict[str, float], cfg: dict[str, Any], has_rgb: bool) -> float:
    raw_weights = cfg.get("weights") or {
        "points": 0.10,
        "plane_fraction": 0.25,
        "plane_extent": 0.20,
        "normal": 0.10,
        "central": 0.15,
        "depth": 0.10,
        "rgb": 0.10,
    }
    if not isinstance(raw_weights, dict):
        raise ValueError("camera.pose_scout.weights must be a mapping")
    total = 0.0
    weighted = 0.0
    for key, value in components.items():
        if key == "rgb" and not has_rgb:
            continue
        weight = float(raw_weights.get(key, 0.0))
        if weight <= 0:
            continue
        weighted += weight * float(value)
        total += weight
    return float(weighted / total) if total > 0 else 0.0


def _window_score(value: float, lo: float, hi: float) -> float:
    if lo > hi:
        lo, hi = hi, lo
    if lo <= value <= hi:
        return 1.0
    width = max(hi - lo, 1e-6)
    if value < lo:
        return _clamp01(1.0 - (lo - value) / width)
    return _clamp01(1.0 - (value - hi) / width)


def _ramp(value: float, *, lo: float, hi: float) -> float:
    if hi <= lo:
        return 1.0 if value >= lo else 0.0
    return _clamp01((value - lo) / (hi - lo))


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _grade(score: float, ok: bool) -> str:
    if not ok:
        return "reject"
    if score >= 0.85:
        return "strong"
    if score >= 0.70:
        return "usable"
    return "marginal"
