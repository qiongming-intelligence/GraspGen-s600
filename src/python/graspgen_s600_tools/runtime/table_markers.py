"""Dynamic colored table-marker detection for no-motion calibration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .hbm_graspgen_runtime import get_nested, resolve_config_path
from .pose_scout import estimate_dominant_plane


@dataclass(frozen=True)
class ColoredPointCloud:
    """Colored point cloud loaded from an aligned RGBD PLY."""

    points_m: np.ndarray
    rgb: np.ndarray
    source_path: Path
    source_units: str


@dataclass(frozen=True)
class DetectedTableMarker:
    """A colored connected component that may correspond to a table marker."""

    marker_id: str | None
    color: str
    camera_point_m: np.ndarray
    median_point_m: np.ndarray
    point_count: int
    extent_m: np.ndarray
    rgb_mean: np.ndarray
    rgb_median: np.ndarray
    plane_distance_m: float | None
    plane_distance_median_m: float | None
    plane_distance_p90_m: float | None
    confidence: float
    accepted: bool
    reason: str
    match_index: int | None = None


@dataclass(frozen=True)
class TableMarkerDetection:
    """Result of dynamic colored marker detection from one RGBD capture."""

    configured_marker_ids: list[str]
    marker_ids: list[str]
    markers: list[DetectedTableMarker]
    candidates: list[DetectedTableMarker]
    missing_marker_ids: list[str]
    unused_candidate_count: int
    camera_points_m: np.ndarray
    plane: dict[str, Any] | None
    warnings: list[str]
    reasons: list[str]
    ok: bool


@dataclass(frozen=True)
class TargetPoseCalibration:
    """Rigid target-board pose solved from detected camera-frame markers."""

    camera_T_target: np.ndarray
    marker_ids: list[str]
    camera_points_m: np.ndarray
    target_points_m: np.ndarray
    residuals_m: np.ndarray
    mean_residual_m: float
    max_residual_m: float
    rms_residual_m: float
    point_count: int
    ok: bool
    reasons: list[str]


def load_colored_point_cloud_ply(path: Path | str, *, source_units: str = "millimeters") -> ColoredPointCloud:
    """Load an ASCII PLY with x/y/z and RGB vertex properties.

    The Orbbec SDK `OB_FORMAT_RGB_POINT` sample writes millimeter xyz values plus
    uchar RGB.  This parser keeps only valid finite points and converts xyz to
    meters for downstream calibration.
    """

    ply_path = Path(path)
    if not ply_path.exists():
        raise FileNotFoundError(ply_path)
    vertex_count: int | None = None
    properties: list[str] = []
    current_element: str | None = None
    header_lines = 0
    format_seen = False
    with ply_path.open("r", encoding="utf-8", errors="replace") as handle:
        first = handle.readline().strip()
        header_lines += 1
        if first != "ply":
            raise ValueError(f"{ply_path} is not a PLY file")
        for line in handle:
            header_lines += 1
            stripped = line.strip()
            if stripped.startswith("format "):
                format_seen = True
                if "ascii" not in stripped:
                    raise ValueError(f"only ASCII PLY is supported, got header line {stripped!r}")
            elif stripped.startswith("element "):
                parts = stripped.split()
                current_element = parts[1] if len(parts) >= 2 else None
                if current_element == "vertex":
                    if len(parts) != 3:
                        raise ValueError(f"invalid vertex element line: {stripped!r}")
                    vertex_count = int(parts[2])
            elif stripped.startswith("property ") and current_element == "vertex":
                parts = stripped.split()
                if len(parts) >= 3:
                    properties.append(parts[-1])
            elif stripped == "end_header":
                break
    if not format_seen:
        raise ValueError(f"{ply_path} is missing a PLY format header")
    if vertex_count is None:
        raise ValueError(f"{ply_path} is missing element vertex")
    required = ["x", "y", "z", "red", "green", "blue"]
    missing = [name for name in required if name not in properties]
    if missing:
        raise ValueError(f"{ply_path} is missing vertex properties: {missing}")

    data = np.loadtxt(ply_path, dtype=np.float32, skiprows=header_lines, max_rows=vertex_count)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < len(properties):
        raise ValueError(f"{ply_path} data columns {data.shape[1]} < properties {len(properties)}")
    columns = {name: properties.index(name) for name in required}
    xyz = np.stack([data[:, columns[axis]] for axis in ("x", "y", "z")], axis=1)
    rgb = np.stack([data[:, columns[name]] for name in ("red", "green", "blue")], axis=1)

    units = str(source_units or "meters").lower()
    if units in {"millimeter", "millimeters", "mm"}:
        scale = 0.001
        normalized_units = "millimeters"
    elif units in {"meter", "meters", "m"}:
        scale = 1.0
        normalized_units = "meters"
    else:
        raise ValueError(f"unsupported colored point cloud units {source_units!r}; expected meters or millimeters")
    points_m = np.asarray(xyz, dtype=np.float32) * np.float32(scale)
    rgb = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 255.0)
    valid = np.isfinite(points_m).all(axis=1) & np.isfinite(rgb).all(axis=1)
    valid &= np.linalg.norm(points_m, axis=1) > 1e-8
    return ColoredPointCloud(
        points_m=np.ascontiguousarray(points_m[valid], dtype=np.float32),
        rgb=np.ascontiguousarray(rgb[valid], dtype=np.float32),
        source_path=ply_path,
        source_units=normalized_units,
    )


def load_configured_colored_point_cloud(config: dict[str, Any], override_path: Path | None = None) -> ColoredPointCloud:
    """Load the configured dynamic-table-marker colored point cloud."""

    marker_cfg = _marker_cfg(config)
    path = override_path or resolve_config_path(marker_cfg.get("colored_point_cloud_path"))
    if path is None:
        raise ValueError("set calibration.dynamic_table_markers.colored_point_cloud_path or pass --colored-point-cloud")
    source_units = str(marker_cfg.get("source_units") or get_nested(config, ("camera", "orbbec", "source_units"), "millimeters"))
    return load_colored_point_cloud_ply(path, source_units=source_units)


def detect_table_markers(config: dict[str, Any], cloud: ColoredPointCloud) -> TableMarkerDetection:
    """Detect configured colored table markers in a colored point cloud."""

    marker_cfg = _marker_cfg(config)
    points = np.asarray(cloud.points_m, dtype=np.float32)
    rgb = np.asarray(cloud.rgb, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"colored point cloud points must be (N,3), got {points.shape}")
    if rgb.shape != points.shape:
        raise ValueError(f"colored point cloud rgb must be (N,3), got {rgb.shape}")

    plane = _estimate_marker_plane(points, marker_cfg, config)
    marker_items = _configured_marker_items(marker_cfg)
    colors = sorted({str(item.get("color") or "").lower() for _, item in marker_items if item.get("color")})
    if not colors:
        colors = sorted(_default_color_thresholds().keys())

    hue, saturation, value = _rgb_to_hsv(rgb)
    plane_distances = None
    if plane is not None:
        normal = np.asarray(plane["plane_normal"], dtype=np.float32)
        offset = float(plane["plane_offset_m"])
        plane_distances = np.abs(points @ normal + offset)

    candidates: list[DetectedTableMarker] = []
    clustering_cfg = marker_cfg.get("clustering") or {}
    cluster_mode = str(clustering_cfg.get("mode", "combined_saturated")).lower()
    if cluster_mode in {"combined", "combined_saturated", "saturated"}:
        mask = saturation >= float(clustering_cfg.get("combined_min_saturation", 0.28))
        mask &= value >= float(clustering_cfg.get("combined_min_value", 0.16))
        indices = np.flatnonzero(mask)
        for component in _connected_components_2d(points[:, :2], indices, marker_cfg):
            color = _classify_component_color(hue[component])
            candidates.append(_component_to_marker(color, points, rgb, component, plane_distances, marker_cfg))
    elif cluster_mode == "per_color":
        for color in colors:
            mask = _color_mask(color, hue, saturation, value, rgb, marker_cfg)
            indices = np.flatnonzero(mask)
            for component in _connected_components_2d(points[:, :2], indices, marker_cfg):
                candidates.append(_component_to_marker(color, points, rgb, component, plane_distances, marker_cfg))
    else:
        raise ValueError("calibration.dynamic_table_markers.clustering.mode must be combined_saturated or per_color")

    markers, missing, warnings = _match_configured_markers(marker_items, candidates, marker_cfg)
    camera_points = np.asarray([item.camera_point_m for item in markers], dtype=np.float64).reshape((-1, 3)) if markers else np.zeros((0, 3), dtype=np.float64)
    min_marker_count = int(marker_cfg.get("min_marker_count", 3))
    reasons: list[str] = []
    if plane is None:
        reasons.append("table_plane_missing")
    if missing:
        reasons.append("markers_missing:" + ",".join(missing))
    if len(markers) < min_marker_count:
        reasons.append(f"too_few_markers:{len(markers)}<{min_marker_count}")
    unused = max(0, sum(1 for candidate in candidates if candidate.accepted) - len(markers))
    return TableMarkerDetection(
        configured_marker_ids=[marker_id for marker_id, _ in marker_items],
        marker_ids=[str(item.marker_id) for item in markers],
        markers=markers,
        candidates=sorted(candidates, key=lambda item: (item.color, item.camera_point_m[1], item.camera_point_m[0])),
        missing_marker_ids=missing,
        unused_candidate_count=unused,
        camera_points_m=np.ascontiguousarray(camera_points, dtype=np.float64),
        plane=plane,
        warnings=warnings,
        reasons=reasons or ["ok"],
        ok=not reasons,
    )


def build_robot_points_for_detection(
    config: dict[str, Any],
    detection: TableMarkerDetection,
    *,
    override_points: np.ndarray | None = None,
) -> np.ndarray:
    """Return robot-base marker points matching `detection.marker_ids` order."""

    if override_points is not None:
        arr = np.asarray(override_points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"override robot points must be (N,3), got {arr.shape}")
        if arr.shape[0] != len(detection.marker_ids):
            raise ValueError(f"override robot points count {arr.shape[0]} != detected marker count {len(detection.marker_ids)}")
        if not np.isfinite(arr).all():
            raise ValueError("override robot points contain non-finite values")
        return np.ascontiguousarray(arr, dtype=np.float64)

    marker_cfg = _marker_cfg(config)
    markers = {marker_id: item for marker_id, item in _configured_marker_items(marker_cfg)}
    origin, axis_x, axis_y = _board_geometry(marker_cfg)
    robot_points: list[np.ndarray] = []
    missing: list[str] = []
    for marker_id in detection.marker_ids:
        item = markers.get(marker_id, {})
        explicit = item.get("robot_base_xyz_m")
        point = _parse_xyz(explicit)
        if point is None:
            board_xy = item.get("board_xy_m")
            if origin is not None and axis_x is not None and axis_y is not None and board_xy is not None:
                xy = np.asarray(board_xy, dtype=np.float64).reshape(-1)
                if xy.shape[0] != 2 or not np.isfinite(xy).all():
                    raise ValueError(f"marker {marker_id} board_xy_m must contain two finite values")
                point = origin + xy[0] * axis_x + xy[1] * axis_y
        if point is None:
            missing.append(marker_id)
        else:
            robot_points.append(point)
    if missing:
        raise ValueError(
            "missing robot-base coordinates for markers "
            + ",".join(missing)
            + "; set robot_base_xyz_m, configure board geometry, or pass --robot-points"
        )
    return np.ascontiguousarray(np.vstack(robot_points), dtype=np.float64)


def solve_camera_T_target_from_detection(
    config: dict[str, Any],
    detection: TableMarkerDetection,
) -> TargetPoseCalibration:
    """Solve `camera_T_target` from detected markers and target geometry."""

    target_points = build_target_points_for_detection(config, detection)
    result = _solve_rigid_transform(target_points, detection.camera_points_m, name="camera_T_target")
    thresholds = _target_thresholds(config)
    reasons: list[str] = []
    if result.mean_residual_m > thresholds["max_mean_residual_m"]:
        reasons.append(
            f"mean_residual_high:{result.mean_residual_m:.6g}>{thresholds['max_mean_residual_m']:.6g}"
        )
    if result.max_residual_m > thresholds["max_max_residual_m"]:
        reasons.append(
            f"max_residual_high:{result.max_residual_m:.6g}>{thresholds['max_max_residual_m']:.6g}"
        )
    return TargetPoseCalibration(
        camera_T_target=result.transform,
        marker_ids=list(detection.marker_ids),
        camera_points_m=np.ascontiguousarray(detection.camera_points_m, dtype=np.float64),
        target_points_m=np.ascontiguousarray(target_points, dtype=np.float64),
        residuals_m=result.residuals_m,
        mean_residual_m=result.mean_residual_m,
        max_residual_m=result.max_residual_m,
        rms_residual_m=result.rms_residual_m,
        point_count=result.point_count,
        ok=not reasons,
        reasons=reasons or ["ok"],
    )


def build_target_points_for_detection(config: dict[str, Any], detection: TableMarkerDetection) -> np.ndarray:
    """Return target-frame marker points matching `detection.marker_ids`."""

    target_cfg = _hand_eye_target_cfg(config)
    markers = target_cfg.get("markers") or {}
    if not isinstance(markers, dict):
        raise ValueError("calibration.hand_eye.target.markers must be a mapping")
    target_points: list[np.ndarray] = []
    missing: list[str] = []
    for marker_id in detection.marker_ids:
        item = markers.get(marker_id, {})
        if not isinstance(item, dict):
            missing.append(marker_id)
            continue
        point = _parse_xyz(item.get("target_xyz_m"))
        if point is None:
            missing.append(marker_id)
        else:
            target_points.append(point)
    if missing:
        raise ValueError(
            "missing target-frame coordinates for markers "
            + ",".join(missing)
            + "; set calibration.hand_eye.target.markers.<id>.target_xyz_m"
        )
    arr = np.ascontiguousarray(np.vstack(target_points), dtype=np.float64)
    min_marker_count = int(target_cfg.get("min_marker_count", 3))
    if arr.shape[0] < min_marker_count:
        raise ValueError(f"too few target markers: {arr.shape[0]}<{min_marker_count}")
    if np.linalg.matrix_rank(arr - arr.mean(axis=0, keepdims=True)) < 2:
        raise ValueError("target marker points are degenerate; use non-collinear target markers")
    return arr


def target_pose_summary(result: TargetPoseCalibration) -> dict[str, Any]:
    """Return JSON-friendly target-pose diagnostics."""

    return {
        "ok": result.ok,
        "reasons": result.reasons,
        "marker_ids": result.marker_ids,
        "camera_T_target": result.camera_T_target.astype(float).tolist(),
        "camera_points_m": result.camera_points_m.astype(float).tolist(),
        "target_points_m": result.target_points_m.astype(float).tolist(),
        "point_count": result.point_count,
        "mean_residual_m": result.mean_residual_m,
        "max_residual_m": result.max_residual_m,
        "rms_residual_m": result.rms_residual_m,
        "residuals_m": result.residuals_m.astype(float).tolist(),
    }


def pairwise_distance_errors(camera_points: np.ndarray, robot_points: np.ndarray, marker_ids: list[str]) -> list[dict[str, Any]]:
    """Compare camera/robot pairwise distances for correspondence sanity checks."""

    camera = np.asarray(camera_points, dtype=np.float64)
    robot = np.asarray(robot_points, dtype=np.float64)
    if camera.shape != robot.shape:
        raise ValueError(f"camera points shape {camera.shape} != robot points shape {robot.shape}")
    rows: list[dict[str, Any]] = []
    for i in range(camera.shape[0]):
        for j in range(i + 1, camera.shape[0]):
            camera_dist = float(np.linalg.norm(camera[i] - camera[j]))
            robot_dist = float(np.linalg.norm(robot[i] - robot[j]))
            rows.append(
                {
                    "marker_a": marker_ids[i],
                    "marker_b": marker_ids[j],
                    "camera_distance_m": camera_dist,
                    "robot_distance_m": robot_dist,
                    "error_m": abs(camera_dist - robot_dist),
                }
            )
    return rows


def detection_summary(detection: TableMarkerDetection) -> dict[str, Any]:
    """Return a JSON-friendly marker detection summary."""

    return {
        "ok": detection.ok,
        "reasons": detection.reasons,
        "configured_marker_ids": detection.configured_marker_ids,
        "marker_ids": detection.marker_ids,
        "missing_marker_ids": detection.missing_marker_ids,
        "warnings": detection.warnings,
        "used_marker_count": len(detection.markers),
        "candidate_count": len(detection.candidates),
        "unused_candidate_count": detection.unused_candidate_count,
        "camera_points_m": detection.camera_points_m.astype(float).tolist(),
        "plane": detection.plane,
        "markers": [_marker_to_dict(item) for item in detection.markers],
        "candidates": [_marker_to_dict(item) for item in detection.candidates],
    }


@dataclass(frozen=True)
class _RigidFit:
    transform: np.ndarray
    residuals_m: np.ndarray
    mean_residual_m: float
    max_residual_m: float
    rms_residual_m: float
    point_count: int


def _solve_rigid_transform(source_points: np.ndarray, target_points: np.ndarray, *, name: str) -> _RigidFit:
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"source_points must be (N,3), got {source.shape}")
    if target.shape != source.shape:
        raise ValueError(f"target_points shape {target.shape} != source_points shape {source.shape}")
    if source.shape[0] < 3:
        raise ValueError("at least 3 point correspondences are required")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("point correspondences contain non-finite values")
    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    source_centered = source - source_centroid[None, :]
    target_centered = target - target_centroid[None, :]
    if np.linalg.matrix_rank(source_centered) < 2 or np.linalg.matrix_rank(target_centered) < 2:
        raise ValueError("point correspondences are degenerate; use non-collinear 3D points")
    covariance = source_centered.T @ target_centered
    u, _, vh = np.linalg.svd(covariance)
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0:
        vh[-1, :] *= -1.0
        rotation = vh.T @ u.T
    translation = target_centroid - rotation @ source_centroid
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    predicted = (rotation @ source.T).T + translation[None, :]
    residuals = np.linalg.norm(predicted - target, axis=1)
    return _RigidFit(
        transform=np.ascontiguousarray(transform, dtype=np.float64),
        residuals_m=np.ascontiguousarray(residuals, dtype=np.float64),
        mean_residual_m=float(residuals.mean()),
        max_residual_m=float(residuals.max()),
        rms_residual_m=float(np.sqrt(np.mean(residuals**2))),
        point_count=int(source.shape[0]),
    )


def _hand_eye_target_cfg(config: dict[str, Any]) -> dict[str, Any]:
    cfg = get_nested(config, ("calibration", "hand_eye", "target"), {}) or {}
    if not isinstance(cfg, dict):
        raise ValueError("calibration.hand_eye.target must be a mapping")
    return cfg


def _target_thresholds(config: dict[str, Any]) -> dict[str, float]:
    target_cfg = _hand_eye_target_cfg(config)
    thresholds = target_cfg.get("thresholds") if isinstance(target_cfg.get("thresholds"), dict) else {}
    return {
        "max_mean_residual_m": float(thresholds.get("max_mean_residual_m", target_cfg.get("max_mean_residual_m", 0.010))),
        "max_max_residual_m": float(thresholds.get("max_max_residual_m", target_cfg.get("max_max_residual_m", 0.020))),
    }


def _marker_cfg(config: dict[str, Any]) -> dict[str, Any]:
    cfg = get_nested(config, ("calibration", "dynamic_table_markers"), {}) or {}
    if not isinstance(cfg, dict):
        raise ValueError("calibration.dynamic_table_markers must be a mapping")
    return cfg


def _configured_marker_items(marker_cfg: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    markers = marker_cfg.get("markers") or {}
    if not isinstance(markers, dict):
        raise ValueError("calibration.dynamic_table_markers.markers must be a mapping")
    items: list[tuple[str, dict[str, Any]]] = []
    for marker_id, raw in markers.items():
        if not isinstance(raw, dict):
            raise ValueError(f"marker {marker_id!r} config must be a mapping")
        if not bool(raw.get("enabled", True)):
            continue
        items.append((str(marker_id), raw))
    return items


def _estimate_marker_plane(points: np.ndarray, marker_cfg: dict[str, Any], config: dict[str, Any]) -> dict[str, Any] | None:
    plane_cfg = marker_cfg.get("plane") or {}
    if not isinstance(plane_cfg, dict):
        raise ValueError("calibration.dynamic_table_markers.plane must be a mapping")
    return estimate_dominant_plane(
        points,
        threshold_m=float(plane_cfg.get("distance_threshold_m", 0.012)),
        iterations=int(plane_cfg.get("ransac_iterations", 256)),
        sample_points=int(plane_cfg.get("ransac_sample_points", 4096)),
        evaluation_points=int(plane_cfg.get("ransac_evaluation_points", 50000)),
        seed=int(plane_cfg.get("seed", get_nested(config, ("runtime", "seed"), 0))),
    )


def _component_to_marker(
    color: str,
    points: np.ndarray,
    rgb: np.ndarray,
    component: np.ndarray,
    plane_distances: np.ndarray | None,
    marker_cfg: dict[str, Any],
) -> DetectedTableMarker:
    comp_points = points[component]
    comp_rgb = rgb[component]
    center_mode = str(marker_cfg.get("center", "mean")).lower()
    if center_mode == "median":
        camera_point = np.median(comp_points, axis=0)
    elif center_mode == "mean":
        camera_point = comp_points.mean(axis=0)
    else:
        raise ValueError("calibration.dynamic_table_markers.center must be 'mean' or 'median'")
    median_point = np.median(comp_points, axis=0)
    extent = comp_points.max(axis=0) - comp_points.min(axis=0)
    rgb_mean = comp_rgb.mean(axis=0)
    rgb_median = np.median(comp_rgb, axis=0)

    plane_distance = None
    plane_median = None
    plane_p90 = None
    reasons: list[str] = []
    if plane_distances is None:
        reasons.append("table_plane_missing")
    else:
        distances = plane_distances[component]
        plane_distance = float(np.median(distances))
        plane_median = float(np.median(distances))
        plane_p90 = float(np.percentile(distances, 90.0))
        max_plane = float(get_nested(marker_cfg, ("plane", "max_marker_plane_distance_m"), 0.010))
        max_plane_p90 = float(get_nested(marker_cfg, ("plane", "max_marker_plane_distance_p90_m"), max(0.018, max_plane * 1.8)))
        if plane_median > max_plane:
            reasons.append("far_from_table_plane")
        if plane_p90 > max_plane_p90:
            reasons.append("plane_distance_spread_high")

    clustering_cfg = marker_cfg.get("clustering") or {}
    max_extent = np.asarray(clustering_cfg.get("max_extent_m", [0.070, 0.070, 0.060]), dtype=np.float32)
    if max_extent.shape != (3,):
        raise ValueError("calibration.dynamic_table_markers.clustering.max_extent_m must have three values")
    if np.any(extent > max_extent):
        reasons.append("component_extent_too_large")
    max_points = clustering_cfg.get("max_points")
    if max_points is not None and int(component.shape[0]) > int(max_points):
        reasons.append("component_too_large")

    accepted = not reasons
    point_count_score = min(1.0, float(component.shape[0]) / max(float(clustering_cfg.get("target_points", 600)), 1.0))
    plane_score = 0.5 if plane_median is None else max(0.0, 1.0 - plane_median / max(float(get_nested(marker_cfg, ("plane", "max_marker_plane_distance_m"), 0.010)), 1e-6))
    extent_score = max(0.0, 1.0 - float(np.max(extent / np.maximum(max_extent, 1e-6))))
    confidence = float(0.45 * point_count_score + 0.40 * plane_score + 0.15 * extent_score)
    return DetectedTableMarker(
        marker_id=None,
        color=color,
        camera_point_m=np.ascontiguousarray(camera_point, dtype=np.float64),
        median_point_m=np.ascontiguousarray(median_point, dtype=np.float64),
        point_count=int(component.shape[0]),
        extent_m=np.ascontiguousarray(extent, dtype=np.float64),
        rgb_mean=np.ascontiguousarray(rgb_mean, dtype=np.float64),
        rgb_median=np.ascontiguousarray(rgb_median, dtype=np.float64),
        plane_distance_m=plane_distance,
        plane_distance_median_m=plane_median,
        plane_distance_p90_m=plane_p90,
        confidence=confidence,
        accepted=accepted,
        reason="ok" if accepted else ";".join(reasons),
    )


def _match_configured_markers(
    marker_items: list[tuple[str, dict[str, Any]]],
    candidates: list[DetectedTableMarker],
    marker_cfg: dict[str, Any],
) -> tuple[list[DetectedTableMarker], list[str], list[str]]:
    accepted_by_color: dict[str, list[DetectedTableMarker]] = {}
    for candidate in candidates:
        if candidate.accepted:
            accepted_by_color.setdefault(candidate.color, []).append(candidate)
    groups: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for marker_id, item in marker_items:
        color = str(item.get("color") or "").lower()
        if not color:
            raise ValueError(f"marker {marker_id} must configure a color")
        groups.setdefault(color, []).append((marker_id, item))

    matched: list[DetectedTableMarker] = []
    missing: list[str] = []
    warnings: list[str] = []
    used_candidate_ids: set[int] = set()
    for color, group in groups.items():
        color_candidates = list(accepted_by_color.get(color, []))
        if not color_candidates:
            missing.extend(marker_id for marker_id, _ in group)
            continue
        axis = _axis_index(str(marker_cfg.get("duplicate_color_match_axis", "x")))
        reverse = bool(marker_cfg.get("duplicate_color_match_reverse", False))
        if len(group) == 1:
            sorted_candidates = sorted(color_candidates, key=lambda item: (-item.confidence, item.camera_point_m[axis]))
        else:
            sorted_candidates = sorted(color_candidates, key=lambda item: item.camera_point_m[axis], reverse=reverse)
            if all("expected_camera_xyz_m" in item for _, item in group):
                warnings.append(f"duplicate_color_{color}_matched_by_expected_camera_hint")
            elif any("match_index" not in item for _, item in group):
                warnings.append(f"duplicate_color_{color}_matched_by_{'xyz'[axis]}_{'desc' if reverse else 'asc'}")
        used_indices_for_color: set[int] = set()
        for order, (marker_id, item) in enumerate(group):
            if "match_index" in item:
                match_index = int(item.get("match_index", order))
            elif "expected_camera_xyz_m" in item:
                expected = np.asarray(item["expected_camera_xyz_m"], dtype=np.float64).reshape(3)
                distances = [
                    np.inf if index in used_indices_for_color else np.linalg.norm(candidate.camera_point_m - expected)
                    for index, candidate in enumerate(sorted_candidates)
                ]
                match_index = int(np.argmin(distances)) if distances else order
            else:
                available = [index for index in range(len(sorted_candidates)) if index not in used_indices_for_color]
                match_index = available[0] if available else order
            if match_index < 0 or match_index >= len(sorted_candidates) or match_index in used_indices_for_color:
                missing.append(marker_id)
                continue
            candidate = sorted_candidates[match_index]
            used_indices_for_color.add(match_index)
            used_candidate_ids.add(id(candidate))
            matched.append(_copy_marker(candidate, marker_id=marker_id, match_index=match_index))
    if accepted_by_color:
        unused = sum(1 for items in accepted_by_color.values() for item in items if id(item) not in used_candidate_ids)
        if unused:
            warnings.append(f"unused_accepted_candidates:{unused}")
    return matched, missing, warnings


def _copy_marker(candidate: DetectedTableMarker, *, marker_id: str, match_index: int | None) -> DetectedTableMarker:
    return DetectedTableMarker(
        marker_id=marker_id,
        color=candidate.color,
        camera_point_m=candidate.camera_point_m,
        median_point_m=candidate.median_point_m,
        point_count=candidate.point_count,
        extent_m=candidate.extent_m,
        rgb_mean=candidate.rgb_mean,
        rgb_median=candidate.rgb_median,
        plane_distance_m=candidate.plane_distance_m,
        plane_distance_median_m=candidate.plane_distance_median_m,
        plane_distance_p90_m=candidate.plane_distance_p90_m,
        confidence=candidate.confidence,
        accepted=candidate.accepted,
        reason=candidate.reason,
        match_index=match_index,
    )


def _connected_components_2d(points_xy: np.ndarray, indices: np.ndarray, marker_cfg: dict[str, Any]) -> list[np.ndarray]:
    if indices.size == 0:
        return []
    clustering_cfg = marker_cfg.get("clustering") or {}
    cell_size = float(clustering_cfg.get("cell_size_m", 0.004))
    min_points = int(clustering_cfg.get("min_points", 40))
    if cell_size <= 0:
        raise ValueError("calibration.dynamic_table_markers.clustering.cell_size_m must be positive")
    xy = np.asarray(points_xy[indices], dtype=np.float32)
    origin = xy.min(axis=0)
    grid = np.floor((xy - origin[None, :]) / cell_size).astype(np.int32)
    cell_map: dict[tuple[int, int], list[int]] = {}
    for local_index, key in enumerate(map(tuple, grid)):
        cell_map.setdefault((int(key[0]), int(key[1])), []).append(local_index)
    visited: set[tuple[int, int]] = set()
    components: list[np.ndarray] = []
    neighbors = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    for key in list(cell_map):
        if key in visited:
            continue
        stack = [key]
        visited.add(key)
        local_indices: list[int] = []
        while stack:
            cx, cy = stack.pop()
            local_indices.extend(cell_map[(cx, cy)])
            for dx, dy in neighbors:
                next_key = (cx + dx, cy + dy)
                if next_key in cell_map and next_key not in visited:
                    visited.add(next_key)
                    stack.append(next_key)
        if len(local_indices) >= min_points:
            components.append(indices[np.asarray(local_indices, dtype=np.int64)])
    return components


def _rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.max(initial=0.0) > 1.5:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    r, g, b = arr[:, 0], arr[:, 1], arr[:, 2]
    mx = np.maximum.reduce([r, g, b])
    mn = np.minimum.reduce([r, g, b])
    diff = mx - mn
    saturation = np.where(mx > 1e-6, diff / np.maximum(mx, 1e-6), 0.0)
    hue = np.zeros_like(mx)
    mask = diff > 1e-6
    red = (mx == r) & mask
    green = (mx == g) & mask
    blue = (mx == b) & mask
    hue[red] = ((g[red] - b[red]) / diff[red]) % 6.0
    hue[green] = (b[green] - r[green]) / diff[green] + 2.0
    hue[blue] = (r[blue] - g[blue]) / diff[blue] + 4.0
    hue = hue / 6.0
    return hue, saturation, mx


def _classify_component_color(hue_values: np.ndarray) -> str:
    """Return a coarse color name from a component's circular mean hue."""

    hue = np.asarray(hue_values, dtype=np.float32).reshape(-1)
    if hue.size == 0:
        return "unknown"
    mean_angle = np.mean(np.exp(2j * np.pi * hue))
    hue_mean = float((np.angle(mean_angle) / (2.0 * np.pi)) % 1.0)
    if hue_mean < 0.04 or hue_mean >= 0.94:
        return "red"
    if hue_mean < 0.10:
        return "orange"
    if hue_mean < 0.18:
        return "yellow"
    if hue_mean < 0.42:
        return "green"
    if hue_mean < 0.55:
        return "cyan"
    if hue_mean < 0.70:
        return "blue"
    if hue_mean < 0.88:
        return "purple"
    return "magenta"


def _color_mask(
    color: str,
    hue: np.ndarray,
    saturation: np.ndarray,
    value: np.ndarray,
    rgb: np.ndarray,
    marker_cfg: dict[str, Any],
) -> np.ndarray:
    thresholds = _default_color_thresholds()
    overrides = marker_cfg.get("color_thresholds") or {}
    if isinstance(overrides, dict) and color in overrides and isinstance(overrides[color], dict):
        threshold = {**thresholds.get(color, {}), **overrides[color]}
    else:
        threshold = thresholds.get(color)
    if threshold is None:
        raise ValueError(f"no color threshold configured for marker color {color!r}")
    mask = np.zeros_like(hue, dtype=bool)
    for lo, hi in threshold.get("hue_ranges", []):
        lo_f = float(lo)
        hi_f = float(hi)
        if lo_f <= hi_f:
            mask |= (hue >= lo_f) & (hue < hi_f)
        else:
            mask |= (hue >= lo_f) | (hue < hi_f)
    mask &= saturation >= float(threshold.get("min_saturation", 0.25))
    mask &= value >= float(threshold.get("min_value", 0.15))
    rgb_norm = np.asarray(rgb, dtype=np.float32) / 255.0
    for channel, index in (("red", 0), ("green", 1), ("blue", 2)):
        min_value = threshold.get(f"min_{channel}")
        if min_value is not None:
            mask &= rgb_norm[:, index] >= float(min_value)
    return mask


def _default_color_thresholds() -> dict[str, dict[str, Any]]:
    return {
        "red": {"hue_ranges": [(0.0, 0.045), (0.94, 1.0)], "min_saturation": 0.35, "min_value": 0.18},
        "orange": {"hue_ranges": [(0.035, 0.10)], "min_saturation": 0.28, "min_value": 0.16},
        "yellow": {"hue_ranges": [(0.045, 0.18)], "min_saturation": 0.28, "min_value": 0.16},
        "green": {"hue_ranges": [(0.20, 0.48)], "min_saturation": 0.25, "min_value": 0.15},
        "cyan": {"hue_ranges": [(0.42, 0.55)], "min_saturation": 0.25, "min_value": 0.15},
        "blue": {"hue_ranges": [(0.48, 0.70)], "min_saturation": 0.25, "min_value": 0.15},
        "purple": {"hue_ranges": [(0.70, 0.88)], "min_saturation": 0.25, "min_value": 0.15},
        "magenta": {"hue_ranges": [(0.82, 0.96)], "min_saturation": 0.25, "min_value": 0.15},
    }


def _board_geometry(marker_cfg: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    origin = _parse_xyz(marker_cfg.get("board_origin_robot_base_m"))
    axis_x = _parse_xyz(marker_cfg.get("board_x_axis_robot_base"))
    axis_y = _parse_xyz(marker_cfg.get("board_y_axis_robot_base"))
    if origin is None or axis_x is None or axis_y is None:
        return None, None, None
    norm_x = float(np.linalg.norm(axis_x))
    norm_y = float(np.linalg.norm(axis_y))
    if norm_x <= 1e-8 or norm_y <= 1e-8:
        raise ValueError("board axes must be nonzero")
    return origin, axis_x / norm_x, axis_y / norm_y


def _parse_xyz(value: Any) -> np.ndarray | None:
    if value in (None, ""):
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"expected xyz list with three values, got {value!r}")
    if any(item is None for item in value):
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(3)
    if not np.isfinite(arr).all():
        raise ValueError(f"xyz contains non-finite values: {value!r}")
    return arr


def _axis_index(axis: str) -> int:
    mapping = {"x": 0, "0": 0, "y": 1, "1": 1, "z": 2, "2": 2}
    key = axis.lower()
    if key not in mapping:
        raise ValueError(f"unsupported match axis {axis!r}; expected x/y/z")
    return mapping[key]


def _marker_to_dict(marker: DetectedTableMarker) -> dict[str, Any]:
    return {
        "marker_id": marker.marker_id,
        "color": marker.color,
        "camera_point_m": marker.camera_point_m.astype(float).tolist(),
        "median_point_m": marker.median_point_m.astype(float).tolist(),
        "point_count": marker.point_count,
        "extent_m": marker.extent_m.astype(float).tolist(),
        "rgb_mean": marker.rgb_mean.astype(float).tolist(),
        "rgb_median": marker.rgb_median.astype(float).tolist(),
        "plane_distance_m": marker.plane_distance_m,
        "plane_distance_median_m": marker.plane_distance_median_m,
        "plane_distance_p90_m": marker.plane_distance_p90_m,
        "confidence": marker.confidence,
        "accepted": marker.accepted,
        "reason": marker.reason,
        "match_index": marker.match_index,
    }
