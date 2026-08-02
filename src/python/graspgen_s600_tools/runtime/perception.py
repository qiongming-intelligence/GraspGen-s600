"""Lightweight point-cloud segmentation helpers for real-grasp dry runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .hbm_graspgen_runtime import get_nested, validate_point_cloud


@dataclass(frozen=True)
class SegmentationResult:
    """Object/scene point clouds plus compact segmentation metadata."""

    object_point_cloud: np.ndarray
    scene_point_cloud: np.ndarray | None
    metadata: dict[str, Any]


def segment_object_point_cloud(
    *,
    object_point_cloud: np.ndarray | None,
    scene_point_cloud: np.ndarray | None,
    config: dict[str, Any],
    robot_base_T_camera: np.ndarray | None = None,
) -> SegmentationResult:
    """Return an object-focused cloud for GraspGen and scene cloud for safety checks.

    The first deployment path is intentionally deterministic and dependency-light:
    filter finite points, optionally crop in camera-frame meters, optionally remove
    a configured/estimated table plane, optionally keep the largest voxel cluster,
    and optionally cap the number of object points sent over ZMQ.  If segmentation
    is disabled, the configured object cloud is passed through after validation.

    `robot_base_T_camera` enables `table.frame: robot_base`, which is the only way
    to cut the table correctly: the table normal is a base-frame direction, and no
    single camera axis is parallel to it.
    """

    seg_cfg = get_nested(config, ("camera", "segmentation"), {}) or {}
    min_points = int(get_nested(config, ("safety", "min_points_before_resample"), 256))
    max_abs = float(get_nested(config, ("safety", "max_abs_coord_m"), 5.0))
    seed = int(get_nested(config, ("runtime", "seed"), 0))

    if not bool(seg_cfg.get("enabled", False)):
        source = object_point_cloud if object_point_cloud is not None else scene_point_cloud
        if source is None:
            raise RuntimeError("no point cloud available for camera capture")
        obj = validate_point_cloud(source, min_points=min_points, max_abs_coord_m=max_abs)
        metadata = _point_cloud_metadata(
            obj,
            prefix="object",
            extra={
                "segmentation_enabled": False,
                "segmentation_method": "passthrough",
                "scene_point_count": int(scene_point_cloud.shape[0]) if scene_point_cloud is not None else None,
                "scene_includes_object": scene_point_cloud is not None,
            },
        )
        return SegmentationResult(
            object_point_cloud=obj,
            scene_point_cloud=_validate_optional_scene(scene_point_cloud, max_abs=max_abs),
            metadata=metadata,
        )

    source_name = str(seg_cfg.get("source") or "scene_point_cloud")
    if source_name == "scene_point_cloud" and scene_point_cloud is not None:
        source = scene_point_cloud
    elif object_point_cloud is not None:
        source = object_point_cloud
        source_name = "object_point_cloud"
    elif scene_point_cloud is not None:
        source = scene_point_cloud
        source_name = "scene_point_cloud"
    else:
        raise RuntimeError("segmentation enabled but no point cloud source is available")

    source = validate_point_cloud(source, min_points=min_points, max_abs_coord_m=max_abs)
    metadata: dict[str, Any] = {
        "segmentation_enabled": True,
        "segmentation_method": str(seg_cfg.get("method") or "crop_table_cluster"),
        "segmentation_source": source_name,
        "source_point_count": int(source.shape[0]),
    }

    points = source
    points, crop_meta = _apply_crop(points, seg_cfg)
    metadata.update(crop_meta)
    if points.shape[0] < min_points:
        raise RuntimeError(
            f"segmentation crop left {points.shape[0]} points, need at least {min_points}"
        )

    points, table_meta = _apply_table_filter(
        points, seg_cfg, config=config, robot_base_T_camera=robot_base_T_camera
    )
    metadata.update(table_meta)
    if points.shape[0] < min_points:
        raise RuntimeError(
            f"table filtering left {points.shape[0]} object points, need at least {min_points}"
        )

    points, cluster_meta = _apply_cluster_filter(
        points, seg_cfg, config=config, robot_base_T_camera=robot_base_T_camera
    )
    metadata.update(cluster_meta)
    if points.shape[0] < min_points:
        raise RuntimeError(
            f"cluster filtering left {points.shape[0]} object points, need at least {min_points}"
        )

    _validate_object_extent(points, seg_cfg)
    max_object_points = seg_cfg.get("max_object_points")
    if max_object_points is not None:
        points = _sample_if_needed(points, int(max_object_points), seed=seed)
        metadata["max_object_points"] = int(max_object_points)

    points = validate_point_cloud(points, min_points=min_points, max_abs_coord_m=max_abs)
    metadata.update(_point_cloud_metadata(points, prefix="object"))
    metadata["scene_point_count"] = int(scene_point_cloud.shape[0]) if scene_point_cloud is not None else None
    metadata["scene_includes_object"] = scene_point_cloud is not None
    return SegmentationResult(
        object_point_cloud=points,
        scene_point_cloud=_validate_optional_scene(scene_point_cloud, max_abs=max_abs),
        metadata=metadata,
    )


def _validate_optional_scene(scene_point_cloud: np.ndarray | None, *, max_abs: float) -> np.ndarray | None:
    if scene_point_cloud is None:
        return None
    pc = np.asarray(scene_point_cloud, dtype=np.float32)
    if pc.ndim != 2 or pc.shape[1] < 3:
        raise ValueError(f"scene point cloud must be (N, >=3), got {pc.shape}")
    pc = pc[:, :3]
    finite = np.isfinite(pc).all(axis=1)
    pc = pc[finite]
    if pc.size and np.abs(pc).max() > max_abs:
        raise ValueError(f"scene point cloud coordinate exceeds configured max_abs_coord_m={max_abs}")
    return np.ascontiguousarray(pc, dtype=np.float32)


def _apply_crop(points: np.ndarray, seg_cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    mask = np.ones(points.shape[0], dtype=bool)
    crop_cfg = seg_cfg.get("crop_bounds_m") or seg_cfg.get("crop_bounds") or {}
    applied: dict[str, list[float | None]] = {}
    if isinstance(crop_cfg, dict):
        for axis_name in ("x", "y", "z"):
            bounds = crop_cfg.get(axis_name)
            if bounds is None:
                continue
            lo, hi = _parse_bounds(bounds)
            axis = _axis_index(axis_name)
            if lo is not None:
                mask &= points[:, axis] >= lo
            if hi is not None:
                mask &= points[:, axis] <= hi
            applied[axis_name] = [lo, hi]
    depth_range = seg_cfg.get("depth_range_m")
    if depth_range is not None:
        lo, hi = _parse_bounds(depth_range)
        if lo is not None:
            mask &= points[:, 2] >= lo
        if hi is not None:
            mask &= points[:, 2] <= hi
        applied["z"] = [lo, hi]
    cropped = np.ascontiguousarray(points[mask], dtype=np.float32)
    return cropped, {
        "crop_bounds_m": applied or None,
        "crop_input_count": int(points.shape[0]),
        "crop_output_count": int(cropped.shape[0]),
    }


def _apply_table_filter(
    points: np.ndarray,
    seg_cfg: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
    robot_base_T_camera: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    table_cfg = seg_cfg.get("table") or {}
    if not bool(table_cfg.get("enabled", False)):
        return points, {"table_filter_enabled": False}

    if str(table_cfg.get("frame") or "camera") == "robot_base":
        return _apply_base_frame_table_filter(
            points, table_cfg, config=config, robot_base_T_camera=robot_base_T_camera
        )

    axis = _axis_index(str(table_cfg.get("axis") or "z"))
    threshold = float(table_cfg.get("threshold_m", table_cfg.get("distance_threshold_m", 0.008)))
    side = str(table_cfg.get("object_side") or "above")
    plane_value = table_cfg.get("plane_value_m")
    estimated = False
    if plane_value is None:
        percentile = float(table_cfg.get("estimate_percentile", 5.0 if side == "above" else 95.0))
        plane = float(np.percentile(points[:, axis], percentile))
        estimated = True
    else:
        plane = float(plane_value)

    if side == "above":
        mask = points[:, axis] >= plane + threshold
    elif side == "below":
        mask = points[:, axis] <= plane - threshold
    else:
        raise ValueError("camera.segmentation.table.object_side must be 'above' or 'below'")
    filtered = np.ascontiguousarray(points[mask], dtype=np.float32)
    return filtered, {
        "table_filter_enabled": True,
        "table_plane_axis": int(axis),
        "table_plane_axis_name": "xyz"[axis],
        "table_plane_value_m": plane,
        "table_plane_estimated": estimated,
        "table_threshold_m": threshold,
        "table_object_side": side,
        "table_filter_input_count": int(points.shape[0]),
        "table_filter_output_count": int(filtered.shape[0]),
    }


def _apply_base_frame_table_filter(
    points: np.ndarray,
    table_cfg: dict[str, Any],
    *,
    config: dict[str, Any] | None,
    robot_base_T_camera: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Cut the table using the calibrated base-frame plane, not a camera axis.

    Filtering on a camera axis cannot work here: the table normal is base +Z and
    no camera axis is parallel to it, so camera z is depth. Measured on the
    2026-07-29 scene, a camera-z cut at the 5th percentile + 8mm kept 92.6% of
    the cloud, i.e. it removed nothing but the nearest sliver, and GraspGen was
    handed the whole table.

    This fails closed: without a trusted transform it raises rather than silently
    degrading to the camera-axis behaviour, since a wrong table cut sends grasps
    into the surface.
    """

    if robot_base_T_camera is None:
        raise RuntimeError(
            "camera.segmentation.table.frame is 'robot_base' but no robot_base_T_camera "
            "was supplied; the planning transform is required to cut the table"
        )
    matrix = np.asarray(robot_base_T_camera, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"robot_base_T_camera must be (4, 4), got {matrix.shape}")

    height = table_cfg.get("height_m")
    if height is None and config is not None:
        height = get_nested(config, ("safety", "table", "height_m"), None)
    if height is None:
        raise RuntimeError(
            "base-frame table filtering needs a plane height: set "
            "camera.segmentation.table.height_m or safety.table.height_m"
        )
    threshold = float(table_cfg.get("threshold_m", table_cfg.get("distance_threshold_m", 0.008)))
    side = str(table_cfg.get("object_side") or "above")
    axis = _axis_index(str(table_cfg.get("base_axis") or "z"))

    base = (matrix[:3, :3] @ points.T).T + matrix[:3, 3]
    if side == "above":
        mask = base[:, axis] >= float(height) + threshold
    elif side == "below":
        mask = base[:, axis] <= float(height) - threshold
    else:
        raise ValueError("camera.segmentation.table.object_side must be 'above' or 'below'")
    filtered = np.ascontiguousarray(points[mask], dtype=np.float32)
    return filtered, {
        "table_filter_enabled": True,
        "table_filter_frame": "robot_base",
        "table_plane_axis_name": "xyz"[axis],
        "table_plane_value_m": float(height),
        "table_plane_estimated": False,
        "table_threshold_m": threshold,
        "table_object_side": side,
        "table_filter_input_count": int(points.shape[0]),
        "table_filter_output_count": int(filtered.shape[0]),
    }


def _apply_cluster_filter(
    points: np.ndarray,
    seg_cfg: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
    robot_base_T_camera: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    cluster_cfg = seg_cfg.get("cluster") or {}
    if not bool(cluster_cfg.get("enabled", False)):
        return points, {"cluster_filter_enabled": False}
    voxel_size = float(cluster_cfg.get("voxel_size_m", 0.015))
    min_cluster_points = int(cluster_cfg.get("min_points", 1))
    select = str(cluster_cfg.get("select") or "most_points")

    # Cluster in the base frame when it is available: voxel adjacency is
    # frame-dependent, and "which cluster is reachable" is only meaningful there.
    cluster_points = points
    frame = "camera"
    if robot_base_T_camera is not None and str(cluster_cfg.get("frame") or "robot_base") == "robot_base":
        matrix = np.asarray(robot_base_T_camera, dtype=np.float64)
        cluster_points = np.ascontiguousarray(
            (matrix[:3, :3] @ points.T).T + matrix[:3, 3], dtype=np.float32
        )
        frame = "robot_base"

    meta: dict[str, Any] = {
        "cluster_filter_enabled": True,
        "cluster_voxel_size_m": voxel_size,
        "cluster_frame": frame,
        "cluster_select": select,
        "cluster_input_count": int(points.shape[0]),
    }

    if select == "most_points_in_workspace" and frame == "robot_base" and config is not None:
        labels, select_meta = _workspace_cluster_mask(
            cluster_points, voxel_size_m=voxel_size, config=config
        )
        meta.update(select_meta)
    else:
        if select == "most_points_in_workspace":
            # Fail closed rather than silently falling back: picking the biggest
            # blob when the caller asked for the reachable one can hand GraspGen a
            # distant object it will place unreachable grasps on.
            raise RuntimeError(
                "cluster.select is 'most_points_in_workspace' but base-frame clustering "
                "is unavailable (needs robot_base_T_camera and cluster.frame robot_base)"
            )
        labels = _largest_voxel_cluster_mask(cluster_points, voxel_size_m=voxel_size)

    filtered = np.ascontiguousarray(points[labels], dtype=np.float32)
    if filtered.shape[0] < min_cluster_points:
        raise RuntimeError(
            f"selected cluster has {filtered.shape[0]} points, below configured min_points={min_cluster_points}"
        )
    meta["cluster_output_count"] = int(filtered.shape[0])
    return filtered, meta


def _voxel_clusters(points: np.ndarray, *, voxel_size_m: float) -> tuple[np.ndarray, list[list[int]], np.ndarray]:
    """Return (voxel inverse index, connected voxel components, points-per-voxel)."""

    if voxel_size_m <= 0:
        raise ValueError("cluster.voxel_size_m must be positive")
    voxels = np.floor(points / voxel_size_m).astype(np.int32)
    unique, inverse = np.unique(voxels, axis=0, return_inverse=True)
    inverse = np.reshape(inverse, -1)
    voxel_to_id = {tuple(v.tolist()): i for i, v in enumerate(unique)}
    seen = np.zeros(unique.shape[0], dtype=bool)
    neighbor_offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]
    components: list[list[int]] = []
    for start in range(unique.shape[0]):
        if seen[start]:
            continue
        seen[start] = True
        queue = [start]
        cluster: list[int] = []
        while queue:
            cur = queue.pop()
            cluster.append(cur)
            base = unique[cur]
            for offset in neighbor_offsets:
                key = (int(base[0] + offset[0]), int(base[1] + offset[1]), int(base[2] + offset[2]))
                nxt = voxel_to_id.get(key)
                if nxt is not None and not seen[nxt]:
                    seen[nxt] = True
                    queue.append(nxt)
        components.append(cluster)
    return inverse, components, np.bincount(inverse, minlength=unique.shape[0])


def _workspace_cluster_mask(
    base_points: np.ndarray, *, voxel_size_m: float, config: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Pick the cluster with the most points inside the reachable workspace box.

    Size alone is the wrong criterion on this rig: the observation pose sees well
    past the arm's reach, so the largest off-plane blob is often a distant table
    edge. A cluster the arm cannot reach is useless regardless of how big it is.
    """

    bounds = get_nested(config, ("safety", "workspace_bounds"), {}) or {}
    inverse, components, per_voxel = _voxel_clusters(base_points, voxel_size_m=voxel_size_m)

    inside = np.ones(base_points.shape[0], dtype=bool)
    for axis_name in ("x", "y", "z"):
        limits = bounds.get(axis_name)
        if limits is None:
            continue
        lo, hi = _parse_bounds(limits)
        axis = _axis_index(axis_name)
        if lo is not None:
            inside &= base_points[:, axis] >= lo
        if hi is not None:
            inside &= base_points[:, axis] <= hi
    inside_per_voxel = np.bincount(inverse[inside], minlength=per_voxel.shape[0]) if inside.any() else None

    best: list[int] = []
    best_key = (-1, -1)
    for cluster in components:
        idx = np.asarray(cluster, dtype=np.int64)
        total = int(per_voxel[idx].sum())
        in_box = int(inside_per_voxel[idx].sum()) if inside_per_voxel is not None else 0
        if (in_box, total) > best_key:
            best_key = (in_box, total)
            best = cluster
    if not best:
        return np.zeros(base_points.shape[0], dtype=bool), {"cluster_count": 0}
    if best_key[0] <= 0:
        raise RuntimeError(
            f"no off-plane cluster has any point inside safety.workspace_bounds "
            f"({len(components)} clusters examined, largest {best_key[1]} points); "
            "the object is outside the arm's reachable box"
        )
    mask = np.isin(inverse, np.asarray(best, dtype=np.int64))
    return mask, {
        "cluster_count": len(components),
        "cluster_selected_points_in_workspace": int(best_key[0]),
        "cluster_selected_total_points": int(best_key[1]),
    }


def enumerate_object_clusters(
    *,
    scene_point_cloud: np.ndarray,
    config: dict[str, Any],
    robot_base_T_camera: np.ndarray,
    min_points: int | None = None,
) -> list[dict[str, Any]]:
    """List every reachable off-table cluster, best-first, for sequential grasping.

    `segment_object_point_cloud` deliberately collapses the scene to ONE object
    because the inference path takes a single cloud. Picking objects up one at a
    time needs the whole list instead, so this walks the same base-frame table cut
    and voxel clustering and returns a per-cluster record rather than a mask.

    Ordering is `(points_in_workspace, total_points)` descending, matching
    `_workspace_cluster_mask` so the first entry here is the cluster that function
    would have chosen. Each record carries the geometry a caller needs to decide
    whether the gripper can actually take it:

    - `narrowest_width_m`: the minimum spread over table-plane closing directions,
      i.e. the aperture the jaws must exceed. Compare against
      `safety.collision.finger_opening_m` (the Dofbot's real inner-face maximum is
      59.7mm, so anything wider cannot be grasped at any approach angle).
    - `points_in_workspace` / `centroid_m`: reachability, since the camera/reach
      overlap on this rig is a narrow band.

    Clusters are NOT filtered by graspability here - callers see everything and
    decide, so an object that is merely too wide is reported rather than hidden.
    """

    if robot_base_T_camera is None:
        raise ValueError("enumerate_object_clusters requires robot_base_T_camera")
    seg_cfg = get_nested(config, ("camera", "segmentation"), {}) or {}
    cluster_cfg = seg_cfg.get("cluster") or {}
    voxel_size_m = float(cluster_cfg.get("voxel_size_m", 0.015))
    floor = int(min_points if min_points is not None else cluster_cfg.get("min_points", 256))

    points = np.asarray(scene_point_cloud, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    table_cfg = seg_cfg.get("table") or {}
    camera_points, _ = _apply_base_frame_table_filter(
        points, table_cfg, config=config, robot_base_T_camera=robot_base_T_camera
    )
    camera_points = np.asarray(camera_points, dtype=np.float64)
    base = (robot_base_T_camera[:3, :3] @ camera_points.T).T + robot_base_T_camera[:3, 3]
    if base.shape[0] == 0:
        return []

    inverse, components, per_voxel = _voxel_clusters(base, voxel_size_m=voxel_size_m)
    bounds = get_nested(config, ("safety", "workspace_bounds"), {}) or {}
    inside = np.ones(base.shape[0], dtype=bool)
    for axis_name in ("x", "y", "z"):
        limits = bounds.get(axis_name)
        if limits is None:
            continue
        lo, hi = _parse_bounds(limits)
        axis = _axis_index(axis_name)
        if lo is not None:
            inside &= base[:, axis] >= lo
        if hi is not None:
            inside &= base[:, axis] <= hi
    inside_per_voxel = (
        np.bincount(inverse[inside], minlength=per_voxel.shape[0]) if inside.any() else None
    )

    angles = np.linspace(0.0, np.pi, 90, endpoint=False)
    directions = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)

    records: list[dict[str, Any]] = []
    for cluster in components:
        idx = np.asarray(cluster, dtype=np.int64)
        total = int(per_voxel[idx].sum())
        if total < floor:
            continue
        mask = np.isin(inverse, idx)
        pts_base = base[mask]
        in_box = int(inside_per_voxel[idx].sum()) if inside_per_voxel is not None else 0
        widths = np.ptp(pts_base @ directions.T, axis=0)
        narrow = int(np.argmin(widths))
        records.append(
            {
                # Camera-frame points for THIS cluster only, ready to hand to
                # inference as `object_point_cloud`. A mask is not returned because
                # it would index the table-filtered subset, not the caller's cloud.
                "points_camera": np.ascontiguousarray(camera_points[mask], dtype=np.float32),
                "point_count": total,
                "points_in_workspace": in_box,
                "centroid_m": pts_base.mean(axis=0).tolist(),
                "bbox_min_m": pts_base.min(axis=0).tolist(),
                "bbox_max_m": pts_base.max(axis=0).tolist(),
                "height_m": float(np.ptp(pts_base[:, 2])),
                "narrowest_width_m": float(widths[narrow]),
                "narrowest_direction": directions[narrow].tolist(),
            }
        )
    records.sort(key=lambda r: (r["points_in_workspace"], r["point_count"]), reverse=True)
    return records


def subdivide_cluster(
    record: dict[str, Any],
    *,
    robot_base_T_camera: np.ndarray,
    patch_size_m: float = 0.030,
    min_points: int = 200,
) -> list[dict[str, Any]]:
    """Tile a too-wide cluster into gripper-sized patches for separate inference.

    Needed because touching objects merge into one cluster and cannot be separated
    geometrically. Tried and failed on abutting ~25mm cubes: voxel sizes 3/4/5mm with
    both 4- and 26-connectivity, top-slice clustering over 2/4/6mm bands, and
    density-threshold components at 25/35/50% of the median all return ONE component.
    Depth noise at this range is millimetres and the seams are thinner.

    Tiling sidesteps the problem. The model samples grasps relative to the cloud it
    is GIVEN, so handing it the whole 88x70mm group makes it propose grasps at the
    group's centre - measured live, all 20 candidates landed centrally and 19 were
    rejected because the finger walls press on a neighbouring cube. An offline sweep
    found 431 poses over the same footprint that pass both collision and material
    checks, all near the edges. The candidates existed; the model was never asked
    about them.

    Patches overlap by half a patch so a grasp site straddling a tile boundary is not
    lost. Returns records in the same shape as `enumerate_object_clusters`, so
    callers can treat them interchangeably.
    """

    pts_cam = np.asarray(record["points_camera"], dtype=np.float64)
    if pts_cam.shape[0] == 0:
        return []
    base = (robot_base_T_camera[:3, :3] @ pts_cam.T).T + robot_base_T_camera[:3, 3]

    step = patch_size_m / 2.0
    angles = np.linspace(0.0, np.pi, 90, endpoint=False)
    directions = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)

    out: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    xs = np.arange(base[:, 0].min(), base[:, 0].max() + step, step)
    ys = np.arange(base[:, 1].min(), base[:, 1].max() + step, step)
    for gx in xs:
        for gy in ys:
            mask = (
                (base[:, 0] >= gx)
                & (base[:, 0] < gx + patch_size_m)
                & (base[:, 1] >= gy)
                & (base[:, 1] < gy + patch_size_m)
            )
            count = int(mask.sum())
            if count < min_points:
                continue
            # Overlapping tiles can select an identical point set; keep one copy.
            key = (count, *np.round(base[mask].mean(axis=0), 4).tolist())
            if key in seen:
                continue
            seen.add(key)
            pb = base[mask]
            widths = np.ptp(pb @ directions.T, axis=0)
            narrow = int(np.argmin(widths))
            out.append(
                {
                    "points_camera": np.ascontiguousarray(pts_cam[mask], dtype=np.float32),
                    "point_count": count,
                    # Inherited: every patch comes from a cluster already known to be
                    # in the workspace, and a patch is a subset of that footprint.
                    "points_in_workspace": count,
                    "centroid_m": pb.mean(axis=0).tolist(),
                    "bbox_min_m": pb.min(axis=0).tolist(),
                    "bbox_max_m": pb.max(axis=0).tolist(),
                    "height_m": float(np.ptp(pb[:, 2])),
                    "narrowest_width_m": float(widths[narrow]),
                    "narrowest_direction": directions[narrow].tolist(),
                }
            )
    out.sort(key=lambda r: r["point_count"], reverse=True)
    return out


def _largest_voxel_cluster_mask(points: np.ndarray, *, voxel_size_m: float) -> np.ndarray:
    if voxel_size_m <= 0:
        raise ValueError("cluster.voxel_size_m must be positive")
    voxels = np.floor(points / voxel_size_m).astype(np.int32)
    unique, inverse = np.unique(voxels, axis=0, return_inverse=True)
    # numpy 2 returns an inverse shaped like the input for axis-wise unique.
    inverse = np.reshape(inverse, -1)
    voxel_to_id = {tuple(v.tolist()): i for i, v in enumerate(unique)}
    # Rank clusters by POINT count, not voxel count. A far, obliquely-viewed
    # surface is sampled sparsely and spreads over many nearly-empty voxels, so
    # a voxel-count ranking picks it over a near, dense object: measured on the
    # 2026-07-29 scene, the distant table edge won with 64 voxels / 5194 points
    # over the actual object at 59 voxels / 17170 points.
    points_per_voxel = np.bincount(inverse, minlength=unique.shape[0])
    seen = np.zeros(unique.shape[0], dtype=bool)
    best_cluster: list[int] = []
    best_points = -1
    neighbor_offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]
    for start in range(unique.shape[0]):
        if seen[start]:
            continue
        seen[start] = True
        queue = [start]
        cluster: list[int] = []
        while queue:
            cur = queue.pop()
            cluster.append(cur)
            base = unique[cur]
            for offset in neighbor_offsets:
                key = (int(base[0] + offset[0]), int(base[1] + offset[1]), int(base[2] + offset[2]))
                nxt = voxel_to_id.get(key)
                if nxt is not None and not seen[nxt]:
                    seen[nxt] = True
                    queue.append(nxt)
        cluster_points = int(points_per_voxel[np.asarray(cluster, dtype=np.int64)].sum())
        if cluster_points > best_points:
            best_points = cluster_points
            best_cluster = cluster
    if not best_cluster:
        return np.zeros(points.shape[0], dtype=bool)
    best = np.zeros(unique.shape[0], dtype=bool)
    best[np.asarray(best_cluster, dtype=np.int64)] = True
    return best[inverse]


def _validate_object_extent(points: np.ndarray, seg_cfg: dict[str, Any]) -> None:
    extent = points.max(axis=0) - points.min(axis=0)
    min_extent = seg_cfg.get("object_extent_min_m")
    if min_extent is not None:
        lo = np.asarray(min_extent, dtype=np.float32)
        if lo.shape != (3,):
            raise ValueError("camera.segmentation.object_extent_min_m must have 3 values")
        if np.any(extent < lo):
            raise RuntimeError(f"segmented object extent {extent.tolist()} is below minimum {lo.tolist()}")
    max_extent = seg_cfg.get("object_extent_max_m")
    if max_extent is not None:
        hi = np.asarray(max_extent, dtype=np.float32)
        if hi.shape != (3,):
            raise ValueError("camera.segmentation.object_extent_max_m must have 3 values")
        if np.any(extent > hi):
            raise RuntimeError(f"segmented object extent {extent.tolist()} exceeds maximum {hi.tolist()}")


def _sample_if_needed(points: np.ndarray, max_points: int, *, seed: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return np.ascontiguousarray(points, dtype=np.float32)
    rng = np.random.default_rng(seed)
    indices = rng.choice(points.shape[0], size=max_points, replace=False)
    return np.ascontiguousarray(points[indices], dtype=np.float32)


def _point_cloud_metadata(
    points: np.ndarray,
    *,
    prefix: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = {
        f"{prefix}_point_count": int(points.shape[0]),
        f"{prefix}_bounds_min": points.min(axis=0).astype(float).tolist(),
        f"{prefix}_bounds_max": points.max(axis=0).astype(float).tolist(),
        f"{prefix}_centroid": points.mean(axis=0).astype(float).tolist(),
    }
    if extra:
        meta.update(extra)
    return meta


def _parse_bounds(value: Any) -> tuple[float | None, float | None]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"bounds must be a two-item list, got {value!r}")
    lo = None if value[0] is None else float(value[0])
    hi = None if value[1] is None else float(value[1])
    if lo is not None and hi is not None and lo > hi:
        raise ValueError(f"lower bound {lo} exceeds upper bound {hi}")
    return lo, hi


def _axis_index(axis: str) -> int:
    mapping = {"x": 0, "0": 0, "y": 1, "1": 1, "z": 2, "2": 2}
    key = str(axis).lower()
    if key not in mapping:
        raise ValueError(f"unsupported axis {axis!r}; expected x/y/z")
    return mapping[key]
