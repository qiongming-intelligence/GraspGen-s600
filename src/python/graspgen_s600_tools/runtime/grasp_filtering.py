"""Client-side GraspGen candidate filtering for no-motion planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .hbm_graspgen_runtime import get_nested


@dataclass(frozen=True)
class CandidateSelection:
    """Filtered candidate table plus selected grasp, if any."""

    selected_index: int | None
    selected_grasp_camera: np.ndarray | None
    selected_grasp_robot: np.ndarray | None
    table: list[dict[str, Any]]
    safety: dict[str, Any]


def select_grasp_candidate(
    response: dict[str, Any],
    *,
    config: dict[str, Any],
    robot_base_T_camera: np.ndarray | None,
    scene_point_cloud: np.ndarray | None,
    scene_metadata: dict[str, Any] | None = None,
) -> CandidateSelection:
    """Filter/rank GraspGen candidates without commanding robot motion."""

    grasps_camera = np.asarray(response.get("grasps", []), dtype=np.float32)
    confidences = np.asarray(response.get("confidences", []), dtype=np.float32)
    ranked = np.asarray(response.get("ranked_indices", []), dtype=np.int64)
    if grasps_camera.ndim != 3 or grasps_camera.shape[1:] != (4, 4):
        raise ValueError(f"response grasps must be (M,4,4), got {grasps_camera.shape}")
    if ranked.size == 0:
        ranked = np.arange(grasps_camera.shape[0], dtype=np.int64)
    grasps_robot = None
    scene_robot = None
    if robot_base_T_camera is not None and grasps_camera.size:
        grasps_robot = np.ascontiguousarray(robot_base_T_camera[None, :, :] @ grasps_camera, dtype=np.float32)
    if robot_base_T_camera is not None and scene_point_cloud is not None:
        scene_robot = _transform_points(scene_point_cloud, robot_base_T_camera)

    workspace_configured = _workspace_configured(config)
    camera_workspace_configured = _camera_workspace_configured(config)
    collision_required = bool(get_nested(config, ("safety", "require_collision_check_for_execute"), True))
    reachability_required = bool(get_nested(config, ("safety", "require_reachability_check_for_execute"), True))
    table_cfg = get_nested(config, ("safety", "table"), {}) or {}
    approach_cfg = get_nested(config, ("safety", "approach"), {}) or {}
    reachability_cfg = get_nested(config, ("safety", "reachability"), {}) or {}
    collision_cfg = get_nested(config, ("safety", "collision"), {}) or {}

    table: list[dict[str, Any]] = []
    selected_index: int | None = None
    for rank, idx_raw in enumerate(ranked):
        idx = int(idx_raw)
        if idx < 0 or idx >= grasps_camera.shape[0]:
            continue
        grasp_camera = grasps_camera[idx]
        grasp_robot = grasps_robot[idx] if grasps_robot is not None else None
        score = float(confidences[idx]) if idx < confidences.shape[0] else float("nan")
        row = _candidate_row(
            idx=idx,
            rank=rank,
            confidence=score,
            grasp_camera=grasp_camera,
            grasp_robot=grasp_robot,
            config=config,
            workspace_configured=workspace_configured,
            camera_workspace_configured=camera_workspace_configured,
            table_cfg=table_cfg,
            approach_cfg=approach_cfg,
            reachability_cfg=reachability_cfg,
            collision_cfg=collision_cfg,
            scene_point_cloud=scene_robot,
            scene_metadata=scene_metadata or {},
            collision_required=collision_required,
            reachability_required=reachability_required,
        )
        if row["accepted"] and selected_index is None:
            selected_index = idx
            row["selected"] = True
        table.append(row)

    selected_camera = grasps_camera[selected_index] if selected_index is not None else None
    selected_robot = grasps_robot[selected_index] if selected_index is not None and grasps_robot is not None else None
    selected_row = next((row for row in table if row.get("selected")), None)
    safety = _selection_safety(
        table=table,
        selected_row=selected_row,
        workspace_configured=workspace_configured,
        camera_workspace_configured=camera_workspace_configured,
        calibration_loaded=robot_base_T_camera is not None,
        collision_required=collision_required,
        reachability_required=reachability_required,
    )
    return CandidateSelection(
        selected_index=selected_index,
        selected_grasp_camera=selected_camera,
        selected_grasp_robot=selected_robot,
        table=table,
        safety=safety,
    )


def _candidate_row(
    *,
    idx: int,
    rank: int,
    confidence: float,
    grasp_camera: np.ndarray,
    grasp_robot: np.ndarray | None,
    config: dict[str, Any],
    workspace_configured: bool,
    camera_workspace_configured: bool,
    table_cfg: dict[str, Any],
    approach_cfg: dict[str, Any],
    reachability_cfg: dict[str, Any],
    collision_cfg: dict[str, Any],
    scene_point_cloud: np.ndarray | None,
    scene_metadata: dict[str, Any],
    collision_required: bool,
    reachability_required: bool,
) -> dict[str, Any]:
    reasons: list[str] = []
    matrix_ok, matrix_reason = _matrix_ok(grasp_camera)
    if not matrix_ok:
        reasons.append(matrix_reason)

    if grasp_robot is None:
        robot_translation = None
        reasons.append("calibration_missing")
    else:
        robot_translation = grasp_robot[:3, 3].astype(float).tolist()

    camera_workspace_ok = True
    if camera_workspace_configured:
        camera_workspace_ok = _translation_in_camera_workspace(grasp_camera[:3, 3], config)
        if not camera_workspace_ok:
            reasons.append("outside_camera_workspace")

    workspace_ok = True
    if workspace_configured and grasp_robot is not None:
        workspace_ok = _translation_in_workspace(grasp_robot[:3, 3], config)
        if not workspace_ok:
            reasons.append("outside_workspace")
    elif not workspace_configured:
        workspace_ok = False
        reasons.append("workspace_unconfigured")

    table_checked, table_ok, table_reason = _table_clearance_ok(grasp_robot, table_cfg, scene_metadata)
    if table_checked and not table_ok:
        reasons.append(table_reason)

    approach_checked, approach_ok, approach_reason = _approach_ok(grasp_robot, approach_cfg)
    if approach_checked and not approach_ok:
        reasons.append(approach_reason)

    collision_checked, collision_ok, collision_reason = _collision_ok(
        grasp_robot,
        scene_point_cloud,
        collision_cfg,
        collision_required=collision_required,
    )
    if collision_checked and not collision_ok:
        reasons.append(collision_reason)
    elif collision_required and not collision_checked:
        reasons.append(collision_reason)

    material_checked, material_ok, material_reason = _grasp_material_ok(
        grasp_robot,
        scene_point_cloud,
        collision_cfg,
    )
    if material_checked and not material_ok:
        reasons.append(material_reason)

    # Reachability runs LAST and its solve is by far the most expensive thing here:
    # measured on the Orange Pi 2026-07-31, `position_ik` is 4017ms per candidate
    # against 22ms for each volume gate, i.e. 81.2s of the 82s a 20-candidate target
    # took. A candidate that has already failed another gate is rejected no matter what
    # IK returns, so the solve is skipped and reported as not-checked. Ordering is what
    # makes this sound: nothing below consumes the reachability verdict.
    if reasons:
        reachability_checked = False
        reachability_ok = False
        reachability_reason = "ik_skipped_already_rejected"
    else:
        reachability_checked, reachability_ok, reachability_reason = _reachability_ok(
            grasp_robot,
            reachability_cfg,
            config,
        )
        if reachability_checked and not reachability_ok:
            reasons.append(reachability_reason)
        elif reachability_required and not reachability_checked:
            reasons.append(reachability_reason)

    reasons = list(dict.fromkeys(reasons))
    accepted = not reasons
    return {
        "index": idx,
        "rank": rank,
        "confidence": confidence,
        "camera_translation_m": grasp_camera[:3, 3].astype(float).tolist(),
        "robot_translation_m": robot_translation,
        "matrix_ok": matrix_ok,
        "camera_workspace_checked": bool(camera_workspace_configured),
        "camera_workspace_ok": bool(camera_workspace_ok),
        "workspace_checked": bool(workspace_configured),
        "workspace_ok": bool(workspace_ok),
        "table_checked": table_checked,
        "table_ok": table_ok,
        "approach_checked": approach_checked,
        "approach_ok": approach_ok,
        "grasp_material_checked": material_checked,
        "grasp_material_ok": material_ok,
        "collision_checked": collision_checked,
        "collision_ok": collision_ok,
        "reachability_checked": reachability_checked,
        "reachability_ok": reachability_ok,
        "accepted": accepted,
        "selected": False,
        "reason": "ok" if accepted else ";".join(reasons),
    }


def _matrix_ok(grasp: np.ndarray) -> tuple[bool, str]:
    if grasp.shape != (4, 4):
        return False, "bad_matrix_shape"
    if not np.isfinite(grasp).all():
        return False, "nonfinite_matrix"
    if not np.allclose(grasp[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-3):
        return False, "bad_homogeneous_row"
    rot = grasp[:3, :3]
    if not np.allclose(rot.T @ rot, np.eye(3), atol=5e-2):
        return False, "rotation_not_orthonormal"
    det = float(np.linalg.det(rot))
    if not 0.9 <= det <= 1.1:
        return False, "rotation_bad_determinant"
    return True, "ok"


def _workspace_configured(config: dict[str, Any]) -> bool:
    bounds = get_nested(config, ("safety", "workspace_bounds"), {}) or {}
    if not isinstance(bounds, dict):
        return False
    for axis in ("x", "y", "z"):
        value = bounds.get(axis)
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return False
        if value[0] is None or value[1] is None:
            return False
    return True


def _translation_in_workspace(translation: np.ndarray, config: dict[str, Any]) -> bool:
    bounds = get_nested(config, ("safety", "workspace_bounds"), {}) or {}
    for axis, name in enumerate(("x", "y", "z")):
        lo = float(bounds[name][0])
        hi = float(bounds[name][1])
        if float(translation[axis]) < lo or float(translation[axis]) > hi:
            return False
    return True


def _camera_workspace_configured(config: dict[str, Any]) -> bool:
    bounds = get_nested(config, ("safety", "camera_workspace_bounds"), None)
    if bounds is None:
        bounds = get_nested(config, ("camera", "segmentation", "crop_bounds_m"), None)
    if not isinstance(bounds, dict):
        return False
    for axis in ("x", "y", "z"):
        value = bounds.get(axis)
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return False
        if value[0] is None or value[1] is None:
            return False
    return True


def _translation_in_camera_workspace(translation: np.ndarray, config: dict[str, Any]) -> bool:
    bounds = get_nested(config, ("safety", "camera_workspace_bounds"), None)
    if bounds is None:
        bounds = get_nested(config, ("camera", "segmentation", "crop_bounds_m"), {}) or {}
    for axis, name in enumerate(("x", "y", "z")):
        lo = float(bounds[name][0])
        hi = float(bounds[name][1])
        if float(translation[axis]) < lo or float(translation[axis]) > hi:
            return False
    return True


def _table_clearance_ok(
    grasp_robot: np.ndarray | None,
    table_cfg: dict[str, Any],
    scene_metadata: dict[str, Any],
) -> tuple[bool, bool, str]:
    if grasp_robot is None:
        return False, False, "calibration_missing"
    enabled = bool(table_cfg.get("enabled", False)) or table_cfg.get("height_m") is not None
    if not enabled:
        return False, False, "table_check_disabled"
    axis = _axis_index(str(table_cfg.get("axis") or "z"))
    height = table_cfg.get("height_m")
    if height is None:
        return True, False, "table_height_missing"
    min_clearance = float(table_cfg.get("min_clearance_m", 0.015))
    clearance = float(grasp_robot[axis, 3]) - float(height)
    if clearance < min_clearance:
        return True, False, "below_table_clearance"
    return True, True, "ok"


def _approach_ok(
    grasp_robot: np.ndarray | None,
    approach_cfg: dict[str, Any],
) -> tuple[bool, bool, str]:
    if grasp_robot is None:
        return False, False, "calibration_missing"
    if not bool(approach_cfg.get("enabled", False)):
        return False, False, "approach_check_disabled"
    # Which gripper-local axis is the approach direction, and its sign, is a
    # property of the gripper model and must be stated rather than assumed. For
    # GraspGen's Robotiq convention the local +Z is the approach direction and it
    # points INTO the object, so a top-down grasp has it pointing down in the base
    # frame: the test is on -Z, hence approach_axis_sign of -1.
    axis_index = _axis_index(str(approach_cfg.get("approach_axis") or "z"))
    axis_sign = float(approach_cfg.get("approach_axis_sign", -1.0))
    min_dot = float(approach_cfg.get("min_top_down_dot", 0.5))
    approach_axis = grasp_robot[:3, axis_index] * axis_sign
    norm = float(np.linalg.norm(approach_axis))
    if norm <= 1e-6:
        return True, False, "bad_approach_axis"
    dot = float(approach_axis[2] / norm)
    if dot < min_dot:
        return True, False, "approach_not_top_down"
    return True, True, "ok"


def _collision_ok(
    grasp_robot: np.ndarray | None,
    scene_point_cloud: np.ndarray | None,
    collision_cfg: dict[str, Any],
    *,
    collision_required: bool,
) -> tuple[bool, bool, str]:
    enabled = bool(collision_cfg.get("enabled", False))
    if not enabled:
        return False, not collision_required, "collision_check_disabled"
    if grasp_robot is None:
        return False, False, "calibration_missing"
    if scene_point_cloud is None:
        return True, False, "scene_cloud_missing"
    scene = np.asarray(scene_point_cloud, dtype=np.float32)
    if scene.ndim != 2 or scene.shape[1] != 3 or scene.shape[0] == 0:
        return True, False, "scene_cloud_invalid"

    mode = str(collision_cfg.get("mode") or "origin_proximity")
    if mode == "gripper_volume":
        return _gripper_volume_collision_ok(grasp_robot, scene, collision_cfg)

    # Legacy proximity test, kept only for configs that still select it.
    #
    # WARNING: this rejects physically correct grasps. GraspGen's grasp origin sits
    # BETWEEN the fingers, so on a real grasp the object is supposed to be there:
    # measured 2026-07-29, the best candidates had scene points 1.9-5.8mm from the
    # origin and 1087 points inside a 10mm radius. It only looked plausible while
    # every candidate landed on bare table. Prefer mode 'gripper_volume'.
    radius = float(collision_cfg.get("reject_radius_m", 0.01))
    max_points = int(collision_cfg.get("max_points_in_radius", 0))
    distances = np.linalg.norm(scene - grasp_robot[:3, 3][None, :], axis=1)
    near = int(np.count_nonzero(distances < radius))
    if near > max_points:
        return True, False, "scene_points_near_grasp"
    return True, True, "ok"


def _gripper_bound_radius(collision_cfg: dict[str, Any]) -> float:
    """Half-extent of an origin-centred box containing every gripper volume tested.

    The collision and material gates both work in the gripper's local frame, whose
    rotation is arbitrary, so the only frame-independent bound is a box (or sphere)
    around the grasp origin. Taking the widest local extent on each axis and combining
    them gives a radius no tested point can lie outside, whatever the orientation.
    """

    half_open = float(collision_cfg.get("finger_opening_m", 0.030)) / 2.0
    finger_len = float(collision_cfg.get("finger_length_m", 0.035))
    finger_thick = float(collision_cfg.get("finger_thickness_m", 0.010))
    palm_depth = float(collision_cfg.get("palm_depth_m", 0.025))
    palm_half_y = float(collision_cfg.get("palm_half_width_m", 0.020))
    margin = float(collision_cfg.get("margin_m", 0.003))
    origin_to_tip = float(collision_cfg.get("origin_to_tip_m", 0.0))

    extent_x = half_open + finger_thick + margin
    extent_y = palm_half_y + margin
    # Depth spans the palm box behind the tips through the far end of the jaws.
    extent_z = max(
        abs(origin_to_tip - palm_depth - margin),
        abs(origin_to_tip + finger_len + margin),
    )
    return float(np.sqrt(extent_x**2 + extent_y**2 + extent_z**2))


def _scene_near_grasp(
    grasp_robot: np.ndarray,
    scene: np.ndarray,
    collision_cfg: dict[str, Any],
) -> np.ndarray:
    """Cut the scene down to points that could possibly touch the gripper.

    WHY. Both volume gates rotate the WHOLE scene into the gripper frame, and the
    scene here is the full capture - 169,840 points, on purpose, so a grasp on one
    block sees the blocks beside it. That is a (3,N) matmul plus several N-by-3
    temporaries per candidate, twice, and a 20-candidate target measured 80.5s of gate
    time against 0.5s of inference on the Orange Pi.

    Points farther from the grasp origin than `_gripper_bound_radius` cannot be inside
    any tested volume in any orientation, so dropping them cannot change a verdict.
    The test is per-axis against precomputed bounds, which avoids materialising the
    difference array at all.
    """

    radius = _gripper_bound_radius(collision_cfg)
    origin = grasp_robot[:3, 3]
    keep = (scene[:, 0] >= origin[0] - radius) & (scene[:, 0] <= origin[0] + radius)
    keep &= (scene[:, 1] >= origin[1] - radius) & (scene[:, 1] <= origin[1] + radius)
    keep &= (scene[:, 2] >= origin[2] - radius) & (scene[:, 2] <= origin[2] + radius)
    return scene[keep]


def _gripper_volume_collision_ok(
    grasp_robot: np.ndarray,
    scene: np.ndarray,
    collision_cfg: dict[str, Any],
) -> tuple[bool, bool, str]:
    """Reject only when scene geometry hits the gripper's own solid volume.

    The correct question is not "is anything near the grasp point" - the object is
    meant to be there - but "would the fingers or palm have to pass through
    something". Two boxes are tested in the gripper frame, both configured from
    measured gripper geometry:

      * the palm/backplate behind the finger tips, which must be clear
      * the swept finger volume, which must be clear of anything WIDER than the
        opening: points between the fingers are the object being grasped, points
        outside the finger span would knock the gripper off

    Axes follow GraspGen's Robotiq convention: local +Z is the approach direction
    pointing into the object, and the fingers close along local X.

    Where the origin sits along +Z matters and is configured, not assumed. Upstream's
    asset frame puts the pose origin at the gripper mount with the contact plane
    `depth` ahead of it, and our r3_so3 decode regresses translation as
    `vec * scale + centroid`, i.e. relative to the object centroid.

    DIRECTION CORRECTED 2026-07-30. This function previously measured depth as
    `origin_to_tip - local_z`, i.e. it swept the volume along -Z, on the belief that
    the origin was the fingertip plane with the body behind it. Measured against the
    live model output, that is backwards: for all 20 candidates the local +Z axis
    points DOWN in the base frame (z component -0.74 to -0.93, i.e. into the table)
    and the object points lie ahead along +Z, not behind - e.g. 1956 points within
    30mm along +Z versus 378 along -Z. `_approach_ok` already documented +Z as the
    into-object direction, so the two disagreed and this one was wrong.

    Consequence of the old sign: the swept volume sat in the empty space BEHIND the
    grasp, so the gate almost never found a real collision and its rejections were
    not measuring what they claimed.

    Depth is now `local_z + origin_to_tip`, growing from the origin along the
    approach direction. `origin_to_tip_m` remains the configured offset from the pose
    origin forward to the fingertip plane.
    """

    # Discard points that cannot reach any tested volume before the rotation, which is
    # what makes this affordable on the full scene cloud. See `_scene_near_grasp`.
    near = _scene_near_grasp(grasp_robot, scene, collision_cfg)
    if near.shape[0] == 0:
        return True, True, "ok"
    inv_rot = grasp_robot[:3, :3].T
    local = (inv_rot @ (near - grasp_robot[:3, 3][None, :]).T).T

    half_open = float(collision_cfg.get("finger_opening_m", 0.030)) / 2.0
    finger_len = float(collision_cfg.get("finger_length_m", 0.035))
    finger_thick = float(collision_cfg.get("finger_thickness_m", 0.010))
    palm_depth = float(collision_cfg.get("palm_depth_m", 0.025))
    palm_half_y = float(collision_cfg.get("palm_half_width_m", 0.020))
    margin = float(collision_cfg.get("margin_m", 0.003))
    max_points = int(collision_cfg.get("max_points_in_volume", 0))
    # Distance from the pose origin forward to the fingertip plane. 0 means the
    # origin IS the tip plane, which is what the r3_so3 path produces.
    origin_to_tip = float(collision_cfg.get("origin_to_tip_m", 0.0))

    ax, ay, az = np.abs(local[:, 0]), np.abs(local[:, 1]), local[:, 2]
    # Depth along the approach direction, measured from the fingertip plane: 0 at the
    # tips, growing as the jaws sweep INTO the object. The jaws span [0, finger_len];
    # the palm sits behind the tips, at negative depth. Sign corrected 2026-07-30 -
    # see the docstring.
    back = az - origin_to_tip

    # Fingers: the two solid jaws. Material inside the opening is the object being
    # grasped and must NOT count; only the jaw walls themselves collide.
    in_jaw_span = (back >= -margin) & (back <= finger_len + margin) & (ay <= palm_half_y + margin)
    finger_hit = in_jaw_span & (ax >= half_open - margin) & (ax <= half_open + finger_thick + margin)

    # Palm: the backplate behind the jaws, i.e. at NEGATIVE depth now that depth
    # grows into the object. Anything here would be struck before the fingers ever
    # closed, so it is a hard reject regardless of lateral position. (Before the
    # 2026-07-30 sign fix this tested `back > finger_len`, which with the old
    # inverted depth put the palm box on the far side of the object.)
    palm_hit = (
        (back < -margin)
        & (back >= -(palm_depth + margin))
        & (ax <= half_open + finger_thick + margin)
        & (ay <= palm_half_y + margin)
    )

    hits = int(np.count_nonzero(palm_hit | finger_hit))
    if hits > max_points:
        if int(np.count_nonzero(palm_hit)) > max_points:
            return True, False, "scene_points_in_gripper_palm"
        return True, False, "scene_points_in_finger_volume"
    return True, True, "ok"


def _grasp_material_ok(
    grasp_robot: np.ndarray | None,
    scene: np.ndarray | None,
    collision_cfg: dict[str, Any],
) -> tuple[bool, bool, str]:
    """Check the jaws would actually close on something.

    The collision gate only asks whether the gripper would STRIKE anything. It
    deliberately ignores material between the jaws, because that is the object being
    grasped. Nothing was asking the complementary question - is there any object
    between the jaws at all? - so a pose hovering above the object passed every gate
    while closing on air.

    Measured on the live scene 2026-07-30: the selected candidate sat 51.0mm above
    the table with the object top at 33.1mm, so the fingertip plane was ~18mm clear
    of the object and the 30mm jaws reached only to 21mm - still above the top face.
    All four gates returned ok and the count of scene points between the jaws was
    ZERO, at every yaw.

    Counts points inside the closing volume: within the jaw opening laterally, inside
    the finger length along the approach axis, inside the palm width across. Requires
    at least `min_points_between_jaws` of them.
    """

    min_points = int(collision_cfg.get("min_points_between_jaws", 0))
    if min_points <= 0:
        return False, True, "grasp_material_unchecked"
    if grasp_robot is None:
        return True, False, "calibration_missing"
    if scene is None or scene.size == 0:
        return True, False, "grasp_material_no_scene"

    half_open = float(collision_cfg.get("finger_opening_m", 0.030)) / 2.0
    finger_len = float(collision_cfg.get("finger_length_m", 0.035))
    palm_half_y = float(collision_cfg.get("palm_half_width_m", 0.020))
    origin_to_tip = float(collision_cfg.get("origin_to_tip_m", 0.0))

    # Same bounding-box prefilter as the collision gate: the closing volume is a subset
    # of the bound, so points outside it cannot be between the jaws.
    near = _scene_near_grasp(grasp_robot, scene, collision_cfg)
    if near.shape[0] == 0:
        return True, False, "no_object_between_jaws"

    local = (grasp_robot[:3, :3].T @ (near - grasp_robot[:3, 3][None, :]).T).T
    # Same convention as the collision gate: depth from the fingertip plane along the
    # approach direction (local +Z), 0 at the tips and growing INTO the object.
    back = local[:, 2] - origin_to_tip
    between = (
        (back >= 0.0)
        & (back <= finger_len)
        & (np.abs(local[:, 0]) <= half_open)
        & (np.abs(local[:, 1]) <= palm_half_y)
    )
    count = int(np.count_nonzero(between))
    if count < min_points:
        return True, False, "no_object_between_jaws"
    return True, True, "ok"


def _reachability_ok(
    grasp_robot: np.ndarray | None,
    reachability_cfg: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> tuple[bool, bool, str]:
    if grasp_robot is None:
        return False, False, "calibration_missing"
    policy = str(reachability_cfg.get("policy") or "dry_run_only")
    if policy in {"dry_run_only", "unchecked"}:
        return False, True, "reachability_not_checked"
    if policy == "workspace_only":
        return True, True, "ok"
    if policy == "position_ik":
        if config is None:
            return True, False, "reachability_config_missing"
        from .kinematics import solve_position_ik

        # A failing IK solve is expensive (every restart runs to its iteration
        # cap), and on the Orange Pi that is ~22s per candidate. The configured
        # workspace box is already verified IK-reachable at every corner, so a
        # point outside it cannot be reachable and needs no solve to reject.
        # Note this is a cost shortcut, not a reachability claim: the box is
        # inscribed in the reachable set, so points outside it may still be
        # reachable. It is sound as a gate because such a point is already
        # rejected by the workspace gate, and fail-closed is the desired bias.
        if _workspace_configured(config) and not _translation_in_workspace(grasp_robot[:3, 3], config):
            return True, False, "ik_skipped_outside_workspace"
        try:
            result = solve_position_ik(
                config,
                grasp_robot[:3, 3],
                seed_servo_positions_deg=reachability_cfg.get("seed_servo_positions_deg"),
                tolerance_m=float(reachability_cfg.get("ik_tolerance_m", 0.005)),
                max_iterations=int(reachability_cfg.get("ik_max_iterations", 200)),
                restarts=int(reachability_cfg.get("ik_restarts", 6)),
            )
        except Exception as exc:  # noqa: BLE001 - a failed solve must fail closed
            return True, False, f"ik_error:{str(exc)[:60]}"
        if not result.ok:
            return True, False, "outside_ik_reachable_set"
        return True, True, "ok"
    if policy == "scripted_bins":
        bins = reachability_cfg.get("bins") or []
        radius = float(reachability_cfg.get("bin_radius_m", 0.03))
        if not bins:
            return True, False, "no_scripted_bins"
        xy = grasp_robot[:2, 3]
        for item in bins:
            center = item.get("center_xy_m") if isinstance(item, dict) else None
            if center is None or len(center) != 2:
                continue
            if float(np.linalg.norm(xy - np.asarray(center, dtype=np.float32))) <= radius:
                return True, True, "ok"
        return True, False, "outside_scripted_bins"
    return True, False, f"unsupported_reachability_policy:{policy}"


def _selection_safety(
    *,
    table: list[dict[str, Any]],
    selected_row: dict[str, Any] | None,
    workspace_configured: bool,
    camera_workspace_configured: bool,
    calibration_loaded: bool,
    collision_required: bool,
    reachability_required: bool,
) -> dict[str, Any]:
    accepted_count = sum(1 for row in table if row.get("accepted"))
    selected = selected_row is not None
    collision_checked = bool(selected_row and selected_row.get("collision_checked"))
    collision_ok = bool(selected_row and selected_row.get("collision_ok"))
    return {
        "client_candidate_filter_checked": True,
        "calibration_loaded": calibration_loaded,
        "camera_workspace_checked": bool(camera_workspace_configured),
        "camera_workspace_ok": bool(not table or any(row.get("camera_workspace_ok") for row in table)),
        "workspace_checked": bool(workspace_configured and selected),
        "workspace_ok": bool(selected_row and selected_row.get("workspace_ok")),
        "table_checked": bool(selected_row and selected_row.get("table_checked")),
        "table_ok": bool(selected_row and selected_row.get("table_ok")),
        "collision_checked": collision_checked,
        "collision_ok": collision_ok,
        "collision_required": collision_required,
        "reachability_checked": bool(selected_row and selected_row.get("reachability_checked")),
        "reachability_ok": bool(selected_row and selected_row.get("reachability_ok")),
        "reachability_required": reachability_required,
        "raw_candidate_count": len(table),
        "accepted_candidate_count": accepted_count,
        "reject_summary": _reject_summary(table),
        "selected_candidate_index": selected_row.get("index") if selected_row else None,
        "selected_candidate_reason": selected_row.get("reason") if selected_row else "no_safe_candidate",
        "motion_authorized": False,
    }


def _reject_summary(table: list[dict[str, Any]]) -> dict[str, int]:
    """Aggregate per-reason reject counts across all candidates.

    Each rejected candidate contributes one count per distinct reason token, so
    the summary explains which safety gate is blocking the most candidates
    without printing every row. Accepted rows are ignored.
    """

    counts: dict[str, int] = {}
    for row in table:
        if row.get("accepted"):
            continue
        reason = str(row.get("reason") or "")
        seen: set[str] = set()
        for token in reason.split(";"):
            token = token.strip()
            if not token or token == "ok" or token in seen:
                continue
            seen.add(token)
            counts[token] = counts.get(token, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    pc = np.asarray(points, dtype=np.float32)
    if pc.ndim != 2 or pc.shape[1] != 3:
        raise ValueError(f"scene point cloud must be (N,3), got {pc.shape}")
    hom = np.concatenate([pc, np.ones((pc.shape[0], 1), dtype=np.float32)], axis=1)
    out = hom @ np.asarray(transform, dtype=np.float32).T
    return np.ascontiguousarray(out[:, :3], dtype=np.float32)


def _axis_index(axis: str) -> int:
    mapping = {"x": 0, "0": 0, "y": 1, "1": 1, "z": 2, "2": 2}
    key = str(axis).lower()
    if key not in mapping:
        raise ValueError(f"unsupported axis {axis!r}; expected x/y/z")
    return mapping[key]
