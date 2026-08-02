"""Config-driven serial-chain forward kinematics for Dofbot/Yahboom arms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .hbm_graspgen_runtime import get_nested


@dataclass(frozen=True)
class ForwardKinematicsResult:
    """A validated base-to-gripper transform computed from servo readback."""

    base_T_gripper: np.ndarray
    base_frame: str | None
    gripper_frame: str | None
    servo_positions_deg: dict[int, float]
    used_servo_ids: list[int]


def forward_kinematics(config: dict[str, Any], servo_positions_deg: dict[Any, Any]) -> np.ndarray:
    """Return `base_T_gripper` for a configured serial chain."""

    return forward_kinematics_with_metadata(config, servo_positions_deg).base_T_gripper


def forward_kinematics_with_metadata(
    config: dict[str, Any],
    servo_positions_deg: dict[Any, Any],
) -> ForwardKinematicsResult:
    """Compute FK from raw Arm_Lib servo degrees using `robot.kinematics`.

    The kinematic chain is intentionally config-driven because the exact Dofbot
    geometry and camera/tool mounting offsets must be measured on the target
    hardware before this can be trusted for grasp planning.
    """

    cfg = _kinematics_cfg(config)
    if not bool(cfg.get("enabled", False)):
        raise ValueError("robot.kinematics.enabled must be true before FK can be used")
    model = str(cfg.get("model") or "serial_chain")
    if model != "serial_chain":
        raise ValueError(f"unsupported robot.kinematics.model {model!r}; expected serial_chain")
    joint_items = _configured_joint_items(cfg.get("joints"))

    servo_map = normalize_servo_positions(servo_positions_deg)
    transform = np.eye(4, dtype=np.float64)
    used: list[int] = []
    for index, joint in enumerate(joint_items):
        joint_type = str(joint.get("type") or "revolute")
        origin = transform_from_xyz_rpy(
            _parse_vector3(joint.get("origin_xyz_m", [0.0, 0.0, 0.0]), label=f"joint {index} origin_xyz_m"),
            _parse_vector3(joint.get("origin_rpy_deg", [0.0, 0.0, 0.0]), label=f"joint {index} origin_rpy_deg"),
            degrees=True,
        )
        transform = transform @ origin
        if joint_type == "fixed":
            continue
        if joint_type != "revolute":
            raise ValueError(f"unsupported joint type {joint_type!r}; expected revolute or fixed")
        sid = _parse_servo_id(joint.get("servo_id"), label=f"joint {index} servo_id")
        if sid not in servo_map:
            raise ValueError(f"servo {sid} readback is required for FK")
        raw_deg = servo_map[sid]
        _check_limits(joint, sid, raw_deg)
        raw_zero = float(joint.get("raw_deg_at_zero", 0.0))
        sign = float(joint.get("sign", 1.0))
        offset = float(joint.get("joint_offset_deg", joint.get("offset_deg", 0.0)))
        if not np.isfinite([raw_zero, sign, offset]).all():
            raise ValueError(f"joint {index} has non-finite raw_deg_at_zero/sign/offset")
        joint_deg = sign * (raw_deg - raw_zero) + offset
        axis = _parse_axis(joint.get("axis", [0.0, 0.0, 1.0]), label=f"joint {index} axis")
        transform = transform @ rotation_about_axis(axis, np.deg2rad(joint_deg))
        used.append(sid)

    tool_xyz = _parse_vector3(cfg.get("tool_offset_xyz_m", [0.0, 0.0, 0.0]), label="tool_offset_xyz_m")
    tool_rpy = _parse_vector3(cfg.get("tool_offset_rpy_deg", [0.0, 0.0, 0.0]), label="tool_offset_rpy_deg")
    transform = transform @ transform_from_xyz_rpy(tool_xyz, tool_rpy, degrees=True)
    transform = validate_transform(transform, name="base_T_gripper")
    return ForwardKinematicsResult(
        base_T_gripper=np.ascontiguousarray(transform, dtype=np.float64),
        base_frame=str(cfg.get("base_frame")) if cfg.get("base_frame") else None,
        gripper_frame=str(cfg.get("tool_frame")) if cfg.get("tool_frame") else None,
        servo_positions_deg={sid: servo_map[sid] for sid in sorted(servo_map)},
        used_servo_ids=used,
    )


@dataclass(frozen=True)
class PositionIkResult:
    """Outcome of a numeric position-only IK solve."""

    ok: bool
    servo_positions_deg: dict[int, float]
    position_error_m: float
    iterations: int
    reason: str


def solve_position_ik(
    config: dict[str, Any],
    target_position_m: Any,
    *,
    seed_servo_positions_deg: dict[Any, Any] | None = None,
    tolerance_m: float = 0.005,
    max_iterations: int = 200,
    damping: float = 0.05,
    restarts: int = 6,
) -> PositionIkResult:
    """Solve for servo angles that place the tool origin at a base-frame point.

    Position-only damped least squares over the configured revolute joints, with
    joint limits enforced by clamping and several deterministic restarts. This
    answers "can the arm reach this point at all", which is what the reachability
    safety gate needs; it does not solve for orientation, so a solution here is a
    necessary but not sufficient condition for executing a full grasp pose.
    """

    target = np.asarray(target_position_m, dtype=np.float64).reshape(3)
    if not np.isfinite(target).all():
        raise ValueError("target_position_m must be finite")

    cfg = _kinematics_cfg(config)
    joint_items = _configured_joint_items(cfg.get("joints"))
    movable = [j for j in joint_items if str(j.get("type") or "revolute") == "revolute"]
    if not movable:
        raise ValueError("robot.kinematics.joints has no revolute joints for IK")

    sids: list[int] = []
    lo_raw: list[float] = []
    hi_raw: list[float] = []
    for index, joint in enumerate(movable):
        sid = _parse_servo_id(joint.get("servo_id"), label=f"joint {index} servo_id")
        limits = joint.get("limits_deg") or [0.0, 180.0]
        sids.append(sid)
        lo_raw.append(float(limits[0]))
        hi_raw.append(float(limits[1]))
    lo = np.asarray(lo_raw, dtype=np.float64)
    hi = np.asarray(hi_raw, dtype=np.float64)

    fixed: dict[int, float] = {}
    if seed_servo_positions_deg:
        for sid, value in normalize_servo_positions(seed_servo_positions_deg).items():
            fixed[sid] = value
    # Servos outside the FK chain (the gripper) still have to be supplied to FK.
    extra = {sid: fixed.get(sid, 90.0) for sid in range(1, 7) if sid not in sids}

    def tool_position(raw: np.ndarray) -> np.ndarray:
        servo_map = {sid: float(raw[i]) for i, sid in enumerate(sids)}
        servo_map.update(extra)
        return forward_kinematics_with_metadata(config, servo_map).base_T_gripper[:3, 3]

    seeds: list[np.ndarray] = []
    if fixed:
        seeds.append(np.array([fixed.get(sid, 90.0) for sid in sids], dtype=np.float64))
    mid = 0.5 * (lo + hi)
    seeds.append(mid)
    rng = np.random.default_rng(0)
    while len(seeds) < max(1, restarts):
        seeds.append(lo + rng.random(len(sids)) * (hi - lo))

    best = (np.inf, np.clip(seeds[0], lo, hi), 0)
    # A degree-scale probe: the Jacobian is in metres per degree, and probes much
    # smaller than this lose the signal in floating-point noise.
    step_deg = 0.05
    for seed in seeds:
        raw = np.clip(np.asarray(seed, dtype=np.float64), lo, hi)
        for iteration in range(1, max_iterations + 1):
            current = tool_position(raw)
            error = target - current
            err_norm = float(np.linalg.norm(error))
            if err_norm < best[0]:
                best = (err_norm, raw.copy(), iteration)
            if err_norm <= tolerance_m:
                return PositionIkResult(
                    True,
                    {sid: float(raw[i]) for i, sid in enumerate(sids)},
                    err_norm,
                    iteration,
                    "ok",
                )
            jac = np.zeros((3, len(sids)), dtype=np.float64)
            for i in range(len(sids)):
                probe = raw.copy()
                probe[i] = np.clip(probe[i] + step_deg, lo[i], hi[i])
                delta = probe[i] - raw[i]
                if abs(delta) < 1e-12:
                    probe[i] = np.clip(raw[i] - step_deg, lo[i], hi[i])
                    delta = probe[i] - raw[i]
                    if abs(delta) < 1e-12:
                        continue
                jac[:, i] = (tool_position(probe) - current) / delta
            # The Jacobian is in metres per degree (~1e-3), so a fixed absolute
            # damping term would dominate J^T J and collapse the solve into slow
            # gradient descent. Scale it to the current Jacobian instead.
            jtj = jac.T @ jac
            scale = float(np.trace(jtj)) / len(sids)
            if not np.isfinite(scale) or scale <= 0.0:
                break
            jtj = jtj + (damping**2) * scale * np.eye(len(sids))
            try:
                update = np.linalg.solve(jtj, jac.T @ error)
            except np.linalg.LinAlgError:
                break
            norm = float(np.linalg.norm(update))
            if not np.isfinite(norm) or norm < 1e-9:
                break
            # cap the per-iteration joint step so the linearization stays valid
            if norm > 10.0:
                update *= 10.0 / norm
            raw = np.clip(raw + update, lo, hi)

    return PositionIkResult(
        False,
        {sid: float(best[1][i]) for i, sid in enumerate(sids)},
        float(best[0]),
        best[2],
        "no_ik_solution_within_tolerance",
    )


def normalize_servo_positions(value: Any) -> dict[int, float]:
    """Normalize a raw servo readback mapping to `{servo_id: degrees}`."""

    if not isinstance(value, dict):
        raise ValueError("servo_positions_deg must be a mapping")
    result: dict[int, float] = {}
    for key, raw in value.items():
        sid = _parse_servo_id(key, label="servo id")
        if raw is None:
            continue
        angle = float(raw)
        if not np.isfinite(angle):
            raise ValueError(f"servo {sid} angle is not finite")
        result[sid] = angle
    if not result:
        raise ValueError("servo_positions_deg contains no finite servo values")
    return result


def transform_from_xyz_rpy(xyz_m: np.ndarray, rpy_deg: np.ndarray, *, degrees: bool = True) -> np.ndarray:
    """Build a homogeneous transform from translation and fixed-axis RPY."""

    xyz = np.asarray(xyz_m, dtype=np.float64).reshape(3)
    rpy = np.asarray(rpy_deg, dtype=np.float64).reshape(3)
    if degrees:
        rpy = np.deg2rad(rpy)
    roll, pitch, yaw = rpy
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)
    transform[:3, 3] = xyz
    return transform


def rotation_about_axis(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Return a homogeneous rotation about a unit axis."""

    axis_arr = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis_arr))
    if norm <= 1e-10:
        raise ValueError("rotation axis must be nonzero")
    axis_arr = axis_arr / norm
    x, y, z = axis_arr
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    v = 1.0 - c
    rot = np.array(
        [
            [x * x * v + c, x * y * v - z * s, x * z * v + y * s],
            [y * x * v + z * s, y * y * v + c, y * z * v - x * s],
            [z * x * v - y * s, z * y * v + x * s, z * z * v + c],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rot
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a rigid homogeneous transform."""

    matrix = validate_transform(np.asarray(transform, dtype=np.float64), name="transform")
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = matrix[:3, :3].T
    out[:3, 3] = -(matrix[:3, :3].T @ matrix[:3, 3])
    return out


def validate_transform(matrix: np.ndarray, *, name: str = "transform") -> np.ndarray:
    """Validate a 4x4 rigid transform matrix."""

    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(arr[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-6):
        raise ValueError(f"{name} last row must be [0, 0, 0, 1]")
    rot = arr[:3, :3]
    if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation is not orthonormal")
    det = float(np.linalg.det(rot))
    if not 0.999 <= det <= 1.001:
        raise ValueError(f"{name} rotation determinant must be close to 1, got {det}")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _configured_joint_items(raw_joints: Any) -> list[dict[str, Any]]:
    if isinstance(raw_joints, dict):
        iterable = [raw_joints[key] for key in sorted(raw_joints, key=lambda item: int(item) if str(item).isdigit() else str(item))]
    else:
        iterable = raw_joints or []
    if not isinstance(iterable, list) or not iterable:
        raise ValueError("robot.kinematics.joints must be a non-empty list or mapping")
    items: list[dict[str, Any]] = []
    for index, joint in enumerate(iterable):
        if not isinstance(joint, dict):
            raise ValueError(f"robot.kinematics.joints[{index}] must be a mapping")
        if not bool(joint.get("enabled", True)):
            continue
        items.append(joint)
    if not items:
        raise ValueError("robot.kinematics.joints has no enabled joints")
    return items


def _kinematics_cfg(config: dict[str, Any]) -> dict[str, Any]:
    cfg = get_nested(config, ("robot", "kinematics"), {}) or {}
    if not isinstance(cfg, dict):
        raise ValueError("robot.kinematics must be a mapping")
    return cfg


def _parse_servo_id(value: Any, *, label: str) -> int:
    try:
        sid = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label}: {value!r}") from exc
    if sid < 1:
        raise ValueError(f"servo id must be positive, got {sid}")
    return sid


def _parse_vector3(value: Any, *, label: str) -> np.ndarray:
    if value in (None, ""):
        raise ValueError(f"{label} must be configured")
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must contain three values")
    if any(item is None for item in value):
        raise ValueError(f"{label} contains null values")
    arr = np.asarray(value, dtype=np.float64).reshape(3)
    if not np.isfinite(arr).all():
        raise ValueError(f"{label} contains non-finite values")
    return arr


def _parse_axis(value: Any, *, label: str) -> np.ndarray:
    axis = _parse_vector3(value, label=label)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-10:
        raise ValueError(f"{label} must be nonzero")
    return axis / norm


def _check_limits(joint: dict[str, Any], sid: int, raw_deg: float) -> None:
    raw_limits = joint.get("limits_deg")
    if raw_limits is None:
        return
    if not isinstance(raw_limits, (list, tuple)) or len(raw_limits) != 2:
        raise ValueError(f"joint for servo {sid} limits_deg must be [min, max]")
    low, high = float(raw_limits[0]), float(raw_limits[1])
    if not np.isfinite([low, high]).all() or low > high:
        raise ValueError(f"joint for servo {sid} has invalid limits_deg {raw_limits!r}")
    if raw_deg < low or raw_deg > high:
        raise ValueError(f"servo {sid} readback {raw_deg:.3f} outside FK limits [{low:.3f}, {high:.3f}]")


def _rot_x(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
