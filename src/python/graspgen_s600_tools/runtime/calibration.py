"""Calibration transform helpers for client-side real-grasp planning."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .hbm_graspgen_runtime import get_nested, resolve_config_path


@dataclass(frozen=True)
class CalibrationTransform:
    """A validated homogeneous transform with frame metadata."""

    matrix: np.ndarray
    source_frame: str | None
    target_frame: str | None
    path: Path
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class EyeInHandValidation:
    """Read-only validation of a derived eye-in-hand camera transform."""

    enabled: bool
    ok: bool
    reason: str
    gripper_frame: str | None
    camera_frame: str | None
    robot_base_frame: str | None
    used_servo_ids: list[int]
    base_T_gripper: np.ndarray | None
    gripper_T_camera_path: Path | None


@dataclass(frozen=True)
class ObservationPoseValidation:
    """Read-only validation of a fixed eye-in-hand observation pose."""

    enabled: bool
    configured: bool
    ok: bool
    name: str | None
    expected_servo_positions_deg: dict[int, float]
    actual_servo_positions_deg: dict[int, float | None]
    tolerance_deg: dict[int, float]
    deltas_deg: dict[int, float | None]
    failed_servos: list[int]
    skipped_servos: list[int]
    reason: str


@dataclass(frozen=True)
class PointCorrespondenceCalibration:
    """Rigid transform solved from camera/base 3D point correspondences."""

    robot_base_T_camera: np.ndarray
    residuals_m: np.ndarray
    mean_residual_m: float
    max_residual_m: float
    rms_residual_m: float
    point_count: int


def solve_robot_base_T_camera_from_points(
    camera_points: np.ndarray,
    robot_base_points: np.ndarray,
) -> PointCorrespondenceCalibration:
    """Solve `robot_base_point ~= R @ camera_point + t` from 3D correspondences.

    Uses the Kabsch rigid alignment without scale. At least three non-collinear
    points are required; more points are strongly preferred for deployment.
    """

    camera = _validate_correspondence_points(camera_points, name="camera_points")
    robot = _validate_correspondence_points(robot_base_points, name="robot_base_points")
    if camera.shape != robot.shape:
        raise ValueError(f"camera_points shape {camera.shape} != robot_base_points shape {robot.shape}")
    if camera.shape[0] < 3:
        raise ValueError("at least 3 point correspondences are required")

    camera_centroid = camera.mean(axis=0)
    robot_centroid = robot.mean(axis=0)
    camera_centered = camera - camera_centroid[None, :]
    robot_centered = robot - robot_centroid[None, :]
    if np.linalg.matrix_rank(camera_centered) < 2 or np.linalg.matrix_rank(robot_centered) < 2:
        raise ValueError("point correspondences are degenerate; use non-collinear 3D points")

    covariance = camera_centered.T @ robot_centered
    u, _, vh = np.linalg.svd(covariance)
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0:
        vh[-1, :] *= -1.0
        rotation = vh.T @ u.T
    translation = robot_centroid - rotation @ camera_centroid

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    validate_transform(transform, name="robot_base_T_camera")

    predicted = (rotation @ camera.T).T + translation[None, :]
    residuals = np.linalg.norm(predicted - robot, axis=1)
    return PointCorrespondenceCalibration(
        robot_base_T_camera=np.ascontiguousarray(transform, dtype=np.float64),
        residuals_m=np.ascontiguousarray(residuals, dtype=np.float64),
        mean_residual_m=float(residuals.mean()),
        max_residual_m=float(residuals.max()),
        rms_residual_m=float(np.sqrt(np.mean(residuals**2))),
        point_count=int(camera.shape[0]),
    )


def _validate_correspondence_points(points: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(arr, dtype=np.float64)


def load_robot_base_T_camera(config: dict[str, Any]) -> CalibrationTransform | None:
    """Load the camera-to-robot-base transform if one is configured.

    Preferred config key:
      calibration.robot_base_T_camera_path

    Legacy keys are accepted only when `calibration.transform_direction` is set
    to `robot_base_T_camera`, because `camera_T_robot_path` is otherwise
    ambiguous.  The loaded file should contain either `robot_base_T_camera` or
    `matrix`, plus optional `source_frame`/`camera_frame` and
    `target_frame`/`robot_base_frame`.
    """

    calib_cfg = get_nested(config, ("calibration",), {}) or {}
    raw_path = calib_cfg.get("robot_base_T_camera_path")
    if raw_path in (None, ""):
        legacy_path = calib_cfg.get("world_T_camera_path") or calib_cfg.get("camera_T_robot_path")
        if legacy_path not in (None, ""):
            direction = str(calib_cfg.get("transform_direction") or "")
            if direction != "robot_base_T_camera":
                raise ValueError(
                    "legacy calibration path keys require calibration.transform_direction="
                    "'robot_base_T_camera' before they can be used for execution planning"
                )
            raw_path = legacy_path
    path = resolve_config_path(raw_path)
    if path is None:
        return None
    data = _load_transform_file(path)
    matrix_value = data.get("robot_base_T_camera", data.get("matrix"))
    if matrix_value is None:
        raise ValueError(f"calibration file {path} must contain robot_base_T_camera or matrix")
    matrix = validate_transform(np.asarray(matrix_value, dtype=np.float64), name=str(path))
    source_frame = data.get("source_frame") or data.get("camera_frame") or data.get("frame_id")
    target_frame = data.get("target_frame") or data.get("robot_base_frame") or data.get("child_frame_id")
    expected_camera = calib_cfg.get("camera_frame")
    expected_robot = calib_cfg.get("robot_base_frame")
    if expected_camera and source_frame and str(expected_camera) != str(source_frame):
        raise ValueError(
            f"calibration camera frame mismatch: config={expected_camera!r} file={source_frame!r}"
        )
    if expected_robot and target_frame and str(expected_robot) != str(target_frame):
        raise ValueError(
            f"calibration robot frame mismatch: config={expected_robot!r} file={target_frame!r}"
        )
    return CalibrationTransform(
        matrix=np.ascontiguousarray(matrix, dtype=np.float64),
        source_frame=str(source_frame) if source_frame else None,
        target_frame=str(target_frame) if target_frame else None,
        path=path,
        metadata=data,
    )


def load_gripper_T_camera(config: dict[str, Any]) -> CalibrationTransform | None:
    """Load the eye-in-hand gripper-to-camera transform if configured."""

    calib_cfg = get_nested(config, ("calibration",), {}) or {}
    raw_path = calib_cfg.get("gripper_T_camera_path")
    path = resolve_config_path(raw_path)
    if path is None:
        return None
    data = _load_transform_file(path)
    matrix_value = data.get("gripper_T_camera", data.get("matrix"))
    if matrix_value is None:
        raise ValueError(f"hand-eye calibration file {path} must contain gripper_T_camera or matrix")
    matrix = validate_transform(np.asarray(matrix_value, dtype=np.float64), name=str(path))
    source_frame = data.get("source_frame") or data.get("camera_frame") or data.get("frame_id")
    target_frame = data.get("target_frame") or data.get("gripper_frame") or data.get("tool_frame")
    expected_camera = calib_cfg.get("camera_frame")
    expected_gripper = calib_cfg.get("gripper_frame") or get_nested(config, ("robot", "kinematics", "tool_frame"), None)
    if expected_camera and source_frame and str(expected_camera) != str(source_frame):
        raise ValueError(
            f"hand-eye camera frame mismatch: config={expected_camera!r} file={source_frame!r}"
        )
    if expected_gripper and target_frame and str(expected_gripper) != str(target_frame):
        raise ValueError(
            f"hand-eye gripper frame mismatch: config={expected_gripper!r} file={target_frame!r}"
        )
    method = str(data.get("method") or "")
    supported_methods = {
        "eye_in_hand_colored_target_ax_xb",
        "eye_in_hand_markerless_point_cloud_ax_xb",
        "mechanical_measurement",
    }
    if method and method not in supported_methods:
        raise ValueError(f"unsupported hand-eye method {method!r}")
    return CalibrationTransform(
        matrix=np.ascontiguousarray(matrix, dtype=np.float64),
        source_frame=str(source_frame) if source_frame else None,
        target_frame=str(target_frame) if target_frame else None,
        path=path,
        metadata=data,
    )


def derive_robot_base_T_camera_from_hand_eye(
    config: dict[str, Any],
    robot_health: dict[str, Any],
) -> tuple[CalibrationTransform | None, EyeInHandValidation]:
    """Derive `robot_base_T_camera` from FK and `gripper_T_camera`."""

    calib_cfg = get_nested(config, ("calibration",), {}) or {}
    enabled = str(calib_cfg.get("mode") or "") == "eye_in_hand"
    camera_frame = str(calib_cfg.get("camera_frame")) if calib_cfg.get("camera_frame") else None
    robot_base_frame = str(calib_cfg.get("robot_base_frame")) if calib_cfg.get("robot_base_frame") else None
    gripper_frame = str(calib_cfg.get("gripper_frame") or get_nested(config, ("robot", "kinematics", "tool_frame"), "")) or None
    if not enabled:
        return None, EyeInHandValidation(False, False, "disabled", gripper_frame, camera_frame, robot_base_frame, [], None, None)
    try:
        hand_eye = load_gripper_T_camera(config)
        if hand_eye is None:
            return None, EyeInHandValidation(True, False, "gripper_T_camera_missing", gripper_frame, camera_frame, robot_base_frame, [], None, None)
        from .kinematics import forward_kinematics_with_metadata

        fk = forward_kinematics_with_metadata(config, robot_health.get("servo_positions_deg") or {})
        matrix = validate_transform(fk.base_T_gripper @ hand_eye.matrix, name="robot_base_T_camera_from_hand_eye")
        transform = CalibrationTransform(
            matrix=np.ascontiguousarray(matrix, dtype=np.float64),
            source_frame=camera_frame or hand_eye.source_frame,
            target_frame=robot_base_frame or fk.base_frame,
            path=hand_eye.path,
            metadata={
                "mode": "eye_in_hand",
                "gripper_T_camera_path": str(hand_eye.path),
                "gripper_frame": gripper_frame or fk.gripper_frame,
                "base_T_gripper": fk.base_T_gripper.astype(float).tolist(),
                "used_servo_ids": fk.used_servo_ids,
                "hand_eye": hand_eye.metadata or {},
            },
        )
        return transform, EyeInHandValidation(
            True,
            True,
            "ok",
            gripper_frame or fk.gripper_frame,
            camera_frame or hand_eye.source_frame,
            robot_base_frame or fk.base_frame,
            fk.used_servo_ids,
            fk.base_T_gripper,
            hand_eye.path,
        )
    except Exception as exc:
        return None, EyeInHandValidation(
            True,
            False,
            str(exc),
            gripper_frame,
            camera_frame,
            robot_base_frame,
            [],
            None,
            resolve_config_path(calib_cfg.get("gripper_T_camera_path")),
        )


def _transform_path_configured(config: dict[str, Any]) -> bool:
    calib_cfg = get_nested(config, ("calibration",), {}) or {}
    for key in ("robot_base_T_camera_path", "world_T_camera_path", "camera_T_robot_path"):
        if calib_cfg.get(key) not in (None, ""):
            return True
    return False


def validate_observation_pose_from_health(
    config: dict[str, Any],
    robot_health: dict[str, Any],
) -> ObservationPoseValidation:
    """Compare live servo readback against configured fixed observation pose."""

    pose_cfg = get_nested(config, ("calibration", "observation_pose"), {}) or {}
    enabled = bool(pose_cfg.get("enabled", False))
    name = pose_cfg.get("name")
    if not enabled:
        return ObservationPoseValidation(
            enabled=False,
            configured=False,
            ok=False,
            name=str(name) if name else None,
            expected_servo_positions_deg={},
            actual_servo_positions_deg=_normalize_servo_map(robot_health.get("servo_positions_deg"), allow_none=True),
            tolerance_deg={},
            deltas_deg={},
            failed_servos=[],
            skipped_servos=list(range(1, 7)),
            reason="disabled",
        )

    expected, skipped = _normalize_expected_servo_map(pose_cfg.get("servo_positions_deg") or {})
    actual = _normalize_servo_map(robot_health.get("servo_positions_deg"), allow_none=True)
    tolerance = _load_tolerance_map(pose_cfg.get("tolerance_deg"), expected.keys())
    if robot_health.get("status") == "error":
        return ObservationPoseValidation(
            enabled=True,
            configured=bool(expected),
            ok=False,
            name=str(name) if name else None,
            expected_servo_positions_deg=expected,
            actual_servo_positions_deg=actual,
            tolerance_deg=tolerance,
            deltas_deg={sid: None for sid in expected},
            failed_servos=sorted(expected),
            skipped_servos=skipped,
            reason="robot_health_error",
        )
    if not expected:
        return ObservationPoseValidation(
            enabled=True,
            configured=False,
            ok=False,
            name=str(name) if name else None,
            expected_servo_positions_deg={},
            actual_servo_positions_deg=actual,
            tolerance_deg={},
            deltas_deg={},
            failed_servos=[],
            skipped_servos=skipped or list(range(1, 7)),
            reason="no_expected_servos_configured",
        )
    if not actual:
        return ObservationPoseValidation(
            enabled=True,
            configured=True,
            ok=False,
            name=str(name) if name else None,
            expected_servo_positions_deg=expected,
            actual_servo_positions_deg={},
            tolerance_deg=tolerance,
            deltas_deg={sid: None for sid in expected},
            failed_servos=sorted(expected),
            skipped_servos=skipped,
            reason="servo_readback_missing",
        )

    deltas: dict[int, float | None] = {}
    failed: list[int] = []
    missing = False
    for sid, target in expected.items():
        current = actual.get(sid)
        if current is None:
            deltas[sid] = None
            failed.append(sid)
            missing = True
            continue
        delta = abs(float(current) - float(target))
        deltas[sid] = delta
        if delta > tolerance[sid]:
            failed.append(sid)
    if failed:
        return ObservationPoseValidation(
            enabled=True,
            configured=True,
            ok=False,
            name=str(name) if name else None,
            expected_servo_positions_deg=expected,
            actual_servo_positions_deg=actual,
            tolerance_deg=tolerance,
            deltas_deg=deltas,
            failed_servos=failed,
            skipped_servos=skipped,
            reason="missing_servo_readback" if missing else "outside_tolerance",
        )
    return ObservationPoseValidation(
        enabled=True,
        configured=True,
        ok=True,
        name=str(name) if name else None,
        expected_servo_positions_deg=expected,
        actual_servo_positions_deg=actual,
        tolerance_deg=tolerance,
        deltas_deg=deltas,
        failed_servos=[],
        skipped_servos=skipped,
        reason="ok",
    )


def observation_pose_metadata(validation: ObservationPoseValidation) -> dict[str, Any]:
    """Return JSON-friendly observation pose metadata."""

    return {
        "enabled": validation.enabled,
        "configured": validation.configured,
        "ok": validation.ok,
        "name": validation.name,
        "expected_servo_positions_deg": _string_keyed(validation.expected_servo_positions_deg),
        "actual_servo_positions_deg": _string_keyed(validation.actual_servo_positions_deg),
        "tolerance_deg": _string_keyed(validation.tolerance_deg),
        "deltas_deg": _string_keyed(validation.deltas_deg),
        "failed_servos": validation.failed_servos,
        "skipped_servos": validation.skipped_servos,
        "reason": validation.reason,
    }


def load_robot_base_T_camera_for_observation(
    config: dict[str, Any],
    robot_health: dict[str, Any],
) -> tuple[CalibrationTransform | None, ObservationPoseValidation]:
    """Load the fixed observation transform only when servo pose matches."""

    validation = validate_observation_pose_from_health(config, robot_health)
    calibration = load_robot_base_T_camera(config)
    if calibration is None:
        return None, validation
    pose_cfg = get_nested(config, ("calibration", "observation_pose"), {}) or {}
    require_for_transform = bool(pose_cfg.get("require_for_transform", bool(pose_cfg.get("enabled", False))))
    if require_for_transform and not validation.ok:
        return None, validation
    return calibration, validation


def load_robot_base_T_camera_for_planning(
    config: dict[str, Any],
    robot_health: dict[str, Any],
) -> tuple[CalibrationTransform | None, dict[str, Any]]:
    """Resolve the trusted planning transform in fixed or eye-in-hand mode."""

    mode = str(get_nested(config, ("calibration", "mode"), "fixed") or "fixed")
    if mode == "eye_in_hand":
        calibration, validation = derive_robot_base_T_camera_from_hand_eye(config, robot_health)
        return calibration, {"mode": "eye_in_hand", "eye_in_hand": eye_in_hand_metadata(validation)}
    calibration, observation = load_robot_base_T_camera_for_observation(config, robot_health)
    return calibration, {"mode": "fixed", "observation_pose": observation_pose_metadata(observation)}


def eye_in_hand_metadata(validation: EyeInHandValidation) -> dict[str, Any]:
    """Return JSON-friendly eye-in-hand validation metadata."""

    return {
        "enabled": validation.enabled,
        "ok": validation.ok,
        "reason": validation.reason,
        "gripper_frame": validation.gripper_frame,
        "camera_frame": validation.camera_frame,
        "robot_base_frame": validation.robot_base_frame,
        "used_servo_ids": validation.used_servo_ids,
        "gripper_T_camera_path": str(validation.gripper_T_camera_path) if validation.gripper_T_camera_path else None,
        "base_T_gripper_translation_m": (
            validation.base_T_gripper[:3, 3].astype(float).tolist()
            if validation.base_T_gripper is not None
            else None
        ),
    }


def _normalize_expected_servo_map(value: Any) -> tuple[dict[int, float], list[int]]:
    if not isinstance(value, dict):
        raise ValueError("calibration.observation_pose.servo_positions_deg must be a mapping")
    expected: dict[int, float] = {}
    skipped: list[int] = []
    for sid in range(1, 7):
        raw = value.get(sid, value.get(str(sid)))
        if raw is None:
            skipped.append(sid)
            continue
        expected[sid] = float(raw)
    for key in value:
        sid = _parse_servo_id(key)
        if sid not in range(1, 7):
            raise ValueError(f"unsupported servo id {key!r}; expected 1..6")
    return expected, skipped


def _normalize_servo_map(value: Any, *, allow_none: bool) -> dict[int, float | None]:
    if value in (None, ""):
        return {}
    if not isinstance(value, dict):
        raise ValueError("servo_positions_deg must be a mapping")
    result: dict[int, float | None] = {}
    for key, raw in value.items():
        sid = _parse_servo_id(key)
        if sid not in range(1, 7):
            continue
        if raw is None:
            if allow_none:
                result[sid] = None
            continue
        result[sid] = float(raw)
    return result


def _load_tolerance_map(value: Any, servo_ids: Any) -> dict[int, float]:
    if isinstance(value, dict):
        default = float(value.get("default", 2.0))
        result = {int(sid): default for sid in servo_ids}
        for key, raw in value.items():
            if key == "default":
                continue
            sid = _parse_servo_id(key)
            if sid in result:
                result[sid] = float(raw)
        return result
    default = 2.0 if value is None else float(value)
    return {int(sid): default for sid in servo_ids}


def _parse_servo_id(value: Any) -> int:
    try:
        sid = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid servo id {value!r}") from exc
    return sid


def _string_keyed(value: dict[int, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in sorted(value.items())}


def _load_transform_file(path: Path) -> dict[str, Any]:
    """Load a small transform file without requiring PyYAML on the Orange Pi."""

    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml  # type: ignore
        except ImportError:
            data = _load_transform_yaml_subset(text)
        else:
            data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"calibration file root must be a mapping: {path}")
    return data


def _load_transform_yaml_subset(text: str) -> dict[str, Any]:
    """Parse the small calibration YAML shape used by deployment configs."""

    data: dict[str, Any] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i].split("#", 1)[0].rstrip()
        i += 1
        if not raw.strip():
            continue
        if raw.startswith(" "):
            continue
        if ":" not in raw:
            raise ValueError(f"unsupported calibration YAML line: {lines[i - 1]!r}")
        key, value = raw.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in {"matrix", "robot_base_T_camera", "gripper_T_camera"} and value == "":
            rows: list[list[float]] = []
            while i < len(lines):
                row_raw = lines[i].split("#", 1)[0].strip()
                if not row_raw:
                    i += 1
                    continue
                if not row_raw.startswith("-"):
                    break
                row_text = row_raw[1:].strip()
                rows.append([float(v) for v in ast.literal_eval(row_text)])
                i += 1
            data[key] = rows
        elif value == "":
            data[key] = None
        else:
            data[key] = _parse_scalar(value)
    return data


def _parse_scalar(value: str) -> Any:
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def validate_transform(matrix: np.ndarray, *, name: str = "transform") -> np.ndarray:
    """Validate a 4x4 rigid transform matrix."""

    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-5):
        raise ValueError(f"{name} last row must be [0, 0, 0, 1]")
    rot = matrix[:3, :3]
    if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-3):
        raise ValueError(f"{name} rotation is not orthonormal")
    det = float(np.linalg.det(rot))
    if not 0.99 <= det <= 1.01:
        raise ValueError(f"{name} rotation determinant must be close to 1, got {det}")
    return matrix


def transform_grasps(grasps_camera: np.ndarray, robot_base_T_camera: np.ndarray) -> np.ndarray:
    """Transform `(M,4,4)` camera-frame grasps into robot base frame."""

    grasps = np.asarray(grasps_camera, dtype=np.float64)
    if grasps.ndim != 3 or grasps.shape[1:] != (4, 4):
        raise ValueError(f"grasps must be (M,4,4), got {grasps.shape}")
    transform = validate_transform(np.asarray(robot_base_T_camera, dtype=np.float64), name="robot_base_T_camera")
    return np.ascontiguousarray(transform[None, :, :] @ grasps, dtype=np.float32)


def calibration_metadata(calibration: CalibrationTransform | None) -> dict[str, Any]:
    """Return JSON-friendly calibration metadata for logs."""

    if calibration is None:
        return {"calibration_loaded": False}
    return {
        "calibration_loaded": True,
        "calibration_path": str(calibration.path),
        "source_frame": calibration.source_frame,
        "target_frame": calibration.target_frame,
        "robot_base_T_camera_translation_m": calibration.matrix[:3, 3].astype(float).tolist(),
    }
