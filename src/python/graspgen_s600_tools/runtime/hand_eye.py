"""Eye-in-hand hand-eye calibration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .hbm_graspgen_runtime import get_nested
from .kinematics import invert_transform, validate_transform


@dataclass(frozen=True)
class HandEyeSample:
    """One valid hand-eye sample pair."""

    sample_id: str
    base_T_gripper: np.ndarray
    camera_T_target: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class HandEyePairResidual:
    """Residual for one relative-motion AX=XB pair."""

    sample_a: str
    sample_b: str
    translation_residual_m: float
    rotation_residual_deg: float
    motion_rotation_deg: float


@dataclass(frozen=True)
class HandEyeCalibrationResult:
    """Solved `gripper_T_camera` and validation diagnostics."""

    gripper_T_camera: np.ndarray
    sample_count: int
    pair_count: int
    used_pairs: list[tuple[int, int]]
    pair_residuals: list[HandEyePairResidual]
    mean_axxb_translation_residual_m: float
    max_axxb_translation_residual_m: float
    mean_axxb_rotation_residual_deg: float
    max_axxb_rotation_residual_deg: float
    base_target_translation_spread_m: float
    base_target_rotation_spread_deg: float
    rotation_system_singular_values: list[float]
    translation_system_rank: int
    ok: bool
    reasons: list[str]


def solve_hand_eye(
    samples: list[HandEyeSample],
    *,
    config: dict[str, Any] | None = None,
) -> HandEyeCalibrationResult:
    """Solve eye-in-hand `gripper_T_camera` from valid samples.

    Each sample must provide `base_T_gripper` from FK and `camera_T_target`
    from target detection.  The target is assumed fixed in the robot base frame
    for the duration of the sample set.
    """

    cfg = _hand_eye_cfg(config or {})
    min_samples = int(cfg.get("min_samples", 3))
    min_pairs = int(cfg.get("min_pairs", max(3, min_samples - 1)))
    min_pair_rotation_deg = float(cfg.get("min_pair_rotation_deg", 3.0))
    max_pair_translation = float(cfg.get("max_pair_translation_residual_m", 0.030))
    max_pair_rotation = float(cfg.get("max_pair_rotation_residual_deg", 5.0))
    max_target_spread = float(cfg.get("max_base_target_translation_spread_m", 0.030))
    max_target_rotation_spread = float(cfg.get("max_base_target_rotation_spread_deg", 5.0))

    if len(samples) < min_samples:
        raise ValueError(f"at least {min_samples} valid hand-eye samples are required, got {len(samples)}")
    base_T_gripper = [validate_transform(sample.base_T_gripper, name=f"{sample.sample_id}.base_T_gripper") for sample in samples]
    camera_T_target = [validate_transform(sample.camera_T_target, name=f"{sample.sample_id}.camera_T_target") for sample in samples]

    pairs: list[tuple[int, int, np.ndarray, np.ndarray, float]] = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            a_ij = invert_transform(base_T_gripper[j]) @ base_T_gripper[i]
            b_ij = camera_T_target[j] @ invert_transform(camera_T_target[i])
            a_ij = validate_transform(a_ij, name="A_ij")
            b_ij = validate_transform(b_ij, name="B_ij")
            motion_angle = max(_rotation_angle_deg(a_ij[:3, :3]), _rotation_angle_deg(b_ij[:3, :3]))
            if motion_angle < min_pair_rotation_deg:
                continue
            pairs.append((i, j, a_ij, b_ij, motion_angle))
    if len(pairs) < min_pairs:
        raise ValueError(
            f"at least {min_pairs} diverse relative-motion pairs are required, got {len(pairs)}; "
            f"increase pose rotation diversity or lower min_pair_rotation_deg"
        )

    rotation, singular_values = _solve_rotation([item[2] for item in pairs], [item[3] for item in pairs])
    translation, translation_rank = _solve_translation(rotation, [item[2] for item in pairs], [item[3] for item in pairs])
    gripper_T_camera = np.eye(4, dtype=np.float64)
    gripper_T_camera[:3, :3] = rotation
    gripper_T_camera[:3, 3] = translation
    gripper_T_camera = validate_transform(gripper_T_camera, name="gripper_T_camera")

    pair_residuals = _pair_residuals(samples, pairs, gripper_T_camera)
    base_target_trans_spread, base_target_rot_spread = _base_target_spread(base_T_gripper, camera_T_target, gripper_T_camera)
    mean_translation = float(np.mean([row.translation_residual_m for row in pair_residuals]))
    max_translation = float(np.max([row.translation_residual_m for row in pair_residuals]))
    mean_rotation = float(np.mean([row.rotation_residual_deg for row in pair_residuals]))
    max_rotation = float(np.max([row.rotation_residual_deg for row in pair_residuals]))

    reasons: list[str] = []
    if translation_rank < 3:
        reasons.append(f"translation_system_rank_low:{translation_rank}<3")
    if max_translation > max_pair_translation:
        reasons.append(f"pair_translation_residual_high:{max_translation:.6g}>{max_pair_translation:.6g}")
    if max_rotation > max_pair_rotation:
        reasons.append(f"pair_rotation_residual_high:{max_rotation:.6g}>{max_pair_rotation:.6g}")
    if base_target_trans_spread > max_target_spread:
        reasons.append(f"base_target_translation_spread_high:{base_target_trans_spread:.6g}>{max_target_spread:.6g}")
    if base_target_rot_spread > max_target_rotation_spread:
        reasons.append(f"base_target_rotation_spread_high:{base_target_rot_spread:.6g}>{max_target_rotation_spread:.6g}")
    return HandEyeCalibrationResult(
        gripper_T_camera=np.ascontiguousarray(gripper_T_camera, dtype=np.float64),
        sample_count=len(samples),
        pair_count=len(pairs),
        used_pairs=[(item[0], item[1]) for item in pairs],
        pair_residuals=pair_residuals,
        mean_axxb_translation_residual_m=mean_translation,
        max_axxb_translation_residual_m=max_translation,
        mean_axxb_rotation_residual_deg=mean_rotation,
        max_axxb_rotation_residual_deg=max_rotation,
        base_target_translation_spread_m=base_target_trans_spread,
        base_target_rotation_spread_deg=base_target_rot_spread,
        rotation_system_singular_values=[float(v) for v in singular_values],
        translation_system_rank=int(translation_rank),
        ok=not reasons,
        reasons=reasons or ["ok"],
    )


@dataclass(frozen=True)
class HandEyeMotionPair:
    """One relative-motion constraint A X = X B for markerless solving."""

    sample_a: str
    sample_b: str
    a_motion: np.ndarray
    b_motion: np.ndarray
    motion_rotation_deg: float
    weight: float = 1.0


def solve_hand_eye_from_motions(
    motion_pairs: list[HandEyeMotionPair],
    *,
    config: dict[str, Any] | None = None,
) -> HandEyeCalibrationResult:
    """Solve `gripper_T_camera` directly from relative-motion pairs.

    Unlike :func:`solve_hand_eye`, this does not assume a fixed target seen by the
    camera.  Each pair supplies ``A`` from robot FK relative motion and ``B`` from
    camera self-motion (for example ICP ``camera_i_T_camera_j``).  This is the
    markerless path: no `camera_T_target` is required, so no target board or marker
    coordinates are needed.  The AX=XB rotation/translation solver and residual
    metrics are shared with the marker-based path.
    """

    cfg = _hand_eye_cfg(config or {})
    min_pairs = int(cfg.get("min_pairs", 3))
    min_pair_rotation_deg = float(cfg.get("min_pair_rotation_deg", 3.0))
    max_pair_translation = float(cfg.get("max_pair_translation_residual_m", 0.030))
    max_pair_rotation = float(cfg.get("max_pair_rotation_residual_deg", 5.0))

    usable: list[HandEyeMotionPair] = []
    for pair in motion_pairs:
        a_motion = validate_transform(pair.a_motion, name=f"A[{pair.sample_a},{pair.sample_b}]")
        b_motion = validate_transform(pair.b_motion, name=f"B[{pair.sample_a},{pair.sample_b}]")
        motion_angle = max(_rotation_angle_deg(a_motion[:3, :3]), _rotation_angle_deg(b_motion[:3, :3]))
        if motion_angle < min_pair_rotation_deg:
            continue
        usable.append(
            HandEyeMotionPair(
                sample_a=pair.sample_a,
                sample_b=pair.sample_b,
                a_motion=a_motion,
                b_motion=b_motion,
                motion_rotation_deg=float(motion_angle),
                weight=float(pair.weight),
            )
        )
    if len(usable) < min_pairs:
        raise ValueError(
            f"at least {min_pairs} diverse relative-motion pairs are required, got {len(usable)}; "
            f"collect more captures with larger rotation between them or lower min_pair_rotation_deg"
        )

    a_motions = [pair.a_motion for pair in usable]
    b_motions = [pair.b_motion for pair in usable]
    rotation, singular_values = _solve_rotation(a_motions, b_motions)
    translation, translation_rank = _solve_translation(rotation, a_motions, b_motions)
    gripper_T_camera = np.eye(4, dtype=np.float64)
    gripper_T_camera[:3, :3] = rotation
    gripper_T_camera[:3, 3] = translation
    gripper_T_camera = validate_transform(gripper_T_camera, name="gripper_T_camera")

    residuals: list[HandEyePairResidual] = []
    for pair in usable:
        lhs = pair.a_motion @ gripper_T_camera
        rhs = gripper_T_camera @ pair.b_motion
        delta = invert_transform(lhs) @ rhs
        delta = validate_transform(delta, name="AXXB residual")
        residuals.append(
            HandEyePairResidual(
                sample_a=pair.sample_a,
                sample_b=pair.sample_b,
                translation_residual_m=float(np.linalg.norm(delta[:3, 3])),
                rotation_residual_deg=_rotation_angle_deg(delta[:3, :3]),
                motion_rotation_deg=pair.motion_rotation_deg,
            )
        )
    mean_translation = float(np.mean([row.translation_residual_m for row in residuals]))
    max_translation = float(np.max([row.translation_residual_m for row in residuals]))
    mean_rotation = float(np.mean([row.rotation_residual_deg for row in residuals]))
    max_rotation = float(np.max([row.rotation_residual_deg for row in residuals]))

    reasons: list[str] = []
    if translation_rank < 3:
        reasons.append(f"translation_system_rank_low:{translation_rank}<3")
    if max_translation > max_pair_translation:
        reasons.append(f"pair_translation_residual_high:{max_translation:.6g}>{max_pair_translation:.6g}")
    if max_rotation > max_pair_rotation:
        reasons.append(f"pair_rotation_residual_high:{max_rotation:.6g}>{max_pair_rotation:.6g}")
    return HandEyeCalibrationResult(
        gripper_T_camera=np.ascontiguousarray(gripper_T_camera, dtype=np.float64),
        sample_count=len({pair.sample_a for pair in usable} | {pair.sample_b for pair in usable}),
        pair_count=len(usable),
        used_pairs=[],
        pair_residuals=residuals,
        mean_axxb_translation_residual_m=mean_translation,
        max_axxb_translation_residual_m=max_translation,
        mean_axxb_rotation_residual_deg=mean_rotation,
        max_axxb_rotation_residual_deg=max_rotation,
        base_target_translation_spread_m=float("nan"),
        base_target_rotation_spread_deg=float("nan"),
        rotation_system_singular_values=[float(v) for v in singular_values],
        translation_system_rank=int(translation_rank),
        ok=not reasons,
        reasons=reasons or ["ok"],
    )


def hand_eye_result_summary(result: HandEyeCalibrationResult) -> dict[str, Any]:
    """Return a JSON-friendly hand-eye result summary."""

    return {
        "ok": result.ok,
        "reasons": result.reasons,
        "sample_count": result.sample_count,
        "pair_count": result.pair_count,
        "gripper_T_camera": result.gripper_T_camera.astype(float).tolist(),
        "mean_axxb_translation_residual_m": result.mean_axxb_translation_residual_m,
        "max_axxb_translation_residual_m": result.max_axxb_translation_residual_m,
        "mean_axxb_rotation_residual_deg": result.mean_axxb_rotation_residual_deg,
        "max_axxb_rotation_residual_deg": result.max_axxb_rotation_residual_deg,
        "base_target_translation_spread_m": result.base_target_translation_spread_m,
        "base_target_rotation_spread_deg": result.base_target_rotation_spread_deg,
        "rotation_system_singular_values": result.rotation_system_singular_values,
        "translation_system_rank": result.translation_system_rank,
        "pair_residuals": [
            {
                "sample_a": row.sample_a,
                "sample_b": row.sample_b,
                "translation_residual_m": row.translation_residual_m,
                "rotation_residual_deg": row.rotation_residual_deg,
                "motion_rotation_deg": row.motion_rotation_deg,
            }
            for row in result.pair_residuals
        ],
    }


def _solve_rotation(a_motions: list[np.ndarray], b_motions: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    for a, b in zip(a_motions, b_motions):
        q_a = _canonical_quaternion(_rotation_to_quaternion(a[:3, :3]))
        q_b = _canonical_quaternion(_rotation_to_quaternion(b[:3, :3]))
        rows.append(_left_quat_matrix(q_a) - _right_quat_matrix(q_b))
    system = np.vstack(rows)
    _, singular_values, vh = np.linalg.svd(system)
    q_x = _canonical_quaternion(vh[-1, :])
    rotation = _quaternion_to_rotation(q_x)
    return rotation, singular_values


def _solve_translation(rotation: np.ndarray, a_motions: list[np.ndarray], b_motions: list[np.ndarray]) -> tuple[np.ndarray, int]:
    lhs_rows: list[np.ndarray] = []
    rhs_rows: list[np.ndarray] = []
    eye = np.eye(3, dtype=np.float64)
    for a, b in zip(a_motions, b_motions):
        lhs_rows.append(a[:3, :3] - eye)
        rhs_rows.append(rotation @ b[:3, 3] - a[:3, 3])
    lhs = np.vstack(lhs_rows)
    rhs = np.concatenate(rhs_rows, axis=0)
    rank = int(np.linalg.matrix_rank(lhs))
    translation, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    return np.ascontiguousarray(translation, dtype=np.float64), rank


def _pair_residuals(
    samples: list[HandEyeSample],
    pairs: list[tuple[int, int, np.ndarray, np.ndarray, float]],
    gripper_T_camera: np.ndarray,
) -> list[HandEyePairResidual]:
    residuals: list[HandEyePairResidual] = []
    for i, j, a_ij, b_ij, motion_angle in pairs:
        lhs = a_ij @ gripper_T_camera
        rhs = gripper_T_camera @ b_ij
        delta = invert_transform(lhs) @ rhs
        delta = validate_transform(delta, name="AXXB residual")
        residuals.append(
            HandEyePairResidual(
                sample_a=samples[i].sample_id,
                sample_b=samples[j].sample_id,
                translation_residual_m=float(np.linalg.norm(delta[:3, 3])),
                rotation_residual_deg=_rotation_angle_deg(delta[:3, :3]),
                motion_rotation_deg=float(motion_angle),
            )
        )
    return residuals


def _base_target_spread(
    base_T_gripper: list[np.ndarray],
    camera_T_target: list[np.ndarray],
    gripper_T_camera: np.ndarray,
) -> tuple[float, float]:
    transforms = [validate_transform(g @ gripper_T_camera @ c, name="base_T_target") for g, c in zip(base_T_gripper, camera_T_target)]
    origins = np.asarray([item[:3, 3] for item in transforms], dtype=np.float64)
    center = origins.mean(axis=0)
    translation_spread = float(np.max(np.linalg.norm(origins - center[None, :], axis=1)))
    quats = np.asarray([_canonical_quaternion(_rotation_to_quaternion(item[:3, :3])) for item in transforms], dtype=np.float64)
    mean_quat = _canonical_quaternion(np.linalg.svd(quats.T @ quats)[0][:, 0])
    angles = []
    for quat in quats:
        dot = abs(float(np.dot(mean_quat, quat)))
        dot = max(-1.0, min(1.0, dot))
        angles.append(np.rad2deg(2.0 * np.arccos(dot)))
    return translation_spread, float(np.max(angles)) if angles else 0.0


def _rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s])
    else:
        axis = int(np.argmax(np.diag(r)))
        if axis == 0:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            q = np.array([(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s])
        elif axis == 1:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            q = np.array([(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s])
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            q = np.array([(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s])
    return _canonical_quaternion(q)


def _quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = _canonical_quaternion(quaternion)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _canonical_quaternion(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        raise ValueError("zero quaternion")
    q = q / norm
    if q[0] < 0.0:
        q = -q
    return q


def _left_quat_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [[w, -x, -y, -z], [x, w, -z, y], [y, z, w, -x], [z, -y, x, w]],
        dtype=np.float64,
    )


def _right_quat_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [[w, -x, -y, -z], [x, w, z, -y], [y, -z, w, x], [z, y, -x, w]],
        dtype=np.float64,
    )


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    rot = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    cos_angle = (float(np.trace(rot)) - 1.0) * 0.5
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return float(np.rad2deg(np.arccos(cos_angle)))


def _hand_eye_cfg(config: dict[str, Any]) -> dict[str, Any]:
    cfg = get_nested(config, ("calibration", "hand_eye"), {}) or {}
    if not isinstance(cfg, dict):
        raise ValueError("calibration.hand_eye must be a mapping")
    thresholds = cfg.get("thresholds") if isinstance(cfg.get("thresholds"), dict) else {}
    return {**cfg, **thresholds}
