#!/usr/bin/env python3
"""Safely scout Dofbot camera poses using depth/RGB evidence, not grasp execution."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from graspgen_s600_tools.runtime import (  # noqa: E402
    get_nested,
    load_yaml_config,
    score_table_view,
)
from real_grasp_pi_client import (  # noqa: E402
    build_camera_adapter,
    build_robot_adapter,
    print_health_result,
    validate_scripted_motion_request,
)

logger = logging.getLogger("dofbot_pose_scout")


def _string_keyed(value: dict[int, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in sorted(value.items())}


def _parse_servo_map(value: Any, *, label: str) -> dict[int, float]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    result: dict[int, float] = {}
    for sid in range(1, 7):
        raw = value.get(sid, value.get(str(sid)))
        if raw is None:
            raise ValueError(f"{label} missing servo {sid}")
        target = float(raw)
        if not np.isfinite(target):
            raise ValueError(f"{label} servo {sid} is not finite")
        result[sid] = target
    return result


def _readback_map(value: Any) -> dict[int, float | None]:
    if not isinstance(value, dict):
        return {}
    result: dict[int, float | None] = {}
    for sid in range(1, 7):
        raw = value.get(sid, value.get(str(sid)))
        result[sid] = None if raw is None else float(raw)
    return result


def _named_pose_targets(config: dict[str, Any], pose_name: str) -> tuple[dict[int, float], str]:
    poses = get_nested(config, ("robot", "scripted_motion", "named_poses"), {}) or {}
    if not isinstance(poses, dict) or pose_name not in poses:
        raise ValueError(f"unknown pose {pose_name!r} in robot.scripted_motion.named_poses")
    pose_cfg = poses[pose_name] or {}
    if not isinstance(pose_cfg, dict):
        raise ValueError(f"robot.scripted_motion.named_poses.{pose_name} must be a mapping")
    source = str(pose_cfg.get("source") or f"robot.scripted_motion.named_poses.{pose_name}")
    if source == "calibration.observation_pose":
        obs = get_nested(config, ("calibration", "observation_pose", "servo_positions_deg"), {}) or {}
        return _parse_servo_map(obs, label="calibration.observation_pose.servo_positions_deg"), source
    target = pose_cfg.get("target_servo_positions_deg", pose_cfg.get("servo_positions_deg"))
    return _parse_servo_map(
        target or {},
        label=f"robot.scripted_motion.named_poses.{pose_name}.servo_positions_deg",
    ), source


def _pose_names(config: dict[str, Any], selected: list[str] | None) -> list[str]:
    if selected:
        return selected
    configured = get_nested(config, ("camera", "pose_scout", "candidate_poses"), None)
    if configured is not None:
        if not isinstance(configured, list):
            raise ValueError("camera.pose_scout.candidate_poses must be a list")
        return [str(name) for name in configured]
    poses = get_nested(config, ("robot", "scripted_motion", "named_poses"), {}) or {}
    if not isinstance(poses, dict):
        raise ValueError("robot.scripted_motion.named_poses must be a mapping")
    return [str(name) for name in poses if str(name) != "current"]


def _motion_request_for_pose(
    config: dict[str, Any],
    pose_name: str,
    robot_health: dict[str, Any],
    *,
    duration_ms: int | None,
) -> dict[str, Any]:
    current = _readback_map(robot_health.get("servo_positions_deg"))
    if not current:
        raise RuntimeError("pose scout requires live servo readback")
    target, source = _named_pose_targets(config, pose_name)
    cfg = get_nested(config, ("robot", "scripted_motion"), {}) or {}
    request = {
        "mode": "named-pose",
        "pose_name": pose_name,
        "current_servo_positions_deg": _string_keyed(current),
        "target_servo_positions_deg": _string_keyed(target),
        "duration_ms": int(duration_ms if duration_ms is not None else cfg.get("default_duration_ms", 3000)),
        "source": source,
        "scripted_motion_only": True,
        "grasp_execution": "refused",
    }
    return validate_scripted_motion_request(config, request, robot_health=robot_health)


def _load_rgb(path: str | None) -> np.ndarray | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    suffix = p.suffix.lower()
    if suffix == ".npy":
        return np.load(p)
    if suffix == ".npz":
        data = np.load(p)
        key = "rgb" if "rgb" in data else data.files[0]
        return data[key]
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Pillow is required to load non-NPY RGB images for pose scouting") from exc
    return np.asarray(Image.open(p).convert("RGB"))


def _run_capture_command(command: str | None, *, timeout_s: float) -> dict[str, Any]:
    if not command:
        return {"command_configured": False, "command_ran": False}
    t0 = time.monotonic()
    proc = subprocess.run(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    return {
        "command_configured": True,
        "command_ran": True,
        "command": command,
        "returncode": proc.returncode,
        "elapsed_ms": (time.monotonic() - t0) * 1000.0,
        "output_tail": proc.stdout[-4000:],
        "ok": proc.returncode == 0,
    }


def _capture_and_score(
    config: dict[str, Any],
    *,
    capture_command: str | None,
    capture_timeout_s: float,
    rgb_path: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    capture_info = _run_capture_command(capture_command, timeout_s=capture_timeout_s)
    if capture_info.get("command_configured") and not capture_info.get("ok"):
        raise RuntimeError(f"capture command failed: {capture_info}")
    camera = build_camera_adapter(config)
    scene = camera.capture()
    rgb = _load_rgb(rgb_path or get_nested(config, ("camera", "pose_scout", "rgb_image_path"), None))
    score_points = scene.scene_point_cloud if scene.scene_point_cloud is not None else scene.object_point_cloud
    score = score_table_view(score_points, config=config, rgb_image=rgb)
    result = {
        "scene": {
            "frame_id": scene.frame_id,
            "object_shape": list(scene.object_point_cloud.shape),
            "scene_shape": None if scene.scene_point_cloud is None else list(scene.scene_point_cloud.shape),
            "metadata": scene.metadata,
        },
        "table_view": {
            "ok": score.ok,
            "score": score.score,
            "grade": score.grade,
            "reasons": score.reasons,
            "metrics": score.metrics,
        },
    }
    return capture_info, result


def _write_report(path: str | None, report: dict[str, Any]) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "real_grasp_orangepi_yahboom_scripted_motion.yaml"),
        help="config with scripted poses and camera paths",
    )
    parser.add_argument("--pose", action="append", default=None, help="named pose to scout; repeatable")
    parser.add_argument("--motion-duration-ms", type=int, default=None, help="scripted pose motion duration")
    parser.add_argument("--settle-s", type=float, default=1.0, help="extra settle before capture after a motion")
    parser.add_argument(
        "--capture-command",
        default=None,
        help="optional command that refreshes camera.point_cloud_path / scene_point_cloud_path before scoring",
    )
    parser.add_argument("--capture-timeout-s", type=float, default=30.0)
    parser.add_argument("--rgb-image", default=None, help="optional RGB image/NPY/NPZ to score color evidence")
    parser.add_argument("--report-json", default=None, help="optional path to write full JSON report")
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="score the current saved/captured cloud without validating or moving a named pose",
    )
    parser.add_argument("--execute", action="store_true", help="move to one named pose before scoring")
    parser.add_argument(
        "--i-understand-this-can-move-the-robot",
        action="store_true",
        help="required with --execute",
    )
    parser.add_argument("--yes", action="store_true", help="auto-approve dry-run scoring prompts")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s: %(message)s")
    config = load_yaml_config(args.config)

    poses = _pose_names(config, args.pose)
    if not poses:
        raise RuntimeError("no poses configured for scouting")
    if args.score_only and args.execute:
        raise RuntimeError("--score-only cannot be combined with --execute")
    if args.execute and len(poses) != 1:
        raise RuntimeError("--execute is limited to exactly one --pose per invocation")
    if args.execute and not args.i_understand_this_can_move_the_robot:
        raise RuntimeError("--execute requires --i-understand-this-can-move-the-robot")

    if args.score_only:
        if len(poses) != 1:
            raise RuntimeError("--score-only is limited to exactly one named pose label")
        pose_name = poses[0]
        pose_target, pose_source = _named_pose_targets(config, pose_name)
        capture_info, score_result = _capture_and_score(
            config,
            capture_command=args.capture_command,
            capture_timeout_s=args.capture_timeout_s,
            rgb_path=args.rgb_image,
        )
        report = {
            "status": "ok",
            "adapter": "dofbot_pose_scout",
            "scripted_motion_only": True,
            "grasp_execution": "refused",
            "request_id": "pose-scout-" + uuid4().hex,
            "score_only": True,
            "pose_name": pose_name,
            "pose_source": pose_source,
            "pose_target_servo_positions_deg": _string_keyed(pose_target),
            "capture": capture_info,
            **score_result,
        }
        _write_report(args.report_json, report)
        print("pose scout score-only result:")
        print(json.dumps(report["table_view"], indent=2, sort_keys=True, default=str))
        return 0

    robot = build_robot_adapter(config)
    robot_health = robot.health_check()
    print_health_result("robot", robot_health)
    if robot_health.get("status") == "error":
        raise RuntimeError(f"robot health check failed: {robot_health}")

    report: dict[str, Any] = {
        "status": "ok",
        "adapter": "dofbot_pose_scout",
        "scripted_motion_only": True,
        "grasp_execution": "refused",
        "request_id": "pose-scout-" + uuid4().hex,
        "poses": [],
    }

    for pose_name in poses:
        motion_request = _motion_request_for_pose(
            config,
            pose_name,
            robot_health,
            duration_ms=args.motion_duration_ms,
        )
        dry_run = robot.dry_run_grasp(
            np.eye(4, dtype=np.float32),
            context={"scripted_motion": motion_request, "robot_health": robot_health, "request_id": report["request_id"]},
        )
        entry: dict[str, Any] = {
            "pose_name": pose_name,
            "motion_request": motion_request,
            "dry_run": dry_run,
            "command_sent": False,
        }
        print(f"pose scout dry-run for {pose_name}:")
        print(json.dumps(entry, indent=2, sort_keys=True, default=str))
        if dry_run.get("status") != "ok":
            entry["status"] = "dry_run_failed"
            report["poses"].append(entry)
            continue

        if args.execute:
            if not motion_request.get("delta_deg"):
                execution = {
                    "status": "ok",
                    "adapter": "dofbot_pose_scout",
                    "command_sent": False,
                    "message": "already at requested pose; scoring current view",
                }
            else:
                answer = input(f"FINAL APPROVAL: move to pose {pose_name!r} for scouting? Type 'MOVE' to execute: ").strip()
                if answer != "MOVE":
                    print("Execution not approved; no robot motion sent.")
                    entry["status"] = "not_approved"
                    report["poses"].append(entry)
                    continue
                execution = robot.execute_grasp(
                    np.eye(4, dtype=np.float32),
                    context={"scripted_motion": motion_request, "robot_health": robot_health, "request_id": report["request_id"]},
                )
            entry["execution"] = execution
            entry["command_sent"] = bool(execution.get("command_sent"))
            if execution.get("status") != "ok":
                entry["status"] = "execution_failed"
                report["poses"].append(entry)
                continue
            if args.settle_s > 0:
                time.sleep(args.settle_s)
        elif not args.yes:
            answer = input(f"Score current saved/captured cloud for pose {pose_name!r}? Type 'yes' to continue: ").strip()
            if answer != "yes":
                entry["status"] = "scoring_not_approved"
                report["poses"].append(entry)
                continue

        capture_info, score_result = _capture_and_score(
            config,
            capture_command=args.capture_command,
            capture_timeout_s=args.capture_timeout_s,
            rgb_path=args.rgb_image,
        )
        entry["capture"] = capture_info
        entry.update(score_result)
        entry["status"] = "ok"
        report["poses"].append(entry)
        print(f"pose scout score for {pose_name}:")
        print(json.dumps(score_result["table_view"], indent=2, sort_keys=True, default=str))

        robot_health = robot.health_check()

    scored_poses = [item for item in report["poses"] if item.get("table_view") is not None]
    ranked = sorted(
        scored_poses,
        key=lambda item: float(item.get("table_view", {}).get("score", -1.0)),
        reverse=True,
    )
    report["ranked_pose_names"] = [item.get("pose_name") for item in ranked]
    report["best_pose"] = ranked[0] if ranked else None
    _write_report(args.report_json, report)
    print("pose scout summary:")
    print(
        json.dumps(
            {
                "ranked_pose_names": report["ranked_pose_names"],
                "best": None
                if report["best_pose"] is None
                else {
                    "pose_name": report["best_pose"].get("pose_name"),
                    "score": report["best_pose"].get("table_view", {}).get("score"),
                    "grade": report["best_pose"].get("table_view", {}).get("grade"),
                    "reasons": report["best_pose"].get("table_view", {}).get("reasons"),
                    "command_sent": report["best_pose"].get("command_sent"),
                },
                "report_json": args.report_json,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
