#!/usr/bin/env python3
"""Validate the distributed real-grasp stack without moving the robot."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from graspgen_s600_tools.runtime import (  # noqa: E402
    S600GraspRuntime,
    get_nested,
    load_yaml_config,
    normalize_point_cloud,
    result_to_response,
    validate_point_cloud,
)
from graspgen_s600_tools.runtime.calibration import (  # noqa: E402
    calibration_metadata,
    load_robot_base_T_camera_for_observation,
    observation_pose_metadata,
)
from graspgen_s600_tools.runtime.grasp_filtering import select_grasp_candidate  # noqa: E402
from real_grasp_pi_client import (  # noqa: E402
    S600InferenceClient,
    build_camera_adapter,
    build_robot_adapter,
)

logger = logging.getLogger("validate_real_grasp_stack")


def make_synthetic_object_pc(n: int = 2048, seed: int = 0) -> np.ndarray:
    """Small box surface point cloud in meters for non-hardware validation."""

    rng = np.random.default_rng(seed)
    hx, hy, hz = 0.04, 0.03, 0.05
    points = []
    per_face = max(1, n // 6)
    for axis in range(3):
        for sign in (-1.0, 1.0):
            uv = rng.uniform(-1.0, 1.0, size=(per_face, 2)).astype(np.float32)
            face = np.zeros((per_face, 3), dtype=np.float32)
            other = [i for i in range(3) if i != axis]
            extents = [hx, hy, hz]
            face[:, axis] = sign * extents[axis]
            face[:, other[0]] = uv[:, 0] * extents[other[0]]
            face[:, other[1]] = uv[:, 1] * extents[other[1]]
            points.append(face)
    pc = np.concatenate(points, axis=0)
    if len(pc) < n:
        pad = pc[rng.integers(0, len(pc), n - len(pc))]
        pc = np.concatenate([pc, pad], axis=0)
    return np.ascontiguousarray(pc[:n], dtype=np.float32)


def check_ssh(config: dict[str, Any], *, timeout_s: int = 5) -> dict[str, Any]:
    """Check SSH connectivity without running hardware commands."""

    user = str(get_nested(config, ("pi", "user"), "pi"))
    host = str(get_nested(config, ("pi", "host"), "192.168.1.156"))
    target = f"{user}@{host}"
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={timeout_s}",
        target,
        "hostname",
    ]
    t0 = time.monotonic()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s + 3,
        check=False,
    )
    return {
        "target": target,
        "returncode": proc.returncode,
        "elapsed_ms": (time.monotonic() - t0) * 1000.0,
        "output": proc.stdout.strip(),
        "ok": proc.returncode == 0,
    }


def local_or_zmq_infer(
    config: dict[str, Any],
    point_cloud: np.ndarray,
    *,
    use_local_mock: bool,
    server_host: str | None,
    server_port: int | None,
) -> dict[str, Any]:
    if use_local_mock:
        runtime = S600GraspRuntime(config, backend="mock")
        return result_to_response(runtime.infer(point_cloud), request_id="local-mock-validation")

    host = server_host or str(get_nested(config, ("network", "inference_host"), "127.0.0.1"))
    port = int(server_port or get_nested(config, ("network", "inference_port"), 5556))
    timeout_ms = int(get_nested(config, ("network", "timeout_ms"), 60000))
    payload = {
        "action": "infer",
        "request_id": "validation-" + str(int(time.time())),
        "timestamp": time.time(),
        "point_cloud": point_cloud,
        "params": {"dry_run": True},
    }
    with S600InferenceClient(host, port, timeout_ms) as client:
        print("server health:", client.request({"action": "health"}))
        print("server metadata backend:", client.request({"action": "metadata"}).get("backend"))
        return client.request(payload)


def validate_response(response: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    grasps = np.asarray(response.get("grasps", []), dtype=np.float32)
    conf = np.asarray(response.get("confidences", []), dtype=np.float32)
    if response.get("status") != "ok":
        errors.append(f"response status is not ok: {response.get('status')}")
    if grasps.ndim != 3 or grasps.shape[1:] != (4, 4):
        errors.append(f"grasps must be (M,4,4), got {grasps.shape}")
    if conf.ndim != 1 or conf.shape[0] != grasps.shape[0]:
        errors.append(f"confidences shape {conf.shape} does not match grasps {grasps.shape}")
    if grasps.size and not np.isfinite(grasps).all():
        errors.append("grasps contain non-finite values")
    if conf.size and not np.isfinite(conf).all():
        errors.append("confidences contain non-finite values")
    if response.get("safety", {}).get("motion_authorized", True):
        errors.append("server response must not authorize motion")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "real_grasp_test.yaml"),
        help="real-grasp YAML config",
    )
    parser.add_argument("--point-cloud", default=None, help="override camera.point_cloud_path")
    parser.add_argument("--server-host", default=None, help="local S600 server host override")
    parser.add_argument("--server-port", type=int, default=None, help="local S600 server port override")
    parser.add_argument("--skip-ssh", action="store_true", help="skip Pi SSH connectivity check")
    parser.add_argument(
        "--local-mock",
        action="store_true",
        help="use local mock runtime instead of connecting to ZMQ server",
    )
    parser.add_argument(
        "--synthetic-if-no-camera-file",
        action="store_true",
        help="use synthetic point cloud if no file camera path is configured",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s: %(message)s")
    config = load_yaml_config(args.config)
    if args.point_cloud:
        config.setdefault("camera", {})["point_cloud_path"] = args.point_cloud

    failures: list[str] = []
    if not args.skip_ssh:
        try:
            ssh = check_ssh(config)
            print("ssh check:", ssh)
            if not ssh["ok"]:
                failures.append(f"SSH check failed: {ssh['output']}")
        except Exception as exc:
            failures.append(f"SSH check error: {exc}")

    camera = build_camera_adapter(config)
    robot = build_robot_adapter(config)
    camera_health = camera.health_check()
    robot_health = robot.health_check()
    print("camera health:", camera_health)
    print("robot health:", robot_health)
    if camera_health.get("status") == "error":
        failures.append(f"camera health failed: {camera_health}")
    if robot_health.get("status") == "error":
        failures.append(f"robot health failed: {robot_health}")

    try:
        scene = camera.capture()
        point_cloud = scene.object_point_cloud
        scene_point_cloud = scene.scene_point_cloud
        scene_metadata = scene.metadata
        print(f"loaded/captured point cloud from frame={scene.frame_id}: {point_cloud.shape}")
        print(
            "scene summary:",
            {
                "scene_shape": None if scene_point_cloud is None else scene_point_cloud.shape,
                "metadata": scene_metadata,
            },
        )
    except Exception as exc:
        if args.synthetic_if_no_camera_file:
            print(f"camera capture unavailable ({exc}); using synthetic dry-run point cloud")
            point_cloud = make_synthetic_object_pc(seed=int(get_nested(config, ("runtime", "seed"), 0)))
            scene_point_cloud = None
            scene_metadata = {"source": "synthetic"}
        else:
            raise

    min_points = int(get_nested(config, ("safety", "min_points_before_resample"), 256))
    max_abs = float(get_nested(config, ("safety", "max_abs_coord_m"), 5.0))
    checked = validate_point_cloud(point_cloud, min_points=min_points, max_abs_coord_m=max_abs)
    normalized = normalize_point_cloud(
        checked,
        num_points=int(get_nested(config, ("runtime", "num_points"), 2048)),
        seed=int(get_nested(config, ("runtime", "seed"), 0)),
        min_points=min_points,
        max_abs_coord_m=max_abs,
    )
    print(
        "point cloud check:",
        {
            "input_shape": checked.shape,
            "bounds_min": checked.min(axis=0),
            "bounds_max": checked.max(axis=0),
            "model_input_shape": normalized.model_input.shape,
            "centroid": normalized.centroid,
            "scale": normalized.scale,
        },
    )

    try:
        response = local_or_zmq_infer(
            config,
            checked,
            use_local_mock=args.local_mock,
            server_host=args.server_host,
            server_port=args.server_port,
        )
        print(
            "inference response:",
            {
                "status": response.get("status"),
                "backend": response.get("backend"),
                "effective_backend": response.get("effective_backend"),
                "num_grasps": response.get("num_grasps"),
                "safety": response.get("safety"),
                "warnings": response.get("warnings"),
            },
        )
        failures.extend(validate_response(response))
        calibration, observation_pose = load_robot_base_T_camera_for_observation(config, robot_health)
        observation_pose_info = observation_pose_metadata(observation_pose)
        print(
            "calibration:",
            {
                **calibration_metadata(calibration),
                "observation_pose": observation_pose_info,
                "observation_transform_trusted": calibration is not None,
            },
        )
        if (
            get_nested(config, ("calibration", "robot_base_T_camera_path"), None) not in (None, "")
            and bool(get_nested(config, ("calibration", "observation_pose", "require_for_transform"), True))
            and not observation_pose.ok
        ):
            failures.append(f"observation pose validation failed for configured transform: {observation_pose.reason}")
        selection = select_grasp_candidate(
            response,
            config=config,
            robot_base_T_camera=calibration.matrix if calibration is not None else None,
            scene_point_cloud=scene_point_cloud,
            scene_metadata=scene_metadata,
        )
        response["client_safety"] = {
            **selection.safety,
            "observation_pose": observation_pose_info,
            "observation_pose_ok": observation_pose.ok,
            "observation_transform_trusted": calibration is not None,
        }
        response["safety"] = {
            **response.get("safety", {}),
            **selection.safety,
            "observation_pose_ok": observation_pose.ok,
            "observation_transform_trusted": calibration is not None,
            "motion_authorized": False,
        }
        print("candidate filtering:", selection.safety)
        for row in selection.table[: int(get_nested(config, ("runtime", "topk"), 5))]:
            print("candidate:", row)
        if selection.selected_grasp_camera is not None:
            dry_run = robot.dry_run_grasp(
                selection.selected_grasp_camera,
                context={"validation": True, "response": response, "candidate_selection": selection.safety},
            )
            print("robot dry-run check:", dry_run)
            if dry_run.get("status") != "ok":
                failures.append(f"robot dry-run failed: {dry_run}")
        else:
            print("robot dry-run skipped: no candidate passed client-side filtering")
    except Exception as exc:
        failures.append(f"inference/dry-run validation failed: {exc}")

    if failures:
        print("\n❌ Real-grasp validation failed:")
        for failure in failures:
            print(f"  - {failure}")
        print("No robot motion was commanded.")
        return 1

    print("\n✅ Real-grasp validation passed. No robot motion was commanded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
