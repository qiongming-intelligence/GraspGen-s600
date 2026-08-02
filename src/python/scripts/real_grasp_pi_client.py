#!/usr/bin/env python3
"""Pi-side real-grasp orchestrator with dry-run-first safety gates."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "python"))

from graspgen_s600_tools.runtime import get_nested, load_yaml_config  # noqa: E402
from graspgen_s600_tools.runtime.calibration import (  # noqa: E402
    calibration_metadata,
    load_robot_base_T_camera_for_observation,
    load_robot_base_T_camera_for_planning,
    observation_pose_metadata,
    validate_observation_pose_from_health,
)
from graspgen_s600_tools.runtime.dofbot_servo import (  # noqa: E402
    read_servo_angles_deg,
    write_servo_angle_deg,
)
from graspgen_s600_tools.runtime.grasp_filtering import select_grasp_candidate  # noqa: E402
from graspgen_s600_tools.runtime.perception import (  # noqa: E402
    enumerate_object_clusters,
    segment_object_point_cloud,
    subdivide_cluster,
)

logger = logging.getLogger("real_grasp_pi_client")


@dataclass(frozen=True)
class CapturedScene:
    object_point_cloud: np.ndarray
    scene_point_cloud: np.ndarray | None
    frame_id: str
    timestamp: float
    metadata: dict[str, Any]


class CameraAdapter:
    def health_check(self) -> dict[str, Any]:
        raise NotImplementedError

    def capture(self, robot_base_T_camera: np.ndarray | None = None) -> CapturedScene:
        raise NotImplementedError


class NullCameraAdapter(CameraAdapter):
    def health_check(self) -> dict[str, Any]:
        return {"status": "ok", "adapter": "null", "captures_real_hardware": False}

    def capture(self, robot_base_T_camera: np.ndarray | None = None) -> CapturedScene:
        raise RuntimeError("NullCameraAdapter cannot capture; configure file_point_cloud or a real adapter")


class FilePointCloudCameraAdapter(CameraAdapter):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        camera_cfg = get_nested(config, ("camera",), {}) or {}
        self.path = camera_cfg.get("point_cloud_path")
        self.scene_path = camera_cfg.get("scene_point_cloud_path")
        self.frame_id = str(camera_cfg.get("frame_id") or "camera")
        self.units = str(camera_cfg.get("units") or "meters")

    def health_check(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.path or self.scene_path else "not_configured",
            "adapter": "file_point_cloud",
            "path": self.path,
            "scene_path": self.scene_path,
            "captures_real_hardware": False,
            "segmentation_enabled": bool(get_nested(self.config, ("camera", "segmentation", "enabled"), False)),
        }

    def capture(self, robot_base_T_camera: np.ndarray | None = None) -> CapturedScene:
        if not self.path and not self.scene_path:
            raise RuntimeError(
                "camera.point_cloud_path or camera.scene_point_cloud_path is required for file_point_cloud adapter"
            )
        object_pc = load_point_cloud_file(Path(self.path)) if self.path else None
        scene_pc = load_point_cloud_file(Path(self.scene_path)) if self.scene_path else None
        segmented = segment_object_point_cloud(
            object_point_cloud=object_pc,
            scene_point_cloud=scene_pc,
            config=self.config,
            robot_base_T_camera=robot_base_T_camera,
        )
        metadata = {
            "source": str(self.path) if self.path else None,
            "scene_source": str(self.scene_path) if self.scene_path else None,
            "units": self.units,
        }
        metadata.update(segmented.metadata)
        return CapturedScene(
            object_point_cloud=segmented.object_point_cloud,
            scene_point_cloud=segmented.scene_point_cloud,
            frame_id=self.frame_id,
            timestamp=time.time(),
            metadata=metadata,
        )


class PiV4L2ProbeCameraAdapter(FilePointCloudCameraAdapter):
    """Probe a Pi V4L2 camera while using a point-cloud file for dry-run input."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        camera_cfg = get_nested(config, ("camera",), {}) or {}
        self.device = str(camera_cfg.get("device") or "/dev/video0")
        self.v4l2_ctl = str(camera_cfg.get("v4l2_ctl") or "v4l2-ctl")
        self.probe_capture = bool(camera_cfg.get("probe_capture", True))
        self.synthetic_if_no_point_cloud = bool(camera_cfg.get("synthetic_if_no_point_cloud", False))
        self.probe_output_path = str(
            camera_cfg.get("probe_output_path") or "/tmp/graspgen_v4l2_probe_frame.raw"
        )
        self.probe_width = int(camera_cfg.get("probe_width") or 640)
        self.probe_height = int(camera_cfg.get("probe_height") or 480)
        self.probe_pixelformat = str(camera_cfg.get("probe_pixelformat") or "YUYV")

    def health_check(self) -> dict[str, Any]:
        base = super().health_check()
        result: dict[str, Any] = {
            "status": "ok",
            "adapter": "pi_v4l2_probe",
            "captures_real_hardware": True,
            "device": self.device,
            "point_cloud_path": self.path,
            "point_cloud_status": base["status"],
        }
        device_path = Path(self.device)
        if not device_path.exists():
            result.update({"status": "error", "error": f"camera device not found: {self.device}"})
            return result
        fmt = self._run_v4l2(["--list-formats-ext"])
        result["formats_returncode"] = fmt.returncode
        result["formats_excerpt"] = fmt.stdout[-2000:]
        if fmt.returncode != 0:
            result.update({"status": "error", "error": fmt.stdout.strip()})
            return result
        if self.probe_capture:
            capture = self._run_v4l2(
                [
                    f"--set-fmt-video=width={self.probe_width},height={self.probe_height},"
                    f"pixelformat={self.probe_pixelformat}",
                    "--stream-mmap",
                    "--stream-count=1",
                    f"--stream-to={self.probe_output_path}",
                ],
                timeout_s=10,
            )
            output = Path(self.probe_output_path)
            result["probe_capture_returncode"] = capture.returncode
            result["probe_capture_output"] = capture.stdout[-2000:]
            result["probe_frame_path"] = self.probe_output_path
            result["probe_frame_bytes"] = output.stat().st_size if output.exists() else 0
            if capture.returncode != 0 or result["probe_frame_bytes"] <= 0:
                result.update({"status": "error", "error": capture.stdout.strip()})
                return result
        return result

    def capture(self, robot_base_T_camera: np.ndarray | None = None) -> CapturedScene:
        if self.path:
            scene = super().capture(robot_base_T_camera)
            object_pc = scene.object_point_cloud
            scene_pc = scene.scene_point_cloud
            metadata = dict(scene.metadata)
        elif self.synthetic_if_no_point_cloud:
            object_pc = make_synthetic_object_pc()
            scene_pc = None
            metadata = {"source": "synthetic", "units": self.units}
        else:
            raise RuntimeError(
                "camera.point_cloud_path is required for pi_v4l2_probe unless "
                "camera.synthetic_if_no_point_cloud is true"
            )
        metadata.update(
            {
                "camera_adapter": "pi_v4l2_probe",
                "camera_device": self.device,
                "probe_frame_path": self.probe_output_path,
                "probe_only": True,
            }
        )
        return CapturedScene(
            object_point_cloud=object_pc,
            scene_point_cloud=scene_pc,
            frame_id=self.frame_id,
            timestamp=time.time(),
            metadata=metadata,
        )

    def _run_v4l2(self, args: list[str], *, timeout_s: float = 5.0) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.v4l2_ctl, "-d", self.device, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            check=False,
        )


def summarize_dry_run_context(context: dict[str, Any]) -> dict[str, Any]:
    """Keep deployment logs compact by omitting large ndarray payloads."""

    response = context.get("response") if isinstance(context, dict) else None
    summary = {key: value for key, value in context.items() if key != "response"}
    if isinstance(response, dict):
        summary["response"] = {
            "request_id": response.get("request_id"),
            "status": response.get("status"),
            "backend": response.get("backend"),
            "effective_backend": response.get("effective_backend"),
            "num_grasps": response.get("num_grasps"),
            "timing": response.get("timing"),
            "safety": response.get("safety"),
            "warnings": response.get("warnings"),
        }
    return summary


class RobotAdapter:
    is_real_hardware = False

    def health_check(self) -> dict[str, Any]:
        raise NotImplementedError

    def dry_run_grasp(self, grasp: np.ndarray, *, context: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def execute_grasp(self, grasp: np.ndarray, *, context: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class NullRobotAdapter(RobotAdapter):
    def health_check(self) -> dict[str, Any]:
        return {"status": "ok", "adapter": "null", "moves_real_hardware": False}

    def dry_run_grasp(self, grasp: np.ndarray, *, context: dict[str, Any]) -> dict[str, Any]:
        del grasp, context
        return {"status": "ok", "adapter": "null", "message": "no robot command sent"}

    def execute_grasp(self, grasp: np.ndarray, *, context: dict[str, Any]) -> dict[str, Any]:
        del grasp, context
        raise RuntimeError("NullRobotAdapter refuses real execution")


class YahboomArmHealthCheckAdapter(RobotAdapter):
    """Read-only Dofbot/Yahboom arm adapter for deployment preflight."""

    is_real_hardware = False

    def __init__(self, config: dict[str, Any]) -> None:
        robot_cfg = get_nested(config, ("robot",), {}) or {}
        self.arm_lib_path = str(robot_cfg.get("arm_lib_path") or "/home/pi/Arm_Lib")
        self.read_positions = bool(robot_cfg.get("read_positions", True))
        self.read_version = bool(robot_cfg.get("read_version", True))
        self.bus_id = int(robot_cfg.get("bus_id") or 1)
        self.addr = int(str(robot_cfg.get("addr") or "0x15"), 0)
        self.probe_buses = [int(bus) for bus in robot_cfg.get("probe_buses", [])]
        self.probe_i2c = bool(robot_cfg.get("probe_i2c", False))

    def health_check(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": "ok",
            "adapter": "yahboom_arm_healthcheck",
            "moves_real_hardware": False,
            "arm_lib_path": self.arm_lib_path,
            "arm_lib_exists": Path(self.arm_lib_path).exists(),
            "bus_id": self.bus_id,
            "addr": hex(self.addr),
            "i2c_device": f"/dev/i2c-{self.bus_id}",
            "i2c_device_exists": Path(f"/dev/i2c-{self.bus_id}").exists(),
            "i2c_device_access": os.access(f"/dev/i2c-{self.bus_id}", os.R_OK | os.W_OK),
            "probe_i2c": self.probe_i2c,
            "probe_buses": self.probe_buses,
        }
        if self.probe_i2c and self.probe_buses:
            result["i2c_scan"] = self._scan_i2c_buses()
        try:
            if self.arm_lib_path not in sys.path:
                sys.path.insert(0, self.arm_lib_path)
            from Arm_Lib import Arm_Device  # type: ignore

            result["arm_device_import"] = "ok"
            arm = Arm_Device(bus_id=self.bus_id, addr=self.addr)
            read_errors: list[str] = []
            if self.read_version:
                result["hardversion"] = arm.Arm_get_hardversion()
                if result["hardversion"] is None:
                    read_errors.append("hardware version did not respond")
            if self.read_positions:
                positions = _read_servo_positions_precise(arm)
                result["servo_positions_deg"] = positions
                result["servo_positions_read"] = True
                result["servo_position_units"] = "deg"
                result["servo_ids"] = list(range(1, 7))
                if all(value is None for value in positions.values()):
                    read_errors.append("no servo positions responded")
            if read_errors:
                result.update(
                    {
                        "status": "error",
                        "error": "; ".join(read_errors),
                        "hint": (
                            "Arm_Lib imported, but the controller did not respond. Verify arm power, "
                            "I2C wiring/pinout, bus_id, and addr before any motion work."
                        ),
                    }
                )
        except Exception as exc:
            result.update(
                {
                    "status": "error",
                    "error": f"arm health check failed: {exc}",
                    "hint": (
                        "Install/copy Arm_Lib so robot.arm_lib_path contains Arm_Lib.py, "
                        "verify I2C bus/address, and keep this adapter read-only."
                    ),
                }
            )
        return result

    def _scan_i2c_buses(self) -> dict[int, dict[str, Any]]:
        scans: dict[int, dict[str, Any]] = {}
        i2cdetect = shutil.which("i2cdetect")
        if i2cdetect is None:
            return {bus: {"status": "error", "error": "i2cdetect not found"} for bus in self.probe_buses}
        for bus in self.probe_buses:
            device = Path(f"/dev/i2c-{bus}")
            entry: dict[str, Any] = {
                "device": str(device),
                "device_exists": device.exists(),
                "device_access": os.access(device, os.R_OK | os.W_OK),
            }
            if not device.exists():
                entry.update({"status": "missing"})
                scans[bus] = entry
                continue
            try:
                proc = subprocess.run(
                    [i2cdetect, "-y", str(bus)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=5,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                output = exc.stdout or ""
                if isinstance(output, bytes):
                    output = output.decode(errors="replace")
                entry.update(
                    {
                        "status": "timeout",
                        "error": "i2cdetect timed out",
                        "output": output[-2000:],
                        "expected_addr_seen": False,
                    }
                )
                scans[bus] = entry
                continue
            entry.update(
                {
                    "status": "ok" if proc.returncode == 0 else "error",
                    "returncode": proc.returncode,
                    "output": proc.stdout[-2000:],
                    "expected_addr_seen": self._i2cdetect_output_has_addr(proc.stdout, self.addr),
                }
            )
            scans[bus] = entry
        return scans

    @staticmethod
    def _i2cdetect_output_has_addr(output: str, addr: int) -> bool:
        token = f"{addr:02x}"
        for word in output.replace(":", " ").split():
            if word.lower() == token:
                return True
        return False

    def dry_run_grasp(self, grasp: np.ndarray, *, context: dict[str, Any]) -> dict[str, Any]:
        logger.info("ARM DRY RUN grasp pose:\n%s", np.array2string(grasp, precision=5))
        logger.info("ARM DRY RUN context: %s", summarize_dry_run_context(context))
        return {"status": "ok", "adapter": "yahboom_arm_healthcheck", "message": "logged only"}

    def execute_grasp(self, grasp: np.ndarray, *, context: dict[str, Any]) -> dict[str, Any]:
        del grasp, context
        raise RuntimeError("YahboomArmHealthCheckAdapter refuses real execution")


class LoggingRobotAdapter(RobotAdapter):
    def health_check(self) -> dict[str, Any]:
        return {"status": "ok", "adapter": "logging", "moves_real_hardware": False}

    def dry_run_grasp(self, grasp: np.ndarray, *, context: dict[str, Any]) -> dict[str, Any]:
        logger.info("DRY RUN grasp pose:\n%s", np.array2string(grasp, precision=5))
        logger.info("DRY RUN context: %s", summarize_dry_run_context(context))
        return {"status": "ok", "adapter": "logging", "message": "logged only"}

    def execute_grasp(self, grasp: np.ndarray, *, context: dict[str, Any]) -> dict[str, Any]:
        del grasp, context
        raise RuntimeError("LoggingRobotAdapter refuses real execution")


class YahboomArmScriptedMotionAdapter(RobotAdapter):
    """Explicitly gated named/delta Yahboom arm motions only."""

    is_real_hardware = True

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        robot_cfg = get_nested(config, ("robot",), {}) or {}
        self.arm_lib_path = str(robot_cfg.get("arm_lib_path") or "/home/pi/Arm_Lib")
        self.read_positions = bool(robot_cfg.get("read_positions", True))
        self.read_version = bool(robot_cfg.get("read_version", True))
        self.bus_id = int(robot_cfg.get("bus_id") or 1)
        self.addr = int(str(robot_cfg.get("addr") or "0x15"), 0)

    def health_check(self) -> dict[str, Any]:
        cfg = _scripted_motion_cfg(self.config)
        result: dict[str, Any] = {
            "status": "ok",
            "adapter": "yahboom_arm_scripted_motion",
            "moves_real_hardware": True,
            "motion_enabled_by_config": bool(cfg.get("enabled", False)),
            "supported_modes": ["named-pose", "observation-pose", "delta"],
            "arm_lib_path": self.arm_lib_path,
            "arm_lib_exists": Path(self.arm_lib_path).exists(),
            "bus_id": self.bus_id,
            "addr": hex(self.addr),
            "i2c_device": f"/dev/i2c-{self.bus_id}",
            "i2c_device_exists": Path(f"/dev/i2c-{self.bus_id}").exists(),
            "i2c_device_access": os.access(f"/dev/i2c-{self.bus_id}", os.R_OK | os.W_OK),
            "scripted_motion": _scripted_motion_public_config(self.config),
        }
        try:
            arm = self._create_arm_device()
            result["arm_device_import"] = "ok"
            read_errors: list[str] = []
            if self.read_version:
                result["hardversion"] = arm.Arm_get_hardversion()
                if result["hardversion"] is None:
                    read_errors.append("hardware version did not respond")
            if self.read_positions:
                positions = self._read_servo_positions(arm)
                result["servo_positions_deg"] = positions
                result["servo_positions_read"] = True
                result["servo_position_units"] = "deg"
                result["servo_ids"] = list(range(1, 7))
                if all(value is None for value in positions.values()):
                    read_errors.append("no servo positions responded")
            if read_errors:
                result.update(
                    {
                        "status": "error",
                        "error": "; ".join(read_errors),
                        "hint": (
                            "Arm_Lib imported, but the controller did not respond. Verify arm power, "
                            "I2C wiring/pinout, bus_id, and addr before scripted motion."
                        ),
                    }
                )
        except Exception as exc:
            result.update(
                {
                    "status": "error",
                    "error": f"arm scripted-motion health check failed: {exc}",
                    "hint": (
                        "Install/copy Arm_Lib so robot.arm_lib_path contains Arm_Lib.py, "
                        "verify I2C bus/address, and keep motion disabled until dry-runs pass."
                    ),
                }
            )
        return result

    def dry_run_grasp(self, grasp: np.ndarray, *, context: dict[str, Any]) -> dict[str, Any]:
        del grasp
        request = _scripted_motion_request_from_context(context)
        planned = validate_scripted_motion_request(
            self.config,
            request,
            robot_health=context.get("robot_health") if isinstance(context, dict) else None,
        )
        logger.info("SCRIPTED ARM DRY RUN: %s", planned)
        return {
            "status": "ok",
            "adapter": "yahboom_arm_scripted_motion",
            "command_sent": False,
            "scripted_motion_only": True,
            "grasp_execution": "refused",
            "mode": planned["mode"],
            "pose_name": planned.get("pose_name"),
            "source": planned.get("source"),
            "current_servo_positions_deg": planned["current_servo_positions_deg"],
            "target_servo_positions_deg": planned["target_servo_positions_deg"],
            "delta_deg": planned["delta_deg"],
            "duration_ms": planned["duration_ms"],
        }

    def execute_grasp(self, grasp: np.ndarray, *, context: dict[str, Any]) -> dict[str, Any]:
        del grasp
        request = _scripted_motion_request_from_context(context)
        live_health = self.health_check()
        if live_health.get("status") == "error":
            raise RuntimeError(f"robot health check failed before scripted motion: {live_health}")
        planned = validate_scripted_motion_request(self.config, request, robot_health=live_health)
        arm = self._create_arm_device()
        pre_positions = _normalize_servo_readback(live_health.get("servo_positions_deg"), allow_none=True)
        planned_delta = _normalize_servo_target_map(
            planned["delta_deg"],
            label="scripted_motion.delta_deg",
            require_all=False,
        )
        changed_servos = [sid for sid, delta in planned_delta.items() if abs(float(delta)) > 1e-6]
        if not changed_servos:
            raise RuntimeError("scripted motion has no changed servos to execute")
        command_method = self._send_servo_command(arm, planned, pre_positions, changed_servos)
        settle_s = float(_scripted_motion_cfg(self.config).get("post_move_settle_s", 0.5))
        if settle_s > 0:
            time.sleep(settle_s)
        post_positions = _normalize_servo_readback(self._read_servo_positions(arm), allow_none=True)
        tolerance = _readback_tolerance_map(self.config, changed_servos)
        failed: list[int] = []
        deltas: dict[int, float | None] = {}
        planned_targets = _normalize_servo_target_map(
            planned["target_servo_positions_deg"],
            label="scripted_motion.target_servo_positions_deg",
            require_all=False,
        )
        for sid in changed_servos:
            actual = post_positions.get(sid)
            target = planned_targets.get(sid)
            if actual is None or target is None:
                deltas[sid] = None
                failed.append(sid)
                continue
            delta = abs(float(actual) - float(target))
            deltas[sid] = delta
            if delta > tolerance[sid]:
                failed.append(sid)
        readback_ok = not failed
        return {
            "status": "ok" if readback_ok else "error",
            "adapter": "yahboom_arm_scripted_motion",
            "command_sent": True,
            "command_method": command_method,
            "scripted_motion_only": True,
            "grasp_execution": "refused",
            "mode": planned["mode"],
            "pose_name": planned.get("pose_name"),
            "source": planned.get("source"),
            "pre_servo_positions_deg": pre_positions,
            "target_servo_positions_deg": planned["target_servo_positions_deg"],
            "post_servo_positions_deg": post_positions,
            "readback_ok": readback_ok,
            "readback_failed_servos": failed,
            "readback_delta_deg": deltas,
            "readback_tolerance_deg": tolerance,
            "duration_ms": planned["duration_ms"],
            "post_move_settle_s": settle_s,
        }

    def _create_arm_device(self) -> Any:
        if self.arm_lib_path not in sys.path:
            sys.path.insert(0, self.arm_lib_path)
        from Arm_Lib import Arm_Device  # type: ignore

        return Arm_Device(bus_id=self.bus_id, addr=self.addr)

    @staticmethod
    def _read_servo_positions(arm: Any) -> dict[int, float | None]:
        return _read_servo_positions_precise(arm)

    @staticmethod
    def _send_servo_command(
        arm: Any,
        planned: dict[str, Any],
        pre_positions: dict[int, float | None],
        changed_servos: list[int],
    ) -> str:
        del pre_positions
        duration_ms = int(planned["duration_ms"])
        targets = _normalize_servo_target_map(
            planned["target_servo_positions_deg"],
            label="scripted_motion.target_servo_positions_deg",
            require_all=False,
        )
        if not hasattr(arm, "Arm_serial_servo_write"):
            raise RuntimeError("Arm_Lib exposes no supported single-servo write method")
        if hasattr(arm, "Arm_serial_set_torque"):
            arm.Arm_serial_set_torque(1)
            time.sleep(0.2)
            method = "Arm_serial_set_torque+write_servo_angle_deg(float)"
        else:
            method = "write_servo_angle_deg(float)"
        for sid in changed_servos:
            if sid not in targets:
                raise RuntimeError(f"target for servo {sid} is missing")
            # Do not pre-round. `Arm_serial_servo_write` accepts a float and only
            # quantizes at the 0.0818 deg tick conversion, so rounding here would
            # throw away most of a tick's worth of precision on every command.
            write_servo_angle_deg(arm, int(sid), float(targets[sid]), duration_ms)
            time.sleep(0.1)
        return method


def _scripted_motion_cfg(config: dict[str, Any]) -> dict[str, Any]:
    cfg = get_nested(config, ("robot", "scripted_motion"), {}) or {}
    if not isinstance(cfg, dict):
        raise ValueError("robot.scripted_motion must be a mapping")
    return cfg


def _scripted_motion_public_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = _scripted_motion_cfg(config)
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "refuse_grasp_execution": bool(cfg.get("refuse_grasp_execution", True)),
        "servo_ids": _configured_servo_ids(config),
        "servo_ranges_deg": cfg.get("servo_ranges_deg", {}),
        "max_delta_deg": cfg.get("max_delta_deg", {"default": 1.0}),
        "max_delta_total_deg": float(cfg.get("max_delta_total_deg", 1.0)),
        "max_changed_servos": int(cfg.get("max_changed_servos", 1)),
        "default_duration_ms": int(cfg.get("default_duration_ms", 3000)),
        "min_duration_ms": int(cfg.get("min_duration_ms", 800)),
        "max_duration_ms": int(cfg.get("max_duration_ms", 5000)),
        "readback_tolerance_deg": cfg.get("readback_tolerance_deg", {"default": 3.0}),
        "write_strategy": "torque_enable_then_single_servo",
        "named_poses": sorted((cfg.get("named_poses") or {}).keys()),
    }


def _parse_servo_id(value: Any) -> int:
    try:
        sid = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid servo id {value!r}") from exc
    if sid not in range(1, 7):
        raise ValueError(f"unsupported servo id {value!r}; expected 1..6")
    return sid


def _configured_servo_ids(config: dict[str, Any]) -> list[int]:
    raw = _scripted_motion_cfg(config).get("servo_ids", [1, 2, 3, 4, 5, 6])
    if not isinstance(raw, list):
        raise ValueError("robot.scripted_motion.servo_ids must be a list")
    ids = [_parse_servo_id(item) for item in raw]
    if len(set(ids)) != len(ids):
        raise ValueError("robot.scripted_motion.servo_ids contains duplicates")
    return ids


def _read_servo_positions_precise(arm: Any) -> dict[int, float | None]:
    """Read every servo from the position register in floating point.

    `Arm_Lib.Arm_serial_servo_read` truncates with `int()` and flips servos
    2/3/4 *after* truncating, so it over-reports 2/3/4 and under-reports 1/5/6
    by up to a full degree. These values feed observation-pose validation and
    FK, where a degree of bias is not affordable, so read the raw ticks and
    convert in float instead. See `runtime.dofbot_servo` for the measurements.
    """

    readings = read_servo_angles_deg(arm, tuple(range(1, 7)))
    return {sid: None if item is None else round(item.angle_deg, 3) for sid, item in readings.items()}


def _normalize_servo_readback(value: Any, *, allow_none: bool) -> dict[int, float | None]:
    if value in (None, ""):
        return {}
    if not isinstance(value, dict):
        raise ValueError("servo_positions_deg must be a mapping")
    result: dict[int, float | None] = {}
    for key, raw in value.items():
        sid = _parse_servo_id(key)
        if raw is None:
            if allow_none:
                result[sid] = None
            continue
        servo_value = float(raw)
        if not np.isfinite(servo_value):
            raise ValueError(f"servo {sid} readback is not finite")
        result[sid] = servo_value
    return result


def _normalize_servo_target_map(value: Any, *, label: str, require_all: bool) -> dict[int, float]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    result: dict[int, float] = {}
    for sid in range(1, 7):
        raw = value.get(sid, value.get(str(sid)))
        if raw is None:
            if require_all:
                raise ValueError(f"{label} missing servo {sid}")
            continue
        target = float(raw)
        if not np.isfinite(target):
            raise ValueError(f"{label} servo {sid} is not finite")
        result[sid] = target
    for key in value:
        _parse_servo_id(key)
    return result


def _map_value(value: Any, sid: int, default: float) -> float:
    if isinstance(value, dict):
        raw = value.get(sid, value.get(str(sid), value.get("default", default)))
    elif value is None:
        raw = default
    else:
        raw = value
    result = float(raw)
    if not np.isfinite(result):
        raise ValueError(f"non-finite servo limit for servo {sid}")
    return result


def _servo_range(config: dict[str, Any], sid: int) -> tuple[float, float]:
    ranges = _scripted_motion_cfg(config).get("servo_ranges_deg", {}) or {}
    raw = ranges.get(sid, ranges.get(str(sid), [0.0, 180.0])) if isinstance(ranges, dict) else [0.0, 180.0]
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError(f"servo range for servo {sid} must be [min, max]")
    low, high = float(raw[0]), float(raw[1])
    if not np.isfinite(low) or not np.isfinite(high) or low > high:
        raise ValueError(f"invalid servo range for servo {sid}: {raw}")
    return low, high


def _max_delta_map(config: dict[str, Any], servo_ids: list[int]) -> dict[int, float]:
    raw = _scripted_motion_cfg(config).get("max_delta_deg", {"default": 1.0})
    return {sid: _map_value(raw, sid, 1.0) for sid in servo_ids}


def _readback_tolerance_map(config: dict[str, Any], servo_ids: list[int]) -> dict[int, float]:
    raw = _scripted_motion_cfg(config).get("readback_tolerance_deg", {"default": 3.0})
    return {sid: _map_value(raw, sid, 3.0) for sid in servo_ids}


def _string_keyed_servo_map(value: dict[int, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in sorted(value.items())}


def parse_servo_delta_args(values: list[str] | None) -> dict[int, float]:
    deltas: dict[int, float] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"--servo-delta must be SID=DEG, got {item!r}")
        sid_text, delta_text = item.split("=", 1)
        sid = _parse_servo_id(sid_text.strip())
        if sid in deltas:
            raise ValueError(f"duplicate --servo-delta for servo {sid}")
        delta = float(delta_text.strip())
        if not np.isfinite(delta):
            raise ValueError(f"servo {sid} delta is not finite")
        deltas[sid] = delta
    return deltas


def _scripted_motion_duration_ms(args: argparse.Namespace, config: dict[str, Any]) -> int:
    cfg = _scripted_motion_cfg(config)
    duration = args.motion_duration_ms if args.motion_duration_ms is not None else cfg.get("default_duration_ms", 3000)
    return int(duration)


def _observation_pose_targets(config: dict[str, Any]) -> dict[int, float]:
    pose_cfg = get_nested(config, ("calibration", "observation_pose"), {}) or {}
    return _normalize_servo_target_map(
        pose_cfg.get("servo_positions_deg") or {},
        label="calibration.observation_pose.servo_positions_deg",
        require_all=True,
    )


def _named_pose_targets(config: dict[str, Any], pose_name: str) -> tuple[dict[int, float], str]:
    named_poses = _scripted_motion_cfg(config).get("named_poses", {}) or {}
    if not isinstance(named_poses, dict) or pose_name not in named_poses:
        raise ValueError(f"unknown scripted motion pose {pose_name!r}")
    pose_cfg = named_poses[pose_name] or {}
    if not isinstance(pose_cfg, dict):
        raise ValueError(f"robot.scripted_motion.named_poses.{pose_name} must be a mapping")
    source = str(pose_cfg.get("source") or f"robot.scripted_motion.named_poses.{pose_name}")
    if source == "calibration.observation_pose":
        return _observation_pose_targets(config), source
    target = pose_cfg.get("target_servo_positions_deg", pose_cfg.get("servo_positions_deg"))
    return (
        _normalize_servo_target_map(
            target or {},
            label=f"robot.scripted_motion.named_poses.{pose_name}.servo_positions_deg",
            require_all=True,
        ),
        source,
    )


def build_scripted_motion_request(
    args: argparse.Namespace,
    config: dict[str, Any],
    robot_health: dict[str, Any],
) -> dict[str, Any]:
    mode = str(args.scripted_motion)
    if str(get_nested(config, ("robot", "adapter"), "null")) != "yahboom_arm_scripted_motion":
        raise RuntimeError("scripted motion requires robot.adapter: yahboom_arm_scripted_motion")
    if not bool(_scripted_motion_cfg(config).get("enabled", False)):
        raise RuntimeError("scripted motion requires robot.scripted_motion.enabled: true")
    current = _normalize_servo_readback(robot_health.get("servo_positions_deg"), allow_none=True)
    if not current:
        raise RuntimeError("scripted motion requires live servo readback")
    duration_ms = _scripted_motion_duration_ms(args, config)
    pose_name: str | None = None
    source: str
    if mode == "delta":
        delta = parse_servo_delta_args(args.servo_delta)
        if not delta:
            raise ValueError("--scripted-motion delta requires at least one --servo-delta SID=DEG")
        target = {sid: value for sid, value in current.items() if value is not None}
        for sid, delta_value in delta.items():
            if current.get(sid) is None:
                raise RuntimeError(f"live readback is required before moving servo {sid}")
            target[sid] = float(current[sid]) + float(delta_value)
        source = "cli.delta"
    elif mode == "observation-pose":
        pose_name = "observation"
        target = _observation_pose_targets(config)
        source = "calibration.observation_pose"
    elif mode == "named-pose":
        pose_name = str(args.scripted_pose or "observation")
        target, source = _named_pose_targets(config, pose_name)
    else:
        raise ValueError(f"unsupported scripted motion mode: {mode}")
    delta = {}
    for sid, target_value in target.items():
        if current.get(sid) is None:
            raise RuntimeError(f"live readback is required before moving servo {sid}")
        raw_delta = float(target_value) - float(current[sid])
        if abs(raw_delta) > 1e-6:
            delta[sid] = raw_delta
    request = {
        "mode": mode,
        "pose_name": pose_name,
        "current_servo_positions_deg": current,
        "target_servo_positions_deg": target,
        "delta_deg": delta,
        "duration_ms": duration_ms,
        "source": source,
        "scripted_motion_only": True,
        "grasp_execution": "refused",
    }
    return validate_scripted_motion_request(config, request, robot_health=robot_health)


def _scripted_motion_request_from_context(context: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(context, dict) or "scripted_motion" not in context:
        raise RuntimeError("refusing arbitrary GraspGen grasp execution; scripted_motion context is required")
    request = context["scripted_motion"]
    if not isinstance(request, dict):
        raise RuntimeError("scripted_motion context must be a mapping")
    if not bool(request.get("scripted_motion_only", False)) or request.get("grasp_execution") != "refused":
        raise RuntimeError("scripted_motion request must explicitly refuse GraspGen grasp execution")
    return request


def validate_scripted_motion_request(
    config: dict[str, Any],
    request: dict[str, Any],
    *,
    robot_health: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if str(get_nested(config, ("robot", "adapter"), "null")) != "yahboom_arm_scripted_motion":
        raise RuntimeError("configured robot adapter is not yahboom_arm_scripted_motion")
    cfg = _scripted_motion_cfg(config)
    if not bool(cfg.get("enabled", False)):
        raise RuntimeError("robot.scripted_motion.enabled is false")
    if not bool(cfg.get("refuse_grasp_execution", True)):
        raise RuntimeError("robot.scripted_motion.refuse_grasp_execution must remain true")
    if request.get("motion_authorized") or request.get("grasp_authorized"):
        raise RuntimeError("scripted motion requests must not carry grasp/motion authorization")
    if not bool(request.get("scripted_motion_only", False)) or request.get("grasp_execution") != "refused":
        raise RuntimeError("scripted motion must explicitly refuse arbitrary GraspGen grasp execution")
    mode = str(request.get("mode") or "")
    if mode not in {"delta", "named-pose", "observation-pose"}:
        raise ValueError(f"unsupported scripted motion mode: {mode!r}")
    duration_ms = int(request.get("duration_ms", cfg.get("default_duration_ms", 3000)))
    min_duration = int(cfg.get("min_duration_ms", 800))
    max_duration = int(cfg.get("max_duration_ms", 5000))
    if duration_ms < min_duration or duration_ms > max_duration:
        raise ValueError(f"duration_ms={duration_ms} outside [{min_duration}, {max_duration}]")
    if mode == "observation-pose":
        pose_cfg = get_nested(config, ("calibration", "observation_pose"), {}) or {}
        if not bool(pose_cfg.get("enabled", False)):
            raise ValueError("observation-pose scripted motion requires calibration.observation_pose.enabled")
    pose_name = request.get("pose_name")
    if mode == "named-pose" and not pose_name:
        raise ValueError("named-pose scripted motion requires pose_name")

    configured_ids = set(_configured_servo_ids(config))
    current = _normalize_servo_readback(request.get("current_servo_positions_deg"), allow_none=True)
    if robot_health is not None:
        live_current = _normalize_servo_readback(robot_health.get("servo_positions_deg"), allow_none=True)
        if live_current:
            current = live_current
    if not current:
        raise RuntimeError("scripted motion requires live servo readback")

    target = _normalize_servo_target_map(
        request.get("target_servo_positions_deg") or {},
        label="scripted_motion.target_servo_positions_deg",
        require_all=False,
    )
    if mode == "delta":
        raw_delta = _normalize_servo_target_map(
            request.get("delta_deg") or {},
            label="scripted_motion.delta_deg",
            require_all=False,
        )
        if not raw_delta:
            raise ValueError("delta scripted motion requires at least one non-zero delta")
        target = {sid: float(value) for sid, value in current.items() if value is not None}
        for sid, delta_value in raw_delta.items():
            if current.get(sid) is None:
                raise RuntimeError(f"live readback is required before moving servo {sid}")
            target[sid] = float(current[sid]) + float(delta_value)
        delta = {sid: float(value) for sid, value in raw_delta.items() if abs(float(value)) > 1e-6}
    else:
        if not target:
            raise ValueError("named/observation scripted motion requires target_servo_positions_deg")
        delta = {}
        for sid, target_value in target.items():
            if current.get(sid) is None:
                raise RuntimeError(f"live readback is required before moving servo {sid}")
            value = float(target_value) - float(current[sid])
            if abs(value) > 1e-6:
                delta[sid] = value

    for sid in set(target) | set(delta):
        if sid not in configured_ids:
            raise ValueError(f"servo {sid} is not enabled in robot.scripted_motion.servo_ids")
    for sid, target_value in target.items():
        low, high = _servo_range(config, sid)
        if target_value < low or target_value > high:
            raise ValueError(f"servo {sid} target {target_value} outside range [{low}, {high}]")
    max_delta = _max_delta_map(config, list(configured_ids))
    for sid, value in delta.items():
        if abs(float(value)) > max_delta[sid]:
            raise ValueError(f"servo {sid} delta {value:.3f} exceeds limit {max_delta[sid]:.3f} deg")
    max_delta_total = float(cfg.get("max_delta_total_deg", 1.0))
    total_delta = float(sum(abs(float(value)) for value in delta.values()))
    if total_delta > max_delta_total:
        raise ValueError(f"total delta {total_delta:.3f} exceeds limit {max_delta_total:.3f} deg")
    max_changed = int(cfg.get("max_changed_servos", 1))
    if len(delta) > max_changed:
        raise ValueError(f"changed servo count {len(delta)} exceeds limit {max_changed}")

    return {
        "mode": mode,
        "pose_name": pose_name,
        "current_servo_positions_deg": _string_keyed_servo_map(current),
        "target_servo_positions_deg": _string_keyed_servo_map(target),
        "delta_deg": _string_keyed_servo_map(delta),
        "duration_ms": duration_ms,
        "source": request.get("source"),
        "scripted_motion_only": True,
        "grasp_execution": "refused",
    }


def make_synthetic_object_pc(n: int = 2048, seed: int = 0) -> np.ndarray:
    """Small box surface point cloud in meters for deployment preflight."""

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


def load_point_cloud_file(path: Path) -> np.ndarray:
    """Load a simple point cloud file for dry-run testing."""

    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        data = np.load(path)
        key = "points" if "points" in data else data.files[0]
        arr = data[key]
    elif suffix in {".xyz", ".txt", ".csv"}:
        arr = np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
    else:
        raise ValueError(f"unsupported point cloud file type: {path.suffix}")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"point cloud must be (N, >=3), got {arr.shape}")
    return np.ascontiguousarray(arr[:, :3], dtype=np.float32)


def build_camera_adapter(config: dict[str, Any]) -> CameraAdapter:
    name = str(get_nested(config, ("camera", "adapter"), "null"))
    if name == "null":
        return NullCameraAdapter()
    if name == "file_point_cloud":
        return FilePointCloudCameraAdapter(config)
    if name == "pi_v4l2_probe":
        return PiV4L2ProbeCameraAdapter(config)
    raise ValueError(f"unsupported camera adapter: {name}")


def build_robot_adapter(config: dict[str, Any]) -> RobotAdapter:
    name = str(get_nested(config, ("robot", "adapter"), "null"))
    if name == "null":
        return NullRobotAdapter()
    if name == "logging":
        return LoggingRobotAdapter()
    if name == "yahboom_arm_healthcheck":
        return YahboomArmHealthCheckAdapter(config)
    if name == "yahboom_arm_scripted_motion":
        return YahboomArmScriptedMotionAdapter(config)
    raise ValueError(
        f"unsupported robot adapter: {name}; add a real adapter only after hardware API is known"
    )


class S600InferenceClient:
    def __init__(self, host: str, port: int, timeout_ms: int, *, wait: bool = False) -> None:
        self.addr = f"tcp://{host}:{port}"
        self.timeout_ms = timeout_ms
        self._zmq, self._msgpack = self._load_stack()
        self.ctx = self._zmq.Context()
        self.socket = self._create_socket()
        if wait:
            self.wait_for_server()

    @staticmethod
    def _load_stack():
        try:
            import msgpack
            import msgpack_numpy
            import zmq
        except ImportError as exc:  # pragma: no cover - depends on Pi env
            raise RuntimeError(
                "pyzmq, msgpack, and msgpack-numpy are required on the Pi client"
            ) from exc
        msgpack_numpy.patch()
        return zmq, msgpack

    def _create_socket(self):
        sock = self.ctx.socket(self._zmq.REQ)
        sock.setsockopt(self._zmq.RCVTIMEO, self.timeout_ms)
        sock.setsockopt(self._zmq.SNDTIMEO, self.timeout_ms)
        sock.setsockopt(self._zmq.LINGER, 0)
        sock.connect(self.addr)
        return sock

    def wait_for_server(self, retry_interval_s: float = 2.0) -> None:
        while True:
            try:
                self.request({"action": "health"})
                return
            except Exception:
                logger.info("waiting for inference server at %s", self.addr)
                time.sleep(retry_interval_s)

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.socket.send(self._msgpack.packb(payload, use_bin_type=True))
        raw = self.socket.recv()
        response = self._msgpack.unpackb(raw, raw=False)
        if response.get("status") == "error" or "error" in response:
            raise RuntimeError(response.get("error", response))
        return response

    def close(self) -> None:
        self.socket.close()
        self.ctx.term()

    def __enter__(self) -> "S600InferenceClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def assert_scripted_motion_execution_allowed(
    config: dict[str, Any],
    robot: RobotAdapter,
    *,
    dry_run_succeeded: bool,
    acknowledged: bool,
    robot_health: dict[str, Any],
    motion_request: dict[str, Any],
) -> None:
    """Fail closed before any real scripted arm motion."""

    if not acknowledged:
        raise RuntimeError("scripted motion execution requires --i-understand-this-can-move-the-robot")
    if not isinstance(robot, YahboomArmScriptedMotionAdapter):
        raise RuntimeError("scripted motion execution requires YahboomArmScriptedMotionAdapter")
    if not robot.is_real_hardware:
        raise RuntimeError("configured robot adapter does not move real hardware")
    if not dry_run_succeeded:
        raise RuntimeError("same-session scripted dry-run must succeed before execute")
    if str(get_nested(config, ("robot", "adapter"), "null")) != "yahboom_arm_scripted_motion":
        raise RuntimeError("configured robot adapter is not yahboom_arm_scripted_motion")
    if not bool(_scripted_motion_cfg(config).get("enabled", False)):
        raise RuntimeError("robot.scripted_motion.enabled is false")
    if robot_health.get("status") == "error":
        raise RuntimeError(f"robot health check failed: {robot_health}")
    validate_scripted_motion_request(config, motion_request, robot_health=robot_health)


def assert_execution_allowed(
    config: dict[str, Any],
    robot: RobotAdapter,
    response: dict[str, Any],
    *,
    dry_run_succeeded: bool,
    acknowledged: bool,
    robot_health: dict[str, Any],
) -> None:
    """Fail closed before any real robot motion."""

    if not acknowledged:
        raise RuntimeError("real execution requires --i-understand-this-can-move-the-robot")
    if not robot.is_real_hardware:
        raise RuntimeError("configured robot adapter does not move real hardware")
    if not dry_run_succeeded:
        raise RuntimeError("same-session dry-run must succeed before execute")
    if response.get("effective_backend") == "mock":
        raise RuntimeError("refusing real execution from mock inference results")
    safety = response.get("safety", {})
    if not safety.get("finite_checked", False):
        raise RuntimeError("server did not confirm finite grasp outputs")
    if get_nested(config, ("safety", "require_workspace_bounds_for_execute"), True):
        if not safety.get("workspace_checked", False) or not safety.get("workspace_ok", False):
            raise RuntimeError("workspace check is required and did not pass")
    if get_nested(config, ("safety", "require_collision_check_for_execute"), True):
        if not safety.get("collision_checked", False) or not safety.get("collision_ok", False):
            raise RuntimeError("collision check is required and did not pass")
    if get_nested(config, ("safety", "require_reachability_check_for_execute"), True):
        if not safety.get("reachability_checked", False) or not safety.get("reachability_ok", False):
            raise RuntimeError("reachability check is required and did not pass")
    calibration, validation = load_robot_base_T_camera_for_planning(config, robot_health)
    if calibration is None:
        raise RuntimeError("trusted robot_base_T_camera planning transform is required for execute")
    if validation.get("mode") == "fixed":
        observation = validation.get("observation_pose") or {}
        if not bool(observation.get("ok", False)):
            raise RuntimeError(f"fixed observation pose validation failed: {observation.get('reason')}")
    elif validation.get("mode") == "eye_in_hand":
        eye_in_hand = validation.get("eye_in_hand") or {}
        if not bool(eye_in_hand.get("ok", False)):
            raise RuntimeError(f"eye-in-hand transform validation failed: {eye_in_hand.get('reason')}")


def print_health_result(label: str, result: dict[str, Any]) -> None:
    print(f"{label} health:")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


def print_observation_pose_result(config: dict[str, Any], robot_health: dict[str, Any]) -> dict[str, Any]:
    observation_pose = validate_observation_pose_from_health(config, robot_health)
    metadata = observation_pose_metadata(observation_pose)
    print("observation pose:")
    print(json.dumps(metadata, indent=2, sort_keys=True, default=str))
    return metadata


def print_candidate_summary(
    response: dict[str, Any],
    topk: int,
    *,
    candidate_table: list[dict[str, Any]] | None = None,
    scene_metadata: dict[str, Any] | None = None,
) -> None:
    grasps = np.asarray(response.get("grasps", []), dtype=np.float32)
    conf = np.asarray(response.get("confidences", []), dtype=np.float32)
    ranked = np.asarray(response.get("ranked_indices", []), dtype=np.int64)
    print("\n==================== GRASP CANDIDATES ====================")
    safety = response.get("safety", {}) or {}
    print(f"status={response.get('status')} backend={response.get('backend')} effective={response.get('effective_backend')}")
    print(f"num_grasps={len(grasps)} warnings={response.get('warnings', [])}")
    print(f"safety={safety}")
    reject_summary = safety.get("reject_summary")
    if reject_summary:
        print("reject_summary=" + json.dumps(reject_summary, sort_keys=True))
    distance_summary = _candidate_distance_summary(grasps, scene_metadata=scene_metadata)
    if distance_summary:
        print("camera_translation_distance_to_object=" + json.dumps(distance_summary, sort_keys=True))
    if candidate_table is None:
        for rank, idx in enumerate(ranked[:topk]):
            pose = grasps[int(idx)]
            score = float(conf[int(idx)]) if int(idx) < len(conf) else float("nan")
            print(f"#{rank + 1}: index={int(idx)} confidence={score:.4f} translation={pose[:3, 3]}")
    else:
        for row in candidate_table[:topk]:
            print(
                "#{rank}: index={idx} confidence={score:.4f} cam={cam} robot={robot} "
                "accepted={accepted} selected={selected} reason={reason}".format(
                    rank=int(row["rank"]) + 1,
                    idx=int(row["index"]),
                    score=float(row.get("confidence", float("nan"))),
                    cam=np.asarray(row.get("camera_translation_m"), dtype=np.float32),
                    robot=row.get("robot_translation_m"),
                    accepted=row.get("accepted"),
                    selected=row.get("selected"),
                    reason=row.get("reason"),
                )
            )
    print("==========================================================\n")


def _candidate_distance_summary(
    grasps: np.ndarray,
    *,
    scene_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    if grasps.ndim != 3 or grasps.shape[0] == 0:
        return {}
    centroid_raw = (scene_metadata or {}).get("object_centroid")
    if centroid_raw is None:
        return {}
    centroid = np.asarray(centroid_raw, dtype=np.float32).reshape(-1)
    if centroid.shape[0] != 3 or not np.isfinite(centroid).all():
        return {}
    trans = np.asarray(grasps, dtype=np.float32)[:, :3, 3]
    distances = np.linalg.norm(trans - centroid[None, :], axis=1)
    return {
        "centroid_m": centroid.astype(float).tolist(),
        "min_m": float(distances.min()),
        "median_m": float(np.median(distances)),
        "mean_m": float(distances.mean()),
        "max_m": float(distances.max()),
    }


def print_scene_summary(scene: CapturedScene) -> None:
    metadata = scene.metadata
    summary_keys = [
        "segmentation_enabled",
        "segmentation_method",
        "segmentation_source",
        "source_point_count",
        "crop_output_count",
        "table_filter_output_count",
        "cluster_output_count",
        "object_point_count",
        "object_centroid",
        "object_bounds_min",
        "object_bounds_max",
        "scene_point_count",
        "table_plane_value_m",
        "table_plane_axis_name",
    ]
    summary = {key: metadata.get(key) for key in summary_keys if key in metadata}
    if "object_bounds_min" in metadata and "object_bounds_max" in metadata:
        lo = np.asarray(metadata["object_bounds_min"], dtype=np.float32)
        hi = np.asarray(metadata["object_bounds_max"], dtype=np.float32)
        if lo.shape == (3,) and hi.shape == (3,):
            summary["object_extent_m"] = (hi - lo).astype(float).tolist()
    source_count = metadata.get("source_point_count")
    crop_count = metadata.get("crop_output_count")
    if source_count and crop_count is not None:
        summary["crop_keep_fraction"] = float(crop_count) / float(source_count)
    print(
        f"captured object cloud shape={scene.object_point_cloud.shape} "
        f"scene_shape={None if scene.scene_point_cloud is None else scene.scene_point_cloud.shape} "
        f"frame={scene.frame_id}"
    )
    print("scene summary:", json.dumps(summary, indent=2, sort_keys=True, default=str))


def run_scripted_motion(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Run the pure scripted-motion branch without camera capture or S600 inference."""

    robot = build_robot_adapter(config)
    robot_health = robot.health_check()
    print_health_result("robot", robot_health)
    print_observation_pose_result(config, robot_health)
    if robot_health.get("status") == "error":
        raise RuntimeError(f"robot health check failed: {robot_health}")

    motion_request = build_scripted_motion_request(args, config, robot_health)
    print("scripted motion request:")
    print(json.dumps(motion_request, indent=2, sort_keys=True, default=str))

    dry_run_result = robot.dry_run_grasp(
        np.eye(4, dtype=np.float32),
        context={
            "scripted_motion": motion_request,
            "robot_health": robot_health,
            "request_id": "scripted-motion-" + uuid4().hex,
        },
    )
    print("scripted motion dry-run result:")
    print(json.dumps(dry_run_result, indent=2, sort_keys=True, default=str))
    dry_run_succeeded = dry_run_result.get("status") == "ok"
    if not args.execute:
        print("Scripted motion dry-run complete. No robot motion was commanded.")
        return 0 if dry_run_succeeded else 1

    assert_scripted_motion_execution_allowed(
        config,
        robot,
        dry_run_succeeded=dry_run_succeeded,
        acknowledged=args.i_understand_this_can_move_the_robot,
        robot_health=robot_health,
        motion_request=motion_request,
    )
    answer = input("FINAL APPROVAL: execute scripted arm motion? Type 'MOVE' to execute: ").strip()
    if answer != "MOVE":
        print("Execution not approved; no robot motion sent.")
        return 0
    result = robot.execute_grasp(
        np.eye(4, dtype=np.float32),
        context={"scripted_motion": motion_request, "robot_health": robot_health},
    )
    print("scripted motion execution result:")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result.get("status") == "ok" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "real_grasp_test.yaml"),
        help="real-grasp YAML config",
    )
    parser.add_argument("--server-host", default=None, help="local S600 server host override")
    parser.add_argument("--server-port", type=int, default=None, help="local S600 server port override")
    parser.add_argument("--point-cloud", default=None, help="override camera.point_cloud_path")
    parser.add_argument("--topk", type=int, default=None, help="number of candidates to print")
    parser.add_argument(
        "--num-diffusion-steps",
        type=int,
        default=None,
        help="override runtime.num_diffusion_steps for this inference request",
    )
    parser.add_argument("--wait", action="store_true", help="wait for inference server")
    parser.add_argument(
        "--scripted-motion",
        choices=["none", "named-pose", "observation-pose", "delta"],
        default="none",
        help="run an explicitly gated Yahboom scripted motion instead of camera/S600 inference",
    )
    parser.add_argument("--scripted-pose", default=None, help="named pose for --scripted-motion named-pose")
    parser.add_argument(
        "--servo-delta",
        action="append",
        default=[],
        help="servo delta for --scripted-motion delta, format SID=DEG; repeatable",
    )
    parser.add_argument("--motion-duration-ms", type=int, default=None, help="scripted motion duration override")
    parser.add_argument("--execute", action="store_true", help="attempt real robot execution after all gates")
    parser.add_argument(
        "--i-understand-this-can-move-the-robot",
        action="store_true",
        help="second explicit acknowledgement required with --execute",
    )
    parser.add_argument("--yes", action="store_true", help="auto-approve dry-run prompt only")
    parser.add_argument(
        "--health-only",
        action="store_true",
        help="run camera and robot health checks only; no capture, inference, dry-run, or execution",
    )
    parser.add_argument(
        "--robot-health-only",
        action="store_true",
        help="run robot health check only; no camera, inference, dry-run, or execution",
    )
    parser.add_argument(
        "--camera-health-only",
        action="store_true",
        help="run camera health check only; no robot, inference, dry-run, or execution",
    )
    parser.add_argument(
        "--multi-object",
        action="store_true",
        help=(
            "segment the scene into separate objects and plan a grasp for each in turn "
            "instead of collapsing it to the single largest cluster (planning only)"
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def plan_object_cluster(
    *,
    object_points: np.ndarray,
    scene: CapturedScene,
    config: dict[str, Any],
    calibration_matrix: np.ndarray | None,
    host: str,
    port: int,
    timeout_ms: int,
    topk: int,
    num_diffusion_steps: int | None,
    wait: bool,
) -> tuple[dict[str, Any], Any]:
    """Run inference plus the full client gate chain for ONE object cluster.

    Used by the multi-object path, which needs the same request/gate sequence
    repeated per cluster. The scene cloud stays the WHOLE scene on purpose: the
    collision gate must see the neighbouring objects, otherwise picking one block
    out of a group would ignore the ones beside it.

    Returns `(response, selection)`. Never moves the robot.

    `response["timing"]` gains `client_request_ms` and `client_gate_ms` so a
    multi-object run says where its wall clock went. Worth having: once the
    persistent-handle helper cut inference to ~0.5s, a target still took ~100s, and
    without this split there was no way to tell the ZMQ round trip (which ships the
    whole scene cloud) apart from the gate chain (which scans it per candidate).
    """

    request_id = uuid4().hex
    params: dict[str, Any] = {"topk": topk, "dry_run": True}
    if num_diffusion_steps is not None:
        params["num_diffusion_steps"] = num_diffusion_steps
    payload = {
        "action": "infer",
        "request_id": request_id,
        "point_cloud": object_points,
        "scene_point_cloud": scene.scene_point_cloud,
        "camera_frame": scene.frame_id,
        "params": params,
        "timestamp": time.time(),
    }
    t_request = time.monotonic()
    with S600InferenceClient(host, port, timeout_ms, wait=wait) as client:
        response = client.request(payload)
    request_ms = (time.monotonic() - t_request) * 1000.0
    if response.get("request_id") != request_id:
        raise RuntimeError("server response request_id mismatch")

    t_gate = time.monotonic()
    selection = select_grasp_candidate(
        response,
        config=config,
        robot_base_T_camera=calibration_matrix,
        scene_point_cloud=scene.scene_point_cloud,
        scene_metadata=scene.metadata,
    )
    gate_ms = (time.monotonic() - t_gate) * 1000.0
    response["timing"] = {
        **(response.get("timing") or {}),
        "client_request_ms": request_ms,
        "client_gate_ms": gate_ms,
    }
    response["safety"] = {
        **response.get("safety", {}),
        **selection.safety,
        "motion_authorized": False,
    }
    return response, selection


def run_multi_object_plan(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    scene: CapturedScene,
    calibration_matrix: np.ndarray | None,
    host: str,
    port: int,
    timeout_ms: int,
    topk: int,
) -> int:
    """Segment the scene into separate objects and plan a grasp for each in turn.

    The normal path collapses the scene to a single object because the network
    takes one cloud. With several blocks close together that silently plans for
    whichever cluster is biggest and ignores the rest, so this enumerates every
    off-table cluster and runs the pipeline once per cluster, best-first.

    Planning only - no motion, and no `--execute` support here. Sequential picking
    needs place poses and a re-capture between objects, neither of which exists
    yet, and grasp execution is refused by config regardless
    (`safety.refuse_grasp_execution`).
    """

    if calibration_matrix is None:
        raise RuntimeError(
            "--multi-object needs a trusted robot_base_T_camera: clusters are found in "
            "the base frame so the table cut and the workspace test both depend on it"
        )
    if scene.scene_point_cloud is None:
        raise RuntimeError("--multi-object needs a full scene cloud to cluster")

    clusters = enumerate_object_clusters(
        scene_point_cloud=scene.scene_point_cloud,
        config=config,
        robot_base_T_camera=calibration_matrix,
    )
    aperture = get_nested(config, ("safety", "collision", "finger_opening_m"), None)
    print(f"\nsegmented {len(clusters)} point cluster(s) above the table:")
    for i, rec in enumerate(clusters):
        cx, cy, cz = rec["centroid_m"]
        note = ""
        if rec["points_in_workspace"] <= 0:
            note = "  [outside workspace_bounds - unreachable]"
        elif aperture is not None and rec["narrowest_width_m"] > float(aperture):
            # A cluster wider than the jaws is NOT necessarily ungraspable. Objects
            # standing against each other merge into one cluster - abutting 25mm
            # cubes read as a single 88x70mm block - and any individual cube still
            # fits. Whether a specific grasp fits is decided per-candidate by the
            # collision gate, which tests the actual gripper volume; this line is
            # informational only.
            note = (
                f"  [cluster spans {rec['narrowest_width_m'] * 1000:.1f}mm > "
                f"{float(aperture) * 1000:.1f}mm jaws: either one large object, or "
                f"several touching ones - per-candidate collision decides]"
            )
        print(
            f"  [{i}] points={rec['point_count']:<7d} in_workspace={rec['points_in_workspace']:<7d} "
            f"centroid=({cx:+.3f}, {cy:+.3f}, {cz:+.3f}) height={rec['height_m'] * 1000:.1f}mm "
            f"narrowest={rec['narrowest_width_m'] * 1000:.1f}mm{note}"
        )
    if not clusters:
        print("no cluster survived the table cut; nothing to plan")
        return 1

    # Expand any cluster wider than the jaws into gripper-sized patches. Touching
    # objects merge into one cluster and cannot be split geometrically, and handing
    # the whole group to the model makes it propose grasps at the group's centre where
    # the finger walls hit a neighbour. One entry per (label, record) to plan.
    targets: list[tuple[str, dict[str, Any]]] = []
    for i, rec in enumerate(clusters):
        if rec["points_in_workspace"] <= 0:
            targets.append((f"{i}", rec))
            continue
        if aperture is not None and rec["narrowest_width_m"] > float(aperture):
            patches = subdivide_cluster(rec, robot_base_T_camera=calibration_matrix)
            if len(patches) > 1:
                print(
                    f"\ncluster {i} is wider than the jaws; split into {len(patches)} "
                    f"gripper-sized patches to plan separately"
                )
                targets.extend((f"{i}.{j}", p) for j, p in enumerate(patches))
                continue
        targets.append((f"{i}", rec))

    planned = 0
    accepted = 0
    for label, rec in targets:
        print(f"\n=== object {label} ({rec['point_count']} points) ===")
        if rec["points_in_workspace"] <= 0:
            print("skipped: no point inside safety.workspace_bounds, the arm cannot reach it")
            continue
        response, selection = plan_object_cluster(
            object_points=rec["points_camera"],
            scene=scene,
            config=config,
            calibration_matrix=calibration_matrix,
            host=host,
            port=port,
            timeout_ms=timeout_ms,
            topk=topk,
            num_diffusion_steps=args.num_diffusion_steps,
            wait=args.wait,
        )
        planned += 1
        timing = response.get("timing") or {}
        print(
            f"timing: request={float(timing.get('client_request_ms', 0.0)) / 1000.0:.1f}s "
            f"(server infer={float(timing.get('infer_ms', 0.0)) / 1000.0:.1f}s) "
            f"gates={float(timing.get('client_gate_ms', 0.0)) / 1000.0:.1f}s"
        )
        print_candidate_summary(
            response,
            topk,
            candidate_table=selection.table,
            scene_metadata=scene.metadata,
        )
        if selection.selected_grasp_camera is None:
            print(f"object {label}: no candidate passed the gate chain")
            continue
        accepted += 1
        grasp = np.asarray(selection.selected_grasp_robot, dtype=np.float64)
        print(
            f"object {label}: candidate {selection.selected_index} accepted, "
            f"base-frame position ({grasp[0, 3]:+.4f}, {grasp[1, 3]:+.4f}, {grasp[2, 3]:+.4f})"
        )

    print(
        f"\nmulti-object planning complete: {accepted}/{planned} planned target(s) "
        f"produced an accepted grasp ({len(clusters)} cluster(s) segmented, "
        f"{len(targets)} target(s) after subdivision). "
        "No robot motion was sent."
    )
    return 0


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s: %(message)s")
    config = load_yaml_config(args.config)
    if args.point_cloud:
        config.setdefault("camera", {})["point_cloud_path"] = args.point_cloud

    if args.scripted_motion != "none":
        return run_scripted_motion(args, config)

    if args.robot_health_only:
        robot = build_robot_adapter(config)
        robot_health = robot.health_check()
        print_health_result("robot", robot_health)
        print_observation_pose_result(config, robot_health)
        return 0 if robot_health.get("status") != "error" else 1

    if args.camera_health_only:
        camera = build_camera_adapter(config)
        camera_health = camera.health_check()
        print_health_result("camera", camera_health)
        return 0 if camera_health.get("status") != "error" else 1

    camera = build_camera_adapter(config)
    robot = build_robot_adapter(config)
    camera_health = camera.health_check()
    robot_health = robot.health_check()
    print_health_result("camera", camera_health)
    print_health_result("robot", robot_health)
    print_observation_pose_result(config, robot_health)
    if args.health_only:
        return 0 if camera_health.get("status") != "error" and robot_health.get("status") != "error" else 1
    if camera_health.get("status") == "error":
        raise RuntimeError(f"camera health check failed: {camera_health}")
    if robot_health.get("status") == "error":
        raise RuntimeError(f"robot health check failed: {robot_health}")

    # Resolve the planning transform BEFORE capture: base-frame segmentation needs
    # it to cut the table, and it is the same transform used to place grasps later.
    calibration, calibration_validation = load_robot_base_T_camera_for_planning(config, robot_health)
    scene = camera.capture(calibration.matrix if calibration is not None else None)
    print_scene_summary(scene)

    host = args.server_host or str(get_nested(config, ("network", "inference_host"), "127.0.0.1"))
    port = int(args.server_port or get_nested(config, ("network", "inference_port"), 5556))
    timeout_ms = int(get_nested(config, ("network", "timeout_ms"), 60000))
    topk = int(args.topk or get_nested(config, ("runtime", "topk"), 5))

    if args.multi_object:
        if args.execute:
            raise RuntimeError(
                "--multi-object is planning only and cannot be combined with --execute"
            )
        return run_multi_object_plan(
            args=args,
            config=config,
            scene=scene,
            calibration_matrix=calibration.matrix if calibration is not None else None,
            host=host,
            port=port,
            timeout_ms=timeout_ms,
            topk=topk,
        )

    request_id = uuid4().hex
    params = {
        "topk": topk,
        "dry_run": True,
    }
    if args.num_diffusion_steps is not None:
        params["num_diffusion_steps"] = args.num_diffusion_steps
    payload = {
        "action": "infer",
        "request_id": request_id,
        "point_cloud": scene.object_point_cloud,
        "scene_point_cloud": scene.scene_point_cloud,
        "camera_frame": scene.frame_id,
        "params": params,
    }

    with S600InferenceClient(host, port, timeout_ms, wait=args.wait) as client:
        health = client.request({"action": "health"})
        metadata = client.request({"action": "metadata"})
        print("server health:", health)
        print("server metadata backend:", metadata.get("backend"))
        payload["timestamp"] = time.time()
        response = client.request(payload)

    if response.get("request_id") != request_id:
        raise RuntimeError("server response request_id mismatch")

    calibration_info = calibration_metadata(calibration)
    observation_pose_info = calibration_validation.get("observation_pose") or {}
    print(
        "calibration:",
        json.dumps(
            {
                **calibration_info,
                **calibration_validation,
                "observation_transform_trusted": calibration is not None,
            },
            indent=2,
            sort_keys=True,
            default=str,
        ),
    )
    selection = select_grasp_candidate(
        response,
        config=config,
        robot_base_T_camera=calibration.matrix if calibration is not None else None,
        scene_point_cloud=scene.scene_point_cloud,
        scene_metadata=scene.metadata,
    )
    calibration_mode = str(calibration_validation.get("mode") or "fixed")
    transform_ok = bool(calibration is not None)
    response["client_safety"] = {
        **selection.safety,
        "calibration_mode": calibration_mode,
        "calibration_validation": calibration_validation,
        "observation_pose": observation_pose_info,
        "observation_pose_ok": bool(observation_pose_info.get("ok", transform_ok)),
        "observation_transform_trusted": transform_ok,
    }
    response["safety"] = {
        **response.get("safety", {}),
        **selection.safety,
        "calibration_mode": calibration_mode,
        "calibration_validation": calibration_validation,
        "observation_pose_ok": bool(observation_pose_info.get("ok", transform_ok)),
        "observation_transform_trusted": transform_ok,
        "motion_authorized": False,
    }
    print_candidate_summary(
        response,
        topk,
        candidate_table=selection.table,
        scene_metadata=scene.metadata,
    )
    if selection.selected_grasp_camera is None:
        print("No candidate passed client-side filtering; no robot command sent.")
        if args.execute:
            raise RuntimeError("cannot execute because no candidate passed client-side filtering")
        print("Dry-run planning complete. Real robot execution was not requested.")
        return 0
    selected_grasp = selection.selected_grasp_camera

    if not args.yes:
        answer = input("Approve dry-run of selected safe candidate? Type 'yes' to continue: ").strip()
        if answer != "yes":
            print("Dry-run not approved; no robot command sent.")
            return 0

    dry_run_result = robot.dry_run_grasp(
        selected_grasp,
        context={
            "request_id": request_id,
            "response": response,
            "candidate_selection": {
                "selected_index": selection.selected_index,
                "safety": selection.safety,
                "selected_grasp_robot": selection.selected_grasp_robot,
            },
            "calibration": calibration_info,
            "observation_pose": observation_pose_info,
        },
    )
    print("dry-run result:", dry_run_result)
    dry_run_succeeded = dry_run_result.get("status") == "ok"

    if args.execute:
        assert_execution_allowed(
            config,
            robot,
            response,
            dry_run_succeeded=dry_run_succeeded,
            acknowledged=args.i_understand_this_can_move_the_robot,
            robot_health=robot_health,
        )
        answer = input("FINAL APPROVAL: move the real robot? Type 'MOVE' to execute: ").strip()
        if answer != "MOVE":
            print("Execution not approved; no robot motion sent.")
            return 0
        print(robot.execute_grasp(selected_grasp, context={"request_id": request_id, "response": response}))
    else:
        print("Dry-run complete. Real robot execution was not requested.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
