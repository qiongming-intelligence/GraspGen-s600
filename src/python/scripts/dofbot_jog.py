#!/usr/bin/env python3
"""Direct guarded Yahboom/Dofbot jog helper for camera viewpoint setup.

Positions are read from the servo position register in floating point rather
than through `Arm_Lib.Arm_serial_servo_read`, which truncates to whole degrees
with a per-servo directional bias. Chaining `write(read() + delta)` on the
truncated value re-injects up to a full degree of error on every step, so a
1 deg jog can produce anywhere from no motion to 2 deg of motion. See
`graspgen_s600_tools.runtime.dofbot_servo` for the measurements.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "python"))

from graspgen_s600_tools.runtime import get_nested, load_yaml_config  # noqa: E402
from graspgen_s600_tools.runtime.dofbot_servo import (  # noqa: E402
    read_servo_angles_deg,
    tick_resolution_deg,
    write_servo_angle_deg,
)


def parse_servo_delta(value: str) -> tuple[int, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected SID=DEG")
    sid_text, delta_text = value.split("=", 1)
    try:
        sid = int(sid_text)
        delta = float(delta_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid SID=DEG value: {value!r}") from exc
    if sid not in range(1, 7):
        raise argparse.ArgumentTypeError("servo id must be 1..6")
    if not np.isfinite(delta):
        raise argparse.ArgumentTypeError("delta must be finite")
    return sid, delta


def read_positions(arm: Any) -> dict[int, float | None]:
    readings = read_servo_angles_deg(arm, tuple(range(1, 7)), include_armlib=True)
    return {sid: None if item is None else item.angle_deg for sid, item in readings.items()}


def read_positions_detailed(arm: Any) -> dict[str, Any]:
    """Return both the precise and the `Arm_Lib` angle, to expose the truncation."""

    readings = read_servo_angles_deg(arm, tuple(range(1, 7)), include_armlib=True)
    return {
        str(sid): None
        if item is None
        else {
            "angle_deg": round(item.angle_deg, 3),
            "raw_ticks": item.raw_ticks,
            "armlib_angle_deg": item.armlib_angle_deg,
            "armlib_truncation_error_deg": None
            if item.truncation_error_deg is None
            else round(item.truncation_error_deg, 3),
        }
        for sid, item in sorted(readings.items())
    }


def string_keyed(value: dict[int, Any]) -> dict[str, Any]:
    return {str(key): (round(item, 3) if isinstance(item, float) else item) for key, item in sorted(value.items())}


def servo_range(config: dict[str, Any], sid: int) -> tuple[float, float]:
    ranges = get_nested(config, ("robot", "scripted_motion", "servo_ranges_deg"), {}) or {}
    raw = ranges.get(sid, ranges.get(str(sid), [0.0, 180.0])) if isinstance(ranges, dict) else [0.0, 180.0]
    low, high = float(raw[0]), float(raw[1])
    return low, high


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "real_grasp_orangepi_yahboom_scripted_motion.yaml"),
        help="config with robot arm_lib_path/bus_id/addr and servo ranges",
    )
    parser.add_argument(
        "--servo-delta",
        action="append",
        type=parse_servo_delta,
        required=True,
        help="servo delta as SID=DEG, for example 2=-5; repeatable",
    )
    parser.add_argument("--duration-ms", type=int, default=1000, help="write duration per repeat")
    parser.add_argument("--repeat", type=int, default=1, help="repeat count for the same delta")
    parser.add_argument("--pause-s", type=float, default=0.25, help="pause after each write before readback")
    parser.add_argument("--max-delta-deg", type=float, default=10.0, help="max absolute delta per servo per repeat")
    parser.add_argument("--min-duration-ms", type=int, default=500, help="minimum allowed duration")
    parser.add_argument("--execute", action="store_true", help="send commands; otherwise dry-run only")
    parser.add_argument(
        "--i-understand-this-can-move-the-robot",
        action="store_true",
        help="required with --execute",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")
    if args.duration_ms < args.min_duration_ms:
        raise ValueError(f"duration {args.duration_ms} ms is below minimum {args.min_duration_ms} ms")
    if args.max_delta_deg <= 0 or args.max_delta_deg > 30:
        raise ValueError("--max-delta-deg must be in (0, 30]")

    deltas: dict[int, float] = {}
    for sid, delta in args.servo_delta:
        if abs(delta) > args.max_delta_deg:
            raise ValueError(f"servo {sid} delta {delta} exceeds max {args.max_delta_deg}")
        deltas[sid] = deltas.get(sid, 0.0) + float(delta)
    if not deltas:
        raise ValueError("at least one --servo-delta is required")
    if len(deltas) > 2:
        raise ValueError("jog helper allows at most two servos per command")

    config = load_yaml_config(args.config)
    robot_cfg = get_nested(config, ("robot",), {}) or {}
    arm_lib_path = str(robot_cfg.get("arm_lib_path") or "/home/HwHiAiUser/Arm_Lib")
    bus_id = int(robot_cfg.get("bus_id") or 7)
    addr = int(str(robot_cfg.get("addr") or "0x15"), 0)
    if arm_lib_path not in sys.path:
        sys.path.insert(0, arm_lib_path)
    from Arm_Lib import Arm_Device  # type: ignore

    arm = Arm_Device(bus_id=bus_id, addr=addr)
    initial = read_positions(arm)
    planned_steps: list[dict[str, Any]] = []
    current = dict(initial)
    for repeat_index in range(args.repeat):
        targets: dict[int, float] = {}
        for sid, delta in deltas.items():
            value = current.get(sid)
            if value is None:
                raise RuntimeError(f"servo {sid} readback is missing")
            target = float(value) + float(delta)
            low, high = servo_range(config, sid)
            if target < low or target > high:
                raise RuntimeError(f"servo {sid} target {target} outside range [{low}, {high}]")
            targets[sid] = target
            current[sid] = target
        planned_steps.append({"repeat": repeat_index + 1, "target_servo_positions_deg": string_keyed(targets)})

    result: dict[str, Any] = {
        "adapter": "dofbot_jog",
        "bus_id": bus_id,
        "addr": hex(addr),
        "duration_ms": args.duration_ms,
        "repeat": args.repeat,
        "delta_deg": string_keyed(deltas),
        "initial_servo_positions_deg": string_keyed(initial),
        "initial_servo_detail": read_positions_detailed(arm),
        "tick_resolution_deg": {str(sid): round(tick_resolution_deg(sid), 4) for sid in sorted(deltas)},
        "planned_steps": planned_steps,
        "command_sent": False,
    }
    print("jog plan:")
    print(json.dumps(result, indent=2, sort_keys=True))

    if not args.execute:
        print("Dry-run only. No robot command sent.")
        return 0
    if not args.i_understand_this_can_move_the_robot:
        raise RuntimeError("--execute requires --i-understand-this-can-move-the-robot")

    steps: list[dict[str, Any]] = []
    # Advance a commanded reference rather than re-reading before every step.
    # Re-reading feeds encoder noise and any servo settling error back into the
    # next target, so cumulative displacement drifts away from repeat * delta.
    commanded = {sid: float(initial[sid]) for sid in deltas if initial.get(sid) is not None}
    if len(commanded) != len(deltas):
        raise RuntimeError("every jogged servo needs a valid initial readback")
    for repeat_index in range(args.repeat):
        before = read_positions(arm)
        if hasattr(arm, "Arm_serial_set_torque"):
            arm.Arm_serial_set_torque(1)
            time.sleep(0.1)
        targets: dict[int, float] = {}
        for sid, delta in deltas.items():
            target = commanded[sid] + float(delta)
            low, high = servo_range(config, sid)
            if target < low or target > high:
                raise RuntimeError(f"servo {sid} target {target:.3f} outside range [{low}, {high}]")
            commanded[sid] = target
            targets[sid] = target
            write_servo_angle_deg(arm, int(sid), target, int(args.duration_ms))
            time.sleep(0.05)
        time.sleep(max(args.pause_s, args.duration_ms / 1000.0))
        after = read_positions(arm)
        steps.append(
            {
                "repeat": repeat_index + 1,
                "pre_servo_positions_deg": string_keyed(before),
                "target_servo_positions_deg": string_keyed(targets),
                "post_servo_positions_deg": string_keyed(after),
                "readback_delta_deg": string_keyed(
                    {
                        sid: None
                        if before.get(sid) is None or after.get(sid) is None
                        else float(after[sid]) - float(before[sid])
                        for sid in deltas
                    }
                ),
                "tracking_error_deg": string_keyed(
                    {
                        sid: None if after.get(sid) is None else float(after[sid]) - targets[sid]
                        for sid in deltas
                    }
                ),
            }
        )
    final = read_positions(arm)
    result["command_sent"] = True
    result["command_method"] = "Arm_serial_set_torque+write_servo_angle_deg(float)"
    result["steps"] = steps
    result["final_servo_positions_deg"] = string_keyed(final)
    result["cumulative_displacement_deg"] = string_keyed(
        {
            sid: None
            if final.get(sid) is None or initial.get(sid) is None
            else float(final[sid]) - float(initial[sid])
            for sid in deltas
        }
    )
    result["intended_displacement_deg"] = string_keyed(
        {sid: float(delta) * args.repeat for sid, delta in deltas.items()}
    )
    result["final_servo_detail"] = read_positions_detailed(arm)
    result["status"] = "ok"
    print("jog result:")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
