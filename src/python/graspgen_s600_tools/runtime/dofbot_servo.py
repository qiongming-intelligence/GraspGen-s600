"""Sub-degree servo I/O for Yahboom/Dofbot control boards.

`Arm_Lib.Arm_serial_servo_read` truncates the servo's 12-bit position register to
whole degrees with `int()`. The register itself resolves ~0.082 deg/tick and the
measured read noise on this hardware is <= 0.16 deg, so the truncation is the
dominant position error, not the sensor.

The truncation is also directionally biased, because `Arm_Lib` flips servos 2/3/4
*after* truncating:

    servos 1, 5, 6   read == floor(true_angle)   -> reported angle is too low
    servos 2, 3, 4   read == ceil(true_angle)    -> reported angle is too high

That bias breaks the common incremental jog pattern `write(read() + delta)`. Each
step re-injects the truncation error, so on servos 2/3/4 the arm overshoots by up
to a full degree per step and on servos 1/5/6 it undershoots by up to a full
degree per step. With a 1 deg step the per-step error reaches 100% of the
commanded motion.

This module reads the raw register and converts in floating point, and writes
float target angles (the `Arm_Lib` write path already accepts floats and only
quantizes at the final tick conversion). Callers should use
`read_servo_angles_deg` / `write_servo_angle_deg` instead of the `Arm_Lib`
degree API whenever the result feeds kinematics or an incremental motion step.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

# Register scaling used by the Yahboom control board firmware.
_TICK_MIN = 900
_TICK_MAX = 3100
_SPAN_DEG = 180.0
_S5_TICK_MIN = 380
_S5_TICK_MAX = 3700
_S5_SPAN_DEG = 270.0

# Servos whose mechanical direction is inverted relative to the register.
_FLIPPED_SERVOS = frozenset({2, 3, 4})

_READ_COMMAND_BASE = 0x30
_READ_SETTLE_S = 0.004


@dataclass(frozen=True)
class ServoReading:
    """One servo position recovered at full register resolution."""

    servo_id: int
    raw_ticks: int
    angle_deg: float
    armlib_angle_deg: int | None

    @property
    def truncation_error_deg(self) -> float | None:
        """`Arm_Lib`'s reported angle minus the true angle, or None."""

        if self.armlib_angle_deg is None:
            return None
        return float(self.armlib_angle_deg) - self.angle_deg


def servo_limits_deg(servo_id: int) -> tuple[float, float]:
    """Return the mechanical angle range for a servo."""

    _validate_servo_id(servo_id)
    return (0.0, _S5_SPAN_DEG if servo_id == 5 else _SPAN_DEG)


def ticks_to_deg(servo_id: int, raw_ticks: int) -> float:
    """Convert a raw position register value to degrees without truncating."""

    _validate_servo_id(servo_id)
    if servo_id == 5:
        angle = _S5_SPAN_DEG * (raw_ticks - _S5_TICK_MIN) / (_S5_TICK_MAX - _S5_TICK_MIN)
    else:
        angle = _SPAN_DEG * (raw_ticks - _TICK_MIN) / (_TICK_MAX - _TICK_MIN)
    if servo_id in _FLIPPED_SERVOS:
        angle = _SPAN_DEG - angle
    return float(angle)


def deg_to_ticks(servo_id: int, angle_deg: float) -> int:
    """Convert degrees to the raw register value the firmware expects."""

    _validate_servo_id(servo_id)
    angle = float(angle_deg)
    if servo_id in _FLIPPED_SERVOS:
        angle = _SPAN_DEG - angle
    if servo_id == 5:
        ticks = (_S5_TICK_MAX - _S5_TICK_MIN) * angle / _S5_SPAN_DEG + _S5_TICK_MIN
    else:
        ticks = (_TICK_MAX - _TICK_MIN) * angle / _SPAN_DEG + _TICK_MIN
    return int(ticks)


def tick_resolution_deg(servo_id: int) -> float:
    """Return degrees per register tick, the floor on achievable precision."""

    _validate_servo_id(servo_id)
    if servo_id == 5:
        return _S5_SPAN_DEG / (_S5_TICK_MAX - _S5_TICK_MIN)
    return _SPAN_DEG / (_TICK_MAX - _TICK_MIN)


def read_servo_ticks(arm: Any, servo_id: int, *, retries: int = 6) -> int | None:
    """Read one servo's raw position register, retrying transient I2C failures."""

    _validate_servo_id(servo_id)
    if retries < 1:
        raise ValueError("retries must be >= 1")
    register = servo_id + _READ_COMMAND_BASE
    for _ in range(retries):
        try:
            arm.bus.write_byte_data(arm.addr, register, 0x0)
            time.sleep(_READ_SETTLE_S)
            word = arm.bus.read_word_data(arm.addr, register)
        except OSError:
            time.sleep(0.02)
            continue
        if word:
            return (word >> 8 & 0xFF) | (word << 8 & 0xFF00)
        time.sleep(0.02)
    return None


def read_servo_angle_deg(
    arm: Any,
    servo_id: int,
    *,
    samples: int = 3,
    retries: int = 6,
    include_armlib: bool = False,
) -> ServoReading | None:
    """Read one servo at register resolution, median-filtered over `samples`."""

    if samples < 1:
        raise ValueError("samples must be >= 1")
    ticks = [value for value in (read_servo_ticks(arm, servo_id, retries=retries) for _ in range(samples)) if value]
    if not ticks:
        return None
    median = sorted(ticks)[len(ticks) // 2]
    armlib: int | None = None
    if include_armlib:
        for _ in range(retries):
            armlib = arm.Arm_serial_servo_read(servo_id)
            if armlib is not None:
                break
            time.sleep(0.02)
    return ServoReading(
        servo_id=servo_id,
        raw_ticks=median,
        angle_deg=ticks_to_deg(servo_id, median),
        armlib_angle_deg=None if armlib is None else int(armlib),
    )


def read_servo_angles_deg(
    arm: Any,
    servo_ids: list[int] | tuple[int, ...] = (1, 2, 3, 4, 5, 6),
    *,
    samples: int = 3,
    retries: int = 6,
    include_armlib: bool = False,
) -> dict[int, ServoReading | None]:
    """Read several servos at register resolution."""

    return {
        sid: read_servo_angle_deg(arm, sid, samples=samples, retries=retries, include_armlib=include_armlib)
        for sid in servo_ids
    }


def angles_from_readings(readings: dict[int, ServoReading | None]) -> dict[int, float]:
    """Reduce a reading map to `{servo_id: degrees}`, dropping failed reads."""

    return {sid: reading.angle_deg for sid, reading in readings.items() if reading is not None}


def write_servo_angle_deg(arm: Any, servo_id: int, angle_deg: float, duration_ms: int) -> int:
    """Command a float target angle and return the register value actually sent.

    `Arm_Lib.Arm_serial_servo_write` accepts a float and only quantizes at the
    tick conversion, so callers must not pre-round the target to whole degrees.
    """

    _validate_servo_id(servo_id)
    low, high = servo_limits_deg(servo_id)
    angle = float(angle_deg)
    if not low <= angle <= high:
        raise ValueError(f"servo {servo_id} target {angle:.3f} outside mechanical range [{low}, {high}]")
    if duration_ms < 1:
        raise ValueError("duration_ms must be >= 1")
    arm.Arm_serial_servo_write(int(servo_id), angle, int(duration_ms))
    return deg_to_ticks(servo_id, angle)


def _validate_servo_id(servo_id: Any) -> None:
    if not isinstance(servo_id, int) or servo_id not in range(1, 7):
        raise ValueError(f"servo id must be an int in 1..6, got {servo_id!r}")
