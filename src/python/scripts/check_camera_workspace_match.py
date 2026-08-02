#!/usr/bin/env python3
"""Check that the depth camera can actually see the robot's reachable workspace.

This is a preflight gate, not a calibration tool. It never moves the robot and
never runs inference. It answers one question that every downstream stage
silently assumes: does the captured depth overlap the volume the arm can reach?

The failure this exists to catch is an observation pose that drifted. A
wrist-mounted camera aimed a few joints away from the table returns a perfectly
healthy-looking cloud of the far room, the segmentation crop then selects zero
points, and every stage downstream - GraspGen, ICP hand-eye, grasp scoring -
silently consumes nothing at all. Nothing errors; the grasps are just wrong.

Run it against a captured `.ply`/`.npy` cloud before trusting any grasp output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "python"))

from graspgen_s600_tools.runtime import get_nested, load_yaml_config  # noqa: E402

# Near-field histogram resolution, and how many populated bins define the
# "leading edge" window used to distinguish a sensor blind zone from a scene
# that merely happens to start at that distance.
_CUTOFF_BIN_M = 0.05
_CUTOFF_EDGE_BINS = 2
_CUTOFF_WINDOW_BINS = 10


def load_points_m(path: Path, source_units: str) -> np.ndarray:
    """Load an (N,3) cloud and convert it to meters."""

    suffix = path.suffix.lower()
    if suffix == ".npy":
        points = np.load(path)
    elif suffix == ".ply":
        points = _load_ply_xyz(path)
    else:
        raise ValueError(f"unsupported point cloud suffix {suffix!r}; expected .ply or .npy")
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"{path} must hold an (N,3) array, got {points.shape}")
    points = points[:, :3]
    points = points[np.isfinite(points).all(axis=1)]
    if points.size == 0:
        raise ValueError(f"{path} contains no finite points")
    if source_units == "millimeters":
        points = points / 1000.0
    elif source_units != "meters":
        raise ValueError(f"unsupported source_units {source_units!r}")
    return points


def _load_ply_xyz(path: Path) -> np.ndarray:
    rows: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip() == "end_header":
                break
        else:
            raise ValueError(f"{path} has no PLY end_header")
        for line in handle:
            fields = line.split()
            if len(fields) < 3:
                continue
            rows.append((float(fields[0]), float(fields[1]), float(fields[2])))
    if not rows:
        raise ValueError(f"{path} has no vertex rows")
    return np.asarray(rows, dtype=np.float64)


def estimate_reach_m(config: dict[str, Any]) -> tuple[float | None, list[str]]:
    """Sum the configured link offsets to bound the arm's maximum reach."""

    notes: list[str] = []
    joints = get_nested(config, ("robot", "kinematics", "joints"), None)
    if isinstance(joints, dict):
        joints = [joints[key] for key in sorted(joints, key=lambda item: str(item))]
    if not isinstance(joints, list) or not joints:
        notes.append("robot.kinematics.joints is not configured; cannot bound reach")
        return None, notes
    total = 0.0
    for joint in joints:
        if not isinstance(joint, dict):
            continue
        origin = joint.get("origin_xyz_m")
        if isinstance(origin, (list, tuple)) and len(origin) == 3:
            total += float(np.linalg.norm(np.asarray(origin, dtype=np.float64)))
    tool = get_nested(config, ("robot", "kinematics", "tool_offset_xyz_m"), None)
    if isinstance(tool, (list, tuple)) and len(tool) == 3:
        total += float(np.linalg.norm(np.asarray(tool, dtype=np.float64)))
    if total <= 0.0:
        notes.append("configured joint origins sum to zero; cannot bound reach")
        return None, notes
    return total, notes


def detect_near_cutoff_m(depth_m: np.ndarray) -> tuple[float | None, dict[str, Any]]:
    """Report where depth measurements start, and whether the edge looks abrupt.

    This is descriptive only. An abrupt leading edge is equally consistent with a
    sensor blind zone and with a close surface filling the frame, so it is
    reported as evidence and never used on its own to fail the check. The
    binding test is whether any measured point falls within the arm's reach.
    """

    if depth_m.size < 100:
        return None, {"reason": "too few points to judge"}
    edges = np.arange(0.0, float(depth_m.max()) + _CUTOFF_BIN_M, _CUTOFF_BIN_M)
    if edges.size < 3:
        return None, {"reason": "depth span too small to histogram"}
    counts, _ = np.histogram(depth_m, bins=edges)
    populated = np.flatnonzero(counts)
    if populated.size == 0:
        return None, {"reason": "empty depth histogram"}
    first = int(populated[0])
    window = counts[first : first + _CUTOFF_WINDOW_BINS]
    peak_offset = int(np.argmax(window))
    evidence = {
        "first_populated_bin_m": [round(float(edges[first]), 3), round(float(edges[first + 1]), 3)],
        "empty_bins_below_first_populated": first,
        "leading_edge_bin_counts": [int(value) for value in window[:4]],
        "density_peak_offset_bins": peak_offset,
        "leading_edge_is_abrupt": bool(first >= 1 and peak_offset < _CUTOFF_EDGE_BINS),
    }
    return float(edges[first]), evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="config with segmentation and robot.kinematics")
    parser.add_argument("--point-cloud", required=True, type=Path, help="captured .ply or .npy scene cloud")
    parser.add_argument(
        "--source-units",
        default=None,
        choices=["millimeters", "meters"],
        help="override the units of the input cloud; defaults to camera.orbbec.source_units",
    )
    parser.add_argument("--depth-axis", default="z", choices=["x", "y", "z"], help="camera optical axis")
    parser.add_argument("--report", type=Path, default=None, help="write the full report as JSON")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    source_units = args.source_units or str(
        get_nested(config, ("camera", "orbbec", "source_units"), None)
        or get_nested(config, ("calibration", "dynamic_table_markers", "source_units"), None)
        or "meters"
    )
    points = load_points_m(args.point_cloud, source_units)
    axis_index = {"x": 0, "y": 1, "z": 2}[args.depth_axis]
    depth = points[:, axis_index]

    failures: list[str] = []
    warnings: list[str] = []

    report: dict[str, Any] = {
        "point_cloud": str(args.point_cloud),
        "source_units": source_units,
        "point_count": int(points.shape[0]),
        "bbox_min_m": [round(float(v), 4) for v in points.min(axis=0)],
        "bbox_max_m": [round(float(v), 4) for v in points.max(axis=0)],
        "depth_axis": args.depth_axis,
        "depth_min_m": round(float(depth.min()), 4),
        "depth_max_m": round(float(depth.max()), 4),
        "depth_percentiles_m": {
            str(q): round(float(np.percentile(depth, q)), 4) for q in (1, 5, 25, 50, 75, 95, 99)
        },
    }

    cutoff_m, cutoff_evidence = detect_near_cutoff_m(depth)
    report["depth_starts_at_m"] = None if cutoff_m is None else round(cutoff_m, 3)
    report["leading_edge_evidence"] = cutoff_evidence

    reach_m, reach_notes = estimate_reach_m(config)
    report["estimated_max_reach_m"] = None if reach_m is None else round(reach_m, 4)
    report["reach_notes"] = reach_notes

    seg_cfg = get_nested(config, ("camera", "segmentation"), {}) or get_nested(config, ("segmentation",), {}) or {}
    crop = seg_cfg.get("crop_bounds_m") if isinstance(seg_cfg, dict) else None
    if isinstance(crop, dict):
        mask = np.ones(points.shape[0], dtype=bool)
        bounds_report: dict[str, Any] = {}
        for name, index in (("x", 0), ("y", 1), ("z", 2)):
            raw = crop.get(name)
            if not isinstance(raw, (list, tuple)) or len(raw) != 2:
                continue
            low, high = float(raw[0]), float(raw[1])
            bounds_report[name] = [low, high]
            mask &= (points[:, index] >= low) & (points[:, index] <= high)
        report["crop_bounds_m"] = bounds_report
        report["points_in_crop"] = int(mask.sum())
        if int(mask.sum()) == 0:
            failures.append(
                f"segmentation crop selects 0 of {points.shape[0]} points; "
                f"crop {bounds_report} does not intersect the captured depth range "
                f"[{depth.min():.3f}, {depth.max():.3f}] m"
            )
        elif int(mask.sum()) < 256:
            warnings.append(f"segmentation crop selects only {int(mask.sum())} points")
    else:
        warnings.append("no segmentation.crop_bounds_m configured; skipping crop check")

    if cutoff_m is not None and reach_m is not None:
        report["workspace_margin_m"] = round(reach_m - cutoff_m, 4)

    if reach_m is not None:
        in_reach = int((depth <= reach_m).sum())
        report["points_within_reach"] = in_reach
        report["points_within_reach_fraction"] = round(in_reach / points.shape[0], 4)
        if in_reach == 0:
            failures.append(
                f"0 of {points.shape[0]} captured points fall within the arm's {reach_m:.3f} m reach; "
                f"the nearest measured surface is {depth.min():.3f} m away. The camera is almost "
                "certainly not aimed at the workspace - check the observation pose before changing "
                "any crop, calibration, or model setting."
            )
        elif in_reach < points.shape[0] // 20:
            warnings.append(
                f"only {in_reach} of {points.shape[0]} points are within the arm's {reach_m:.3f} m reach; "
                "the camera may be aimed past the workspace"
            )

    report["failures"] = failures
    report["warnings"] = warnings
    report["ok"] = not failures

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.report is not None:
        args.report.write_text(text + "\n", encoding="utf-8")

    for warning in warnings:
        print(f"⚠️  {warning}", file=sys.stderr)
    if failures:
        for failure in failures:
            print(f"❌ {failure}", file=sys.stderr)
        print("\n❌ Camera/workspace preflight FAILED. Grasp output from this setup is meaningless.", file=sys.stderr)
        return 1
    print("\n✅ Camera/workspace preflight passed. No robot motion was commanded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
