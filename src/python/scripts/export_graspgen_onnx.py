#!/usr/bin/env python3
"""
Export GraspGen Generator + Discriminator ONNX models to the paths declared
in configs/manifests/*.json, ready for Horizon S600 HBM compilation.

Outputs:
  models/onnx/graspgen_generator_pointnet.onnx     (single-step denoiser)
  models/onnx/graspgen_discriminator_pointnet.onnx (grasp scorer)

These export the ONNX-compatible architecture (PointNet++ backbone, r3_6d).
Weights are randomly initialized unless --generator-ckpt / --discriminator-ckpt
are provided (Phase 4 supplies trained checkpoints).
"""

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "python"))

from graspgen_s600_tools.models.graspgen_onnx import (
    GraspGenGeneratorONNX,
    GraspGenDiscriminatorONNX,
)

NUM_POINTS = 2048
NUM_GRASPS = 20
SAMPLE_DIM = 9


def _load_manifest(name: str) -> dict:
    path = PROJECT_ROOT / "configs" / "manifests" / f"{name}.json"
    with open(path) as f:
        return json.load(f)


def export_generator(ckpt: str | None, verbose: bool = True) -> Path:
    manifest = _load_manifest("graspgen_generator")
    out_path = PROJECT_ROOT / manifest["onnx_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = GraspGenGeneratorONNX(num_grasps=NUM_GRASPS, grasp_repr="r3_6d")
    if ckpt:
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state.get("model", state))
        if verbose:
            print(f"  Loaded generator weights from {ckpt}")
    model.eval()

    pc = torch.randn(1, NUM_POINTS, 3)
    noisy = torch.randn(NUM_GRASPS, SAMPLE_DIM)
    timestep = torch.tensor([0], dtype=torch.long)

    torch.onnx.export(
        model,
        (pc, noisy, timestep),
        str(out_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["pc", "noisy_grasps", "timestep"],
        output_names=["noise_pred"],
        dynamic_axes=None,
    )
    if verbose:
        print(f"✓ Generator ONNX -> {out_path}")
    return out_path


def export_discriminator(ckpt: str | None, verbose: bool = True) -> Path:
    manifest = _load_manifest("graspgen_discriminator")
    out_path = PROJECT_ROOT / manifest["onnx_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = GraspGenDiscriminatorONNX(num_grasps=NUM_GRASPS, grasp_repr="r3_6d")
    if ckpt:
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state.get("model", state))
        if verbose:
            print(f"  Loaded discriminator weights from {ckpt}")
    model.eval()

    pc = torch.randn(1, NUM_POINTS, 3)
    grasps = torch.randn(1, NUM_GRASPS, SAMPLE_DIM)

    torch.onnx.export(
        model,
        (pc, grasps),
        str(out_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["pc", "grasps"],
        output_names=["scores"],
        dynamic_axes=None,
    )
    if verbose:
        print(f"✓ Discriminator ONNX -> {out_path}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export GraspGen ONNX models")
    parser.add_argument("--generator-ckpt", default=None)
    parser.add_argument("--discriminator-ckpt", default=None)
    parser.add_argument("--only", choices=["generator", "discriminator"], default=None)
    args = parser.parse_args()

    if args.only != "discriminator":
        gen_path = export_generator(args.generator_ckpt)
    if args.only != "generator":
        disc_path = export_discriminator(args.discriminator_ckpt)

    # Verify exported files load
    import onnx

    for p in (PROJECT_ROOT / "models" / "onnx").glob("graspgen_*_pointnet.onnx"):
        onnx.checker.check_model(onnx.load(str(p)))
        print(f"✓ Verified: {p.name}")

    print("\n✅ Export complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
