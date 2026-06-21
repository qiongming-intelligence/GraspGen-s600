#!/usr/bin/env python3
"""
GPU baseline inference for the upstream GraspGen model (Robotiq 2F-140, pointnet
backbone). This runs the ORIGINAL NVlabs model on real CUDA (spconv-free pointnet
path + pointnet2_ops FPS), establishing a reference for the S600 ONNX port.

We only need the generator to produce grasp poses; the discriminator confidence
is reported but not required. Use --no-disc-threshold to keep all grasps.

Usage (on ws-wan, in venv with pointnet2_ops built):
  python src/python/scripts/gpu_infer_baseline.py \
      --gripper-config models/upstream/graspgen_robotiq_2f_140.yml \
      --num-grasps 200
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
GRASPGEN = PROJECT_ROOT / "third_party" / "GraspGen"
sys.path.insert(0, str(GRASPGEN))


def make_synthetic_object_pc(n: int = 2048, seed: int = 0) -> np.ndarray:
    """A simple box surface point cloud (meters), centered near origin."""
    rng = np.random.default_rng(seed)
    # Box half-extents (m): a small graspable object ~8x6x10 cm
    hx, hy, hz = 0.04, 0.03, 0.05
    pts = []
    per_face = n // 6
    for axis in range(3):
        for sign in (-1.0, 1.0):
            uv = rng.uniform(-1.0, 1.0, size=(per_face, 2))
            face = np.zeros((per_face, 3))
            other = [i for i in range(3) if i != axis]
            ext = [hx, hy, hz]
            face[:, axis] = sign * ext[axis]
            face[:, other[0]] = uv[:, 0] * ext[other[0]]
            face[:, other[1]] = uv[:, 1] * ext[other[1]]
            pts.append(face)
    pc = np.concatenate(pts, axis=0)
    # pad/truncate to exactly n
    if len(pc) < n:
        pad = pc[rng.integers(0, len(pc), n - len(pc))]
        pc = np.concatenate([pc, pad], axis=0)
    return pc[:n].astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gripper-config",
        default=str(PROJECT_ROOT / "models/upstream/graspgen_robotiq_2f_140.yml"),
    )
    parser.add_argument("--num-grasps", type=int, default=200)
    parser.add_argument("--pc-npy", default=None, help="optional (N,3) .npy point cloud")
    parser.add_argument("--save", default=None, help="optional path to save grasps .npy")
    args = parser.parse_args()

    from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg

    print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
    print(f"Loading config: {args.gripper_config}")
    cfg = load_grasp_cfg(args.gripper_config)
    print(f"  backbone={cfg.diffusion.obs_backbone}  grasp_repr={cfg.diffusion.grasp_repr}"
          f"  iters_eval={cfg.diffusion.num_diffusion_iters_eval}")

    t0 = time.time()
    sampler = GraspGenSampler(cfg)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    if args.pc_npy:
        pc = np.load(args.pc_npy).astype(np.float32)
    else:
        pc = make_synthetic_object_pc(2048)
        print("Using synthetic box point cloud (2048 pts)")
    print(f"Point cloud: {pc.shape}  bounds={pc.min(0)} .. {pc.max(0)}")

    # Warmup + timed inference
    torch.cuda.synchronize()
    t0 = time.time()
    grasps, conf = GraspGenSampler.run_inference(
        pc,
        sampler,
        grasp_threshold=-1.0,      # keep top-k by confidence, no hard cutoff
        num_grasps=args.num_grasps,
        remove_outliers=False,
    )
    torch.cuda.synchronize()
    dt = time.time() - t0

    grasps = grasps.detach().cpu().numpy() if torch.is_tensor(grasps) else np.array(grasps)
    conf = conf.detach().cpu().numpy() if torch.is_tensor(conf) else np.array(conf)

    print("\n==================== RESULT ====================")
    print(f"  grasps: {grasps.shape}  (expect (M,4,4) homogeneous)")
    print(f"  conf:   {conf.shape}  range=[{conf.min():.3f}, {conf.max():.3f}]"
          if conf.size else "  conf: (empty)")
    print(f"  inference wall time: {dt:.2f}s for up to {args.num_grasps} grasps")
    if grasps.size:
        print(f"  example grasp[0] translation: {grasps[0, :3, 3]}")
    print("================================================")

    if args.save and grasps.size:
        np.save(args.save, grasps)
        print(f"Saved grasps -> {args.save}")

    return 0 if grasps.size else 1


if __name__ == "__main__":
    sys.exit(main())
