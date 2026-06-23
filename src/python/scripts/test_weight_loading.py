#!/usr/bin/env python3
"""
Test loading upstream Robotiq generator weights into PointNetUpstream encoder.

This script:
1. Loads the upstream generator checkpoint
2. Extracts the object_encoder state dict
3. Loads it into our PointNetUpstream (with key remapping if needed)
4. Runs both upstream and our encoder on the same input
5. Compares outputs to verify weight compatibility

Usage (on ws-wan):
  python src/python/scripts/test_weight_loading.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
GRASPGEN = PROJECT_ROOT / "third_party" / "GraspGen"
sys.path.insert(0, str(GRASPGEN))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "python"))


def compare_fps_indices(upstream_encoder, test_pc, device) -> bool:
    """Compare CUDA FPS indices from upstream SA layers against pure-PyTorch FPS."""
    from grasp_gen.models.pointnet.pointnet2_utils import furthest_point_sample
    from graspgen_s600_tools.models.pointnet_upstream import fps_faithful

    print("\n[5/6] Comparing FPS indices layer-by-layer...")
    if device.type != "cuda":
        print("  ⚠ Skipping FPS index comparison: upstream FPS requires CUDA")
        return False

    with torch.no_grad():
        xyz = test_pc
        features = None
        all_match = True

        for layer_idx, sa_module in enumerate(upstream_encoder.obj_SA_modules):
            if sa_module.npoint is None:
                print(f"  SA{layer_idx + 1}: group_all, no FPS")
                break

            cuda_idx = furthest_point_sample(xyz.contiguous(), sa_module.npoint)
            torch_idx = fps_faithful(xyz, sa_module.npoint)
            mismatch = (cuda_idx != torch_idx).sum().item()
            total = cuda_idx.numel()
            print(
                f"  SA{layer_idx + 1}: {total - mismatch}/{total} indices match "
                f"({mismatch} mismatch)"
            )
            if mismatch:
                first_bad = (cuda_idx != torch_idx).nonzero(as_tuple=False)[0]
                b, j = first_bad.tolist()
                print(
                    f"    first mismatch at batch={b}, step={j}: "
                    f"cuda={cuda_idx[b, j].item()}, torch={torch_idx[b, j].item()}"
                )
                all_match = False

            xyz, _, features, _ = sa_module(xyz, features)

    return all_match


def main() -> int:
    from grasp_gen.models.model_utils import PointNetPlusPlus
    from graspgen_s600_tools.models.pointnet_upstream import PointNetUpstream

    print("=" * 70)
    print("Testing Weight Loading: Upstream Robotiq -> PointNetUpstream")
    print("=" * 70)

    # Load upstream checkpoint
    ckpt_path = PROJECT_ROOT / "models/upstream/graspgen_robotiq_2f_140_gen.pth"
    print(f"\n[1/6] Loading upstream checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"  Top-level keys: {list(ckpt.keys())}")

    # The state dict is nested under 'model'
    if 'model' in ckpt:
        state_dict = ckpt['model']
        print(f"  Found 'model' key, has {len(state_dict)} keys")
    else:
        state_dict = ckpt
        print(f"  Using checkpoint directly as state dict")

    # Extract object_encoder weights
    encoder_keys = [k for k in state_dict.keys() if k.startswith("object_encoder.")]
    print(f"  Found {len(encoder_keys)} object_encoder keys")
    if encoder_keys:
        print(f"  Example keys: {encoder_keys[:3]}")
    else:
        # Debug: show what prefixes exist
        prefixes = set(k.split('.')[0] for k in state_dict.keys())
        print(f"  Available top-level prefixes: {sorted(prefixes)[:10]}")

    # Strip prefix
    encoder_state = {}
    prefix = "object_encoder."
    for k in encoder_keys:
        new_k = k[len(prefix):]
        encoder_state[new_k] = state_dict[k]

    print(f"\n[2/6] Stripped prefix, state dict keys: {list(encoder_state.keys())[:5]}...")

    # Create upstream PointNetPlusPlus for reference
    print("\n[3/6] Creating upstream PointNetPlusPlus...")
    upstream_encoder = PointNetPlusPlus(output_embedding_dim=512, feature_dim=-1)
    upstream_encoder.eval()

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    upstream_encoder = upstream_encoder.to(device)

    # Load into upstream (should work directly)
    try:
        upstream_encoder.load_state_dict(encoder_state, strict=True)
        print("  ✓ Loaded into upstream PointNetPlusPlus (reference)")
    except Exception as e:
        print(f"  ✗ Failed to load into upstream: {e}")
        return 1

    # Create our PointNetUpstream
    print("\n[4/6] Creating our PointNetUpstream...")
    our_encoder = PointNetUpstream(output_embedding_dim=512, feature_dim=-1)
    our_encoder.eval()
    our_encoder = our_encoder.to(device)

    print("  Our keys:", list(our_encoder.state_dict().keys())[:8])
    print("  Checkpoint keys:", list(encoder_state.keys())[:8])

    # Attempt to load (may need key remapping)
    try:
        our_encoder.load_state_dict(encoder_state, strict=True)
        print("  ✓ Loaded into PointNetUpstream (strict=True)")
        strict_ok = True
    except Exception as e:
        print(f"  ⚠ Strict load failed: {e}")
        print("  Attempting partial load...")
        missing, unexpected = our_encoder.load_state_dict(encoder_state, strict=False)
        print(f"    Missing: {len(missing)} keys")
        print(f"    Unexpected: {len(unexpected)} keys")
        if missing:
            print(f"    Missing keys (first 5): {missing[:5]}")
        if unexpected:
            print(f"    Unexpected keys (first 5): {unexpected[:5]}")
        strict_ok = False

    torch.manual_seed(42)
    test_pc = torch.randn(1, 2048, 3).to(device)

    fps_match = compare_fps_indices(upstream_encoder, test_pc, device)

    # Compare outputs
    print("\n[6/6] Comparing encoder outputs on same input...")
    with torch.no_grad():
        upstream_out = upstream_encoder(test_pc)
        our_out = our_encoder(test_pc)

    print(f"  Upstream output: {upstream_out.shape}  mean={upstream_out.mean():.6f}")
    print(f"  Our output:      {our_out.shape}  mean={our_out.mean():.6f}")

    diff = (upstream_out - our_out).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"\n  Max diff:  {max_diff:.6e}")
    print(f"  Mean diff: {mean_diff:.6e}")

    if strict_ok and fps_match and max_diff < 1e-4:
        print("\n✅ Weight loading + faithful FPS SUCCESS! Outputs match within 1e-4.")
        return 0
    elif strict_ok and max_diff < 1e-4:
        print("\n✅ Weight loading SUCCESS! Outputs match within 1e-4.")
        print("  ⚠ FPS index comparison was skipped or not fully matched.")
        return 0
    elif strict_ok:
        print(f"\n⚠ Weights loaded (strict=True) but outputs differ (max={max_diff:.3e}).")
        print("  This indicates a remaining sampling/grouping semantic mismatch.")
        return 1
    else:
        print("\n❌ Weight loading INCOMPLETE. Key mismatch needs resolution.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
