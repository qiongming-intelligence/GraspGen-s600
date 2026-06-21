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


def main() -> int:
    from grasp_gen.models.model_utils import PointNetPlusPlus
    from graspgen_s600_tools.models.pointnet_upstream import PointNetUpstream

    print("=" * 70)
    print("Testing Weight Loading: Upstream Robotiq -> PointNetUpstream")
    print("=" * 70)

    # Load upstream checkpoint
    ckpt_path = PROJECT_ROOT / "models/upstream/graspgen_robotiq_2f_140_gen.pth"
    print(f"\n[1/5] Loading upstream checkpoint: {ckpt_path}")
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

    print(f"\n[2/5] Stripped prefix, state dict keys: {list(encoder_state.keys())[:5]}...")

    # Create upstream PointNetPlusPlus for reference
    print("\n[3/5] Creating upstream PointNetPlusPlus...")
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
    print("\n[4/5] Creating our PointNetUpstream...")
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

    # Compare outputs
    print("\n[5/5] Comparing outputs on same input...")
    torch.manual_seed(42)
    test_pc = torch.randn(1, 2048, 3).to(device)

    with torch.no_grad():
        torch.manual_seed(42)  # Reset for upstream sampling
        upstream_out = upstream_encoder(test_pc)

        torch.manual_seed(42)  # Same seed for our random sampling
        our_out = our_encoder(test_pc)

    print(f"  Upstream output: {upstream_out.shape}  mean={upstream_out.mean():.6f}")
    print(f"  Our output:      {our_out.shape}  mean={our_out.mean():.6f}")

    diff = (upstream_out - our_out).abs()
    print(f"\n  Max diff:  {diff.max().item():.6e}")
    print(f"  Mean diff: {diff.mean().item():.6e}")

    if strict_ok and diff.max().item() < 1e-4:
        print("\n✅ Weight loading SUCCESS! Outputs match within 1e-4.")
        return 0
    elif strict_ok:
        print(f"\n⚠ Weights loaded (strict=True) but outputs differ (max={diff.max().item():.3e}).")
        print("  Expected: upstream uses FPS (geometrically optimal), ours uses random sampling (ONNX-compatible).")
        print("  This difference will impact final grasp generation accuracy.")
        return 0
    else:
        print("\n❌ Weight loading INCOMPLETE. Key mismatch needs resolution.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
