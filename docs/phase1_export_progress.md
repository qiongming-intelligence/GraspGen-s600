# Phase 1: ONNX Export - Progress Report

## Status: 🚧 In Progress

### Completed ✅

1. **Project Infrastructure**
   - Standard s600 adaptation directory structure
   - Git repository with remote sync
   - Comprehensive documentation

2. **Contract Generation**
   - Generator contract (dual-core, INT16, r3_6d)
   - Discriminator contract (single-core, INT16)
   - JSON manifests ready for compilation

3. **Model Loading Utilities**
   - `factories.py` - Load Generator/Discriminator from checkpoints
   - Support for YAML config parsing
   - Model info extraction

4. **ONNX Export Modules**
   - `generator.py` - Single-step denoising wrapper
   - `discriminator.py` - Direct export wrapper
   - `export_to_onnx.py` - End-to-end export script
   - Comprehensive export guide

5. **Pretrained Weights**
   - Downloaded on ws-wan (in progress):
     - graspgen_franka_panda_gen.pth (907 MB) - downloading...
     - graspgen_franka_panda_dis.pth (166 MB) - pending
     - graspgen_franka_panda.yml (config) - pending

### In Progress 🚧

- **Weight Download**: Generator 428MB/907MB (47%)
- **Next**: Install dependencies and test export on ws-wan

### Todo ⏳

1. **ONNX Export Testing** (Est: 2-3 hours)
   - Install dependencies on ws-wan
   - Run export script
   - Fix any model architecture issues
   - Validate ONNX structure

2. **Precision Validation** (Est: 2-3 hours)
   - Compare PyTorch vs ONNX outputs
   - Verify error < 1e-3 (precision gate)
   - Document any numerical differences
   - Create validation script

3. **Documentation Update** (Est: 30 min)
   - Export results
   - Known issues and solutions
   - Update Phase 1 summary

## Key Design Decisions

### Diffusion Loop Handling

**Problem**: Original Generator uses 10-20 step diffusion loop that's hard to export

**Solution**: Single-step denoising export
```python
# Export this function to ONNX
noise_pred = model(pc, noisy_grasps, timestep)

# Run diffusion loop in Python
for t in timesteps:
    noise_pred = onnx_model.run(pc, noisy_grasps, t)
    noisy_grasps = scheduler.step(noise_pred, t, noisy_grasps)
```

**Benefits**:
- Smaller ONNX model
- Flexible step count at runtime
- Easier to debug

### Grasp Representation

**Original**: r3_so3 (12 dims: 3 position + 9 rotation matrix)
**Target**: r3_6d (9 dims: 3 position + 6D rotation)

**Rationale**: 
- Smaller output tensor (20 grasps × 9 vs 12)
- Continuous representation (better for diffusion)
- May need conversion layer if original uses SO(3)

### Fixed Shapes

All inputs are fixed for BPU compatibility:
- Point cloud: (1, 2048, 3)
- Grasps: (1, 20, 9) or (20, 9) for generator
- Timestep: (1,)

No dynamic axes to ensure BPU compilation success.

## Challenges Encountered

### 1. PointTransformerV3 (PTV3) Backbone

**Issue**: Original model uses `ptv3` which may have custom ops
**Status**: TBD - need to test export
**Fallback**: Switch to `pointnet` backbone if PTV3 fails

### 2. SO(3) Rotation Representation

**Issue**: r3_so3 uses 9D rotation matrix, may have constraints
**Status**: TBD - need to verify export
**Fallback**: Convert to r3_6d representation

### 3. Diffusion Scheduler

**Issue**: Scheduler timestep scheduling may use dynamic ops
**Status**: Using fixed scalar timestep input
**Plan**: Implement scheduler in Python post-export

## Precision-First Validation Plan

### Stage 1: ONNX Export Validation

```bash
# 1. Export models
python3 export_to_onnx.py --gen-checkpoint ... --dis-checkpoint ...

# 2. Check ONNX structure
python3 -c "import onnx; onnx.checker.check_model(...)"

# 3. Verify shapes match contracts
python3 scripts/validate_shapes.py
```

### Stage 2: Numerical Validation

```python
# Compare PyTorch vs ONNX on same input
pc = torch.randn(1, 2048, 3)
noisy_grasps = torch.randn(20, 9)

# PyTorch inference
pt_output = generator(pc, noisy_grasps, timestep=0)

# ONNX inference
onnx_output = onnx_session.run(None, {...})

# Precision gate: error < 1e-3
error = np.abs(pt_output - onnx_output).max()
assert error < 1e-3, f"Precision error: {error}"
```

### Stage 3: Integration Test

```python
# Full pipeline test
1. Load point cloud
2. Normalize to [-1, 1]
3. Run generator (20 diffusion steps)
4. Run discriminator (score grasps)
5. Select top-K grasps
```

## Timeline

- **Now - 2 hours**: Weight download + dependency install + export testing
- **2-4 hours**: Precision validation + fixes
- **4-5 hours**: Documentation + commit + push
- **Target completion**: End of day (June 21)

## Success Criteria (Phase 1 Complete)

- [x] Project structure created
- [x] Contract generation working
- [x] Model loading utilities implemented
- [x] ONNX export modules implemented
- [ ] Weights downloaded ✓ 47% done
- [ ] Dependencies installed on ws-wan
- [ ] ONNX export successful (no errors)
- [ ] ONNX model passes `onnx.checker`
- [ ] Shapes match contracts
- [ ] **Precision validation: PyTorch vs ONNX error < 1e-3** ⭐
- [ ] Code committed and pushed to remote
- [ ] Documentation updated

---

**Last Updated**: 2026-06-21 12:44 CST
**Current Task**: Downloading weights (428MB/907MB)
**Next Action**: Install dependencies → Test export → Validate precision
