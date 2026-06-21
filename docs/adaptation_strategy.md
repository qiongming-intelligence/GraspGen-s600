# GraspGen-s600 Adaptation Strategy

## Core Principle: 🎯 Precision First

**精度优先于速度** - 确保抓取质量和成功率是首要目标。

### Quantization Strategy (Precision-First)

#### Priority 1: Maximum Precision
- **INT16 quantization** for all layers (default)
- **FP32 output layers** to preserve final precision
- **Max calibration** with comprehensive real-world data (>100 samples)
- No aggressive optimizations until baseline precision validated

#### Precision Gates (Must Pass Before Optimization)
```
✓ Grasp success rate: > 85% (match PyTorch baseline)
✓ Position error: < 5mm
✓ Rotation error: < 5°
✓ BPU vs CPU output diff: < 1e-3
```

#### Fallback Strategy
If INT16 shows precision degradation:
1. **Try FP16 mixed precision** (critical layers in FP16)
2. **Identify sensitive layers** via profiling
3. **Per-layer precision tuning**
4. Only move to INT8 after explicit validation

### Compilation Settings (Precision-First)

```yaml
# High-precision compilation profile
optimize_level: O2          # Balanced optimization
core_num: 2                 # Generator (dual-core for speed)
core_num: 1                 # Discriminator (single-core sufficient)
calibration_type: max       # Use maximum activation values
optimization: set_all_nodes_int16  # INT16 default

# Precision-critical settings
preserve_fp32_nodes:
  - "diffusion_head.output"  # Final grasp prediction
  - "discriminator.output"   # Quality scores
  
calibration_samples: 100+   # Comprehensive calibration
```

### Development Phases (Precision-First)

#### Phase 1: Basic Adaptation ✅ (Current)
- [x] Project structure
- [x] Contract generation
- [x] Upstream integration
- [ ] ONNX export
- [ ] **Validation: ONNX vs PyTorch precision < 1e-3**

#### Phase 2: Precision Validation (Week 3-4)
Focus: Establish precision baseline before any optimization
- [ ] Collect diverse calibration data (100+ objects)
- [ ] INT16 HBM compilation
- [ ] **Gate 1: BPU vs CPU grasp success rate (> 85%)**
- [ ] **Gate 2: Position/rotation error validation**
- [ ] If failed: fallback to FP16 mixed precision

#### Phase 3: Performance Optimization (Week 5-6)
Only after precision gates passed:
- [ ] Dual-core compilation (Generator)
- [ ] Diffusion steps tuning (20 → 10 steps, validate precision)
- [ ] Point cloud downsampling (validate precision first)
- [ ] Target: 20+ FPS while maintaining precision

#### Phase 4: Deployment (Week 7-8)
- [ ] C++ runtime (optional, if Python meets requirements)
- [ ] Real robot validation
- [ ] Documentation

### Reference: Precision-First Success Cases

**FoundationPose-s600:**
- Used INT16 across the board
- Maintained pose estimation accuracy
- Achieved real-time performance (40-50ms)

**Lesson:** Don't rush to INT8 or aggressive optimizations. INT16 provides good balance.

### Validation Checklist

Before declaring any phase complete:

**ONNX Export:**
- [ ] `onnx.checker.check_model()` passes
- [ ] All shapes match contracts
- [ ] PyTorch vs ONNX numerical diff < 1e-3

**HBM Compilation:**
- [ ] `hrt_model_exec model_info` loads successfully
- [ ] No unexpected CPU fallback ops
- [ ] BPU utilization > 80%

**Precision Validation:**
- [ ] 100+ diverse test samples
- [ ] Grasp success rate > 85%
- [ ] Position error < 5mm
- [ ] Rotation error < 5°

**Performance (Secondary):**
- [ ] Generator: < 30ms
- [ ] Discriminator: < 20ms
- [ ] Total pipeline: < 50ms (20 FPS)

### Decision Tree: When Precision Issues Arise

```
Precision issue detected
    ↓
Is it in ONNX export?
    YES → Fix PyTorch model wrapper, ensure numerical stability
    NO → Continue
    ↓
Is it in HBM compilation?
    YES → Check calibration data quality
         → Try FP16 mixed precision
         → Profile sensitive layers
    NO → Continue
    ↓
Is it specific to certain objects/poses?
    YES → Add more calibration samples
         → Analyze failure cases
    NO → Consider model architecture changes

NEVER compromise on precision gates to meet speed targets.
```

### Success Criteria

**Phase 1 Complete:**
- ONNX export with < 1e-3 error
- Ready for HBM compilation

**Phase 2 Complete:**
- BPU inference working
- Precision gates passed (> 85% success rate)
- Documented precision validation results

**Phase 3 Complete:**
- 20+ FPS achieved
- Precision maintained
- Benchmarks documented

**Phase 4 Complete:**
- Deployed on real robot
- Production-ready documentation

---

**Last Updated:** 2026-06-21  
**Status:** Phase 1 initialization complete, ready for ONNX export  
**Next Milestone:** ONNX export with validated precision
