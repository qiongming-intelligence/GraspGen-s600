# Phase 2: Weight Loading Results

## Summary

Successfully created a weight-compatible PointNet++ encoder (`PointNetUpstream`) that can load upstream Robotiq pretrained weights with `strict=True`.

**Date**: 2025-06-21  
**Status**: ✅ Weight loading verified  
**Device**: ws-wan RTX 4090 (CUDA)

## Architecture Match

### Upstream Structure (grasp_gen/models/model_utils.py)
```python
class PointNetPlusPlus:
    - obj_SA_modules: ModuleList[PointnetSAModule]
        - PointnetSAModule.mlps: ModuleList[Sequential]
            - Sequential: [Conv2d(bias=False), BN, ReLU, ...]
    - prediction_head: Linear(512→1024) → ReLU → Linear(1024→1024) → ReLU → Linear(1024→512)
```

### Our Implementation (graspgen_s600_tools/models/pointnet_upstream.py)
```python
class PointNetUpstream:
    - obj_SA_modules: ModuleList[PointNetSetAbstraction]
        - PointNetSetAbstraction.mlps: ModuleList[Sequential]
            - Sequential: [Conv2d(bias=False), BN, ReLU, ...]
    - prediction_head: (same as upstream)
```

**Key differences**:
- **Sampling**: Upstream uses CUDA FPS, ours uses random sampling (ONNX-compatible)
- **State dict keys**: Identical structure, strict loading succeeds

## Test Results

### Weight Loading
```
Checkpoint: models/upstream/graspgen_robotiq_2f_140_gen.pth
State dict: checkpoint['model']['object_encoder.*']
Parameters: 42 keys loaded with strict=True
Result: ✅ SUCCESS
```

### Output Comparison (same input, same weights, seed=42)
```
Input: (1, 2048, 3) random point cloud
Upstream (FPS):     mean=-0.059028
Ours (random):      mean=-0.058584
Max diff:           3.590e+00
Mean diff:          4.436e-01
```

## Impact Analysis

### Sampling Strategy Impact

| Metric | Upstream (FPS) | Ours (Random) | 
|--------|----------------|---------------|
| Sampling | Geometrically optimal | Uniform random |
| ONNX export | ❌ Not supported | ✅ Supported |
| Point distribution | Maximally spread | Random coverage |
| Encoder output diff | baseline | ~0.44 mean, ~3.59 max |

**Expected consequences**:
1. ✅ Full ONNX exportability achieved
2. ⚠️ Encoder feature quality degraded (random < FPS)
3. ⚠️ Final grasp generation accuracy will be lower
4. 📊 Accuracy impact must be measured end-to-end

## Next Steps

1. **Integrate into full generator**: Replace `object_encoder` in generator with `PointNetUpstream`
2. **Export to ONNX**: Full generator pipeline with random sampling
3. **Accuracy measurement**: Compare ONNX output vs GPU baseline on test data
4. **Quantify FPS impact**: Run experiments to measure grasp success rate degradation

## Files Modified

### Created
- `src/python/graspgen_s600_tools/models/pointnet_upstream.py` - Weight-compatible encoder
- `src/python/scripts/test_weight_loading.py` - Weight loading verification script

### Architecture Details

**Hyperparameters (matching upstream)**:
- OBJ_NPOINTS = [256, 64, None]
- OBJ_RADII = [0.02, 0.04, None]
- OBJ_NSAMPLES = [64, 128, None]
- OBJ_MLPS = [[0,64,128], [128,128,256], [256,256,512]]
- output_embedding_dim = 512
- feature_dim = -1 (xyz-only, no additional features)

**Key Implementation Choices**:
1. `use_xyz=True` in all SA layers (concat xyz to features)
2. `use_random_sample=True` (ONNX-compatible)
3. Conv2d `bias=False` (upstream uses bias only when norm="")
4. Forward signature: `forward(pc)` not `forward(xyz, features)` to match upstream

## References

- Upstream checkpoint: `models/upstream/graspgen_robotiq_2f_140_gen.pth`
- GPU baseline: `src/python/scripts/gpu_infer_baseline.py`
- Test data: `test_data/box_seed42_2048pts.npy`
- Baseline output: `test_data/gpu_baseline_grasps.npy`
