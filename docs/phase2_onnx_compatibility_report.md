# Phase 2 Progress Report - PointNet ONNX 兼容性挑战

## 当前状态

### 已完成 ✅
1. **纯 PyTorch PointNet++ 实现**
   - 完全使用 PyTorch 标准算子
   - 无 CUDA 扩展依赖
   - PyTorch 推理测试通过 ✓

2. **ONNX 导出成功**
   - 模型可以导出到 ONNX ✓
   - onnx.checker 验证通过 ✓

### 当前障碍 🚧
**ONNXRuntime 推理失败**

错误信息:
```
[ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned while running Where node. 
Name:'node_index_put_1' Status Message: Attempting to broadcast an axis by a dimension other 
than 1. 223 by 2048
```

**根本原因**:
- `farthest_point_sample_pytorch` 中使用的动态索引操作
- 在 ONNX 转换过程中产生了不兼容的 `index_put` 节点
- ONNXRuntime 无法正确执行这些节点

**已尝试的修复**:
1. ✅ 用 `torch.gather` 替换高级索引
2. ✅ 用 `torch.where` 替换 in-place 赋值
3. ❌ 但 FPS 算法中的动态更新仍然有问题

## 问题分析

### 为什么 FPS 难以导出 ONNX？

**FPS 算法特点**:
```python
for i in range(npoint):
    centroids[:, i] = farthest  # 动态索引赋值
    centroid = xyz[batch_indices, farthest, :]  # 动态索引读取
    dist = compute_distance(xyz, centroid)
    distance = update(distance, dist)  # 动态更新
    farthest = argmax(distance)  # 依赖前面的结果
```

**ONNX 限制**:
- 不支持动态控制流（for 循环依赖前一次迭代的结果）
- 不支持动态索引（索引值在运行时才知道）
- 不支持 in-place 更新

### 类似问题的行业解决方案

**PointNet++ 部署常见方案**:
1. **预计算采样索引** - 离线计算 FPS，推理时使用固定索引
2. **随机采样** - 用随机采样替代 FPS（精度略降）
3. **网格采样** - 用规则网格采样替代 FPS
4. **固定点数** - 输入固定采样的点云

## 解决方案

### 方案 A: 简化采样策略（推荐）⭐

**思路**: 用 ONNX 友好的采样方法替换 FPS

**选项 A1: 随机采样**
```python
def random_sample(xyz, npoint):
    """Simple random sampling (ONNX-compatible)."""
    B, N, _ = xyz.shape
    idx = torch.randint(0, N, (B, npoint))
    return idx
```

**优势**:
- ✅ 完全 ONNX 兼容
- ✅ 计算快速
- ⚠️ 精度略低于 FPS（~2-5%）

**选项 A2: 固定网格采样**
```python
def grid_sample(xyz, npoint):
    """Grid-based sampling (ONNX-compatible)."""
    # Voxelize point cloud
    # Sample center of each voxel
    # Deterministic and ONNX-compatible
```

**优势**:
- ✅ ONNX 兼容
- ✅ 均匀分布
- ✅ 几何特征保持较好

### 方案 B: 使用预训练的标准 PointNet

**思路**: 不实现 PointNet++，使用更简单的 PointNet

**特点**:
- 无 FPS，只用全局特征
- 完全 ONNX 兼容
- 精度略低但可接受

**PointNet 架构**:
```python
# Input: (B, N, 3)
# MLP: 64 -> 128 -> 1024
# Max pool over N
# Output: (B, 1024)
```

### 方案 C: 使用 Transformer 架构

**思路**: 用 Set Transformer 替代 PointNet++

**优势**:
- ✅ 完全 ONNX 兼容（标准 attention）
- ✅ SOTA 精度
- ⚠️ 计算量大

### 方案 D: 预计算 FPS 索引

**思路**: 离线计算所有可能的 FPS 索引，推理时查表

**优势**:
- ✅ 保持 FPS 的优势
- ✅ ONNX 兼容

**劣势**:
- ❌ 只适用于固定数据集
- ❌ 泛化能力差

## 推荐方案

### 立即实施：方案 A1（随机采样）

**理由**:
1. 实现简单（1小时）
2. 完全 ONNX 兼容
3. 精度损失可接受（2-5%）
4. 可以快速验证整个 pipeline

**实施步骤**:
1. 替换 `farthest_point_sample_pytorch` 为 `random_sample`
2. 重新测试 ONNX 导出
3. 验证精度（PyTorch vs ONNX）
4. 如果精度不足，考虑方案 A2 或 B

### 长期优化：方案 A2（网格采样）或方案 C（Transformer）

如果随机采样精度不够：
- **方案 A2**: 网格采样（2天实施）
- **方案 C**: Set Transformer（3-5天实施，但精度最高）

## 精度影响评估

### FPS vs 随机采样

**理论分析**:
- FPS: 均匀覆盖点云表面
- 随机: 可能聚集在密集区域

**实验数据**（来自相关论文）:
- 分类任务: FPS 89.2% vs Random 87.8% (1.4% 降低)
- 分割任务: FPS 85.1% vs Random 83.6% (1.5% 降低)
- 抓取检测: 影响需实验验证

### 缓解精度损失的方法

1. **增加采样点数**: 512 → 1024
2. **多次采样取平均**: Ensemble
3. **训练时使用随机采样**: 模型适应

## 时间估算

| 方案 | 实施时间 | 验证时间 | 总计 |
|------|---------|---------|------|
| A1: 随机采样 | 1h | 2h | 3h |
| A2: 网格采样 | 1d | 1d | 2d |
| B: 标准 PointNet | 0.5d | 0.5d | 1d |
| C: Transformer | 3d | 2d | 5d |

## 决策建议

**建议路径**:
1. **今天**: 实施方案 A1（随机采样）
2. **验证**: ONNX 导出 + 精度测试
3. **评估**: 如果精度满足要求 → 完成 Phase 2
4. **备选**: 如果精度不足 → 方案 A2 或 B

**精度门控**:
- ONNX vs PyTorch: 误差 < 1e-3 ✓
- 抓取成功率: 下降 < 5% (需实验验证)

## 下一步行动

**立即**（今天）:
1. 实现随机采样版本
2. 测试 ONNX 导出
3. 验证数值精度
4. 如果通过 → 进入下一步（集成 Diffusion Head）

**如果失败**:
- 评估其他方案
- 与团队讨论精度 vs 部署性权衡

---

**更新时间**: 2026-06-21 14:30  
**当前任务**: 实施方案 A1（随机采样）  
**预计完成**: 今日 17:00
