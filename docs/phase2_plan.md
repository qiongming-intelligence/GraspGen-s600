# Phase 2 实施计划 - 架构适配与 ONNX 导出

## 目标

将 GraspGen 模型适配为 ONNX 兼容版本，实现：
1. 替换 PTV3 backbone 为纯 PyTorch PointNet
2. 成功导出 ONNX 模型
3. 通过精度验证（PyTorch vs ONNX 误差 < 1e-3）

## 策略选择

基于 Phase 1 分析，采用**方案 D**（直接替换 backbone）：
- 使用纯 PyTorch PointNet++ 实现（无 CUDA 扩展）
- 保持 Diffusion Head 架构不变
- 从头训练或使用预训练 encoder

**理由**：
- ✅ 完全兼容 ONNX 导出
- ✅ 无复杂依赖
- ⚠️ 需要训练/微调（可接受的时间成本）

## Phase 2 任务分解

### Task 1: 实现纯 PyTorch PointNet++（2天）

#### 1.1 调研和选择实现
**选项**：
- PyTorch3D PointNet
- 第三方纯 PyTorch 实现
- 自己实现简化版

**决策标准**：
- ONNX 兼容性
- 性能和精度
- 维护性

#### 1.2 集成到项目
```python
# 目标架构
src/python/graspgen_s600_tools/
├── models/
│   ├── pointnet_encoder.py    # 纯 PyTorch PointNet++
│   ├── diffusion_head.py       # 从上游提取
│   └── graspgen_onnx.py        # 完整的 ONNX 友好模型
```

#### 1.3 单元测试
- 输入输出形状验证
- ONNX 导出测试
- 数值稳定性测试

### Task 2: 模型训练/迁移学习（2-3天）

#### 2.1 准备数据
**数据源**：
- ACRONYM 数据集（Franka Panda 子集）
- 或使用合成数据快速验证

**数据格式**：
```python
{
    'point_cloud': (N, 2048, 3),  # 归一化到 [-1, 1]
    'grasps': (N, K, 7),           # position + quaternion
    'scores': (N, K),              # 抓取质量分数
}
```

#### 2.2 训练策略
**方案 A：从头训练**
- 训练 PointNet encoder + Diffusion head
- 需要大量数据和时间
- 精度最可控

**方案 B：迁移学习（推荐）**
- 使用预训练 PointNet encoder
- 仅训练 Diffusion head
- 快速收敛

**方案 C：知识蒸馏**
- PTV3 模型作为 teacher
- PointNet 模型作为 student
- 保持精度的同时简化架构

#### 2.3 训练配置
```yaml
model:
  encoder: pointnet_msg  # Multi-scale grouping
  num_points: 2048
  num_grasps: 20
  diffusion_steps: 20

training:
  batch_size: 16
  epochs: 50
  learning_rate: 1e-4
  optimizer: AdamW
  scheduler: cosine

precision_target:
  grasp_success_rate: > 80%  # 初始目标
  position_error: < 10mm
  rotation_error: < 10°
```

### Task 3: ONNX 导出和验证（1天）

#### 3.1 导出流程
```python
# 1. 加载训练好的模型
model = GraspGenONNX.load_from_checkpoint(...)

# 2. 单步去噪导出
export_generator_onnx(
    model.generator,
    'models/onnx/graspgen_generator_pointnet.onnx',
    num_points=2048,
    num_grasps=20,
)

# 3. Discriminator 导出
export_discriminator_onnx(
    model.discriminator,
    'models/onnx/graspgen_discriminator_pointnet.onnx',
    num_points=2048,
    num_candidates=20,
)
```

#### 3.2 精度验证（关键！）
```python
# PyTorch vs ONNX 对比
pc = torch.randn(1, 2048, 3)
noisy_grasps = torch.randn(20, 9)
timestep = torch.tensor([5])

# PyTorch
pt_output = pytorch_model(pc, noisy_grasps, timestep)

# ONNX
onnx_output = onnx_session.run(None, {
    'pc': pc.numpy(),
    'noisy_grasps': noisy_grasps.numpy(),
    'timestep': timestep.numpy(),
})

# 精度门控
error = np.abs(pt_output.numpy() - onnx_output[0]).max()
assert error < 1e-3, f"Precision gate failed: {error}"
```

#### 3.3 形状验证
```python
# 验证与 contract 一致
generator_contract = load_contract('configs/manifests/graspgen_generator.json')
assert onnx_model.inputs[0].shape == generator_contract['input_shapes']['pc']
```

### Task 4: 性能基准测试（1天）

#### 4.1 推理性能
```python
# 测试项
- Single-step denoising: < 50ms (CPU)
- Full diffusion (20 steps): < 1s (CPU)
- Discriminator: < 30ms (CPU)
```

#### 4.2 精度基准
```python
# 在测试集上评估
- Grasp success rate (simulation)
- Position/rotation accuracy
- vs PTV3 baseline (如果可用)
```

#### 4.3 ONNX 优化
```python
# 如果性能不足
- onnxruntime 优化
- 算子融合
- 量化感知训练（为 Phase 3 准备）
```

## 里程碑和交付物

### Milestone 1: PointNet 实现（Day 1-2）
**交付物**：
- [ ] `models/pointnet_encoder.py` 实现
- [ ] 单元测试通过
- [ ] ONNX 导出测试通过
- [ ] 文档更新

**验收标准**：
- 代码可运行
- ONNX 导出无错误
- 形状正确

### Milestone 2: 模型训练（Day 3-5）
**交付物**：
- [ ] 训练脚本和配置
- [ ] 训练好的 checkpoint
- [ ] 训练日志和指标
- [ ] 验证报告

**验收标准**：
- 训练收敛
- 抓取成功率 > 80%（初始目标）
- 损失曲线正常

### Milestone 3: ONNX 导出（Day 6）
**交付物**：
- [ ] Generator ONNX 模型
- [ ] Discriminator ONNX 模型
- [ ] 精度验证报告
- [ ] 导出脚本更新

**验收标准**：
- ✅ ONNX 导出成功
- ✅ onnx.checker 验证通过
- ✅ 精度误差 < 1e-3 ⭐
- ✅ 形状与 contract 一致

### Milestone 4: 性能验证（Day 7）
**交付物**：
- [ ] 性能基准测试报告
- [ ] 精度对比报告
- [ ] Phase 2 总结文档
- [ ] 代码和文档 commit

**验收标准**：
- 推理速度满足要求
- 精度达到目标
- 所有文档完整

## 精度优先检查点

在每个关键步骤后验证精度：

1. **PointNet 实现后**：
   - 验证输出形状和数值范围
   - 与参考实现对比（如有）

2. **训练过程中**：
   - 监控验证集精度
   - 与 baseline 对比（如有）
   - Early stopping 避免过拟合

3. **ONNX 导出后**：
   - PyTorch vs ONNX 误差 < 1e-3 ⭐
   - 多个测试用例验证
   - 边界情况测试

4. **最终验证**：
   - 完整 pipeline 测试
   - 实际抓取场景模拟
   - 性能和精度权衡分析

## 风险和缓解

### 风险 1: PointNet 精度不足
**概率**: 中  
**影响**: 高  
**缓解**:
- 使用 PointNet++ (multi-scale)
- 增加网络容量
- 知识蒸馏从 PTV3

### 风险 2: 训练数据不足
**概率**: 中  
**影响**: 高  
**缓解**:
- 使用数据增强
- 合成数据生成
- 迁移学习

### 风险 3: ONNX 算子不支持
**概率**: 低  
**影响**: 高  
**缓解**:
- 提前测试 ONNX 兼容性
- 避免动态形状和控制流
- 使用标准 PyTorch 算子

### 风险 4: 精度-性能权衡
**概率**: 中  
**影响**: 中  
**缓解**:
- 分步优化（先精度后性能）
- 量化感知训练
- 架构搜索

## 成功标准

Phase 2 完成标准：

- [ ] ✅ 纯 PyTorch PointNet 实现完成
- [ ] ✅ 模型训练收敛
- [ ] ✅ ONNX 导出成功
- [ ] ✅ **精度验证通过（误差 < 1e-3）** ⭐
- [ ] ✅ 形状与 contract 一致
- [ ] ✅ 性能满足初步要求
- [ ] ✅ 所有代码和文档推送到远程
- [ ] ✅ Phase 2 总结报告完成

## 时间估算

| 任务 | 预计时间 | 依赖 |
|------|---------|------|
| PointNet 实现 | 2 天 | - |
| 模型训练 | 2-3 天 | PointNet |
| ONNX 导出 | 1 天 | 训练 |
| 性能验证 | 1 天 | ONNX |
| **总计** | **6-7 天** | |

## 下一步行动

### 立即开始（今天）

1. **调研 PointNet 实现**
   - 评估 PyTorch3D、pointnet2_pytorch 等
   - 选择最适合的实现

2. **创建开发分支**
   ```bash
   git checkout -b phase2-pointnet-adaptation
   ```

3. **搭建代码框架**
   - 创建 `models/` 目录
   - 实现基础 PointNet encoder

4. **设置训练环境**
   - 准备数据集（或合成数据）
   - 配置训练脚本

---

**创建时间**: 2026-06-21  
**目标完成**: 2026-06-28  
**当前状态**: 准备开始
