# Phase 3 最终报告 - 完整模型集成与 ONNX 导出

## 项目概述

**项目**: GraspGen 到 Horizon S600 的适配
**阶段**: Phase 3 - Diffusion Head 集成 + 完整 Generator/Discriminator ONNX 导出
**日期**: 2026-06-21
**状态**: ✅ 完成

---

## Phase 3 目标

在 Phase 2（ONNX 兼容 PointNet++ encoder）基础上，组装完整的可部署模型：

1. ✅ 移植 Diffusion Head（噪声预测网络）为 ONNX 兼容模块
2. ✅ 集成完整 Generator（PointNet + Diffusion Head）
3. ✅ 集成完整 Discriminator（PointNet + 评分头）
4. ✅ 导出 ONNX 并通过精度门控（误差 < 1e-3）
5. ✅ 实现 Python 端 DDPM 去噪循环

---

## 完成情况

| 目标 | 状态 | 精度 |
|------|------|------|
| DiffusionHead 移植 | ✅ | - |
| Generator 集成 | ✅ | 误差 0.00000004 |
| Discriminator 集成 | ✅ | 误差 0.00000003 |
| ONNX 导出 | ✅ | 两模型均通过 onnx.checker |
| DDPM Python 循环 | ✅ | 20 步去噪，输出有限 |

### 验证输出

```
======================================================================
Generator: export + precision check (single denoising step)
======================================================================
  PyTorch noise_pred: (20, 9)
  Exported: /tmp/...onnx
  onnx.checker: OK
  ONNXRuntime noise_pred: (20, 9)
  Max error: 0.00000004
  ✅ PASS (gate < 0.001)

======================================================================
Discriminator: export + precision check
======================================================================
  PyTorch scores: (1, 20)  range=[0.482, 0.490]
  onnx.checker: OK
  ONNXRuntime scores: (1, 20)
  Max error: 0.00000003
  ✅ PASS (gate < 0.001)

======================================================================
DDPM sampling loop (Python-side orchestration)
======================================================================
  Ran 20 denoising steps
  Final grasps: (20, 9)  finite=True
  ✅ PASS

✅ Phase 3 validation passed!
```

---

## 架构设计

### 关键设计决策：单步去噪 + Python 循环

**问题**: 扩散模型的反向去噪是一个迭代循环（每步依赖前一步结果）。
ONNX/BPU 不支持数据依赖的动态控制流。

**解决方案**: 将**单步去噪**导出为静态 ONNX 图，迭代循环在 Python 端编排。

```
        ┌─────────────────────────────────────────┐
        │  Python DDPM 循环 (runtime)               │
        │                                           │
        │   noisy_grasps = randn(K, 9)              │
        │   for k in timesteps:                     │
        │     ┌───────────────────────────────┐     │
        │     │  ONNX Generator (单步)         │     │
        │     │  (pc, noisy_grasps, k)         │     │
        │     │       → noise_pred             │     │
        │     └───────────────────────────────┘     │
        │     noisy_grasps =                         │
        │        scheduler.step(noise_pred, k, ...)  │
        │                                           │
        │   return noisy_grasps  # 最终抓取          │
        └─────────────────────────────────────────┘
```

**优势**:
- ✅ ONNX 图完全静态（无循环、无动态分支）
- ✅ BPU 友好
- ✅ 调度器逻辑（DDPM step）保留在 Python，灵活可调
- ✅ 点云只编码一次，跨 K 个抓取广播复用

### 模型组件

#### 1. DiffusionHead (`models/diffusion_head.py`)

移植自上游 `DiffusionNoisePredictionNet`（`pose_repr="mlp"`，"cat" 路径）。

```
timestep --> SinusoidalPosEmb --> Linear --> Mish --> Linear   (步嵌入)
sample   --> Linear --> ReLU --> Linear                        (样本嵌入)
embed = cat([sample_embed, step_embed, obs_embed])
noise_pred = prediction_head(embed)
```

- 仅使用标准算子（Linear, ReLU, Mish, sin/cos, cat）
- 与 backbone 无关：只消费观测嵌入向量
- **未移植**上游的 transformer attention 分支（`cat_attn`），
  S600 部署采用静态 MLP 头（"cat"）以保证 BPU 兼容性

#### 2. GraspGenGeneratorONNX (`models/graspgen_onnx.py`)

```
pc (1, 2048, 3) ──> PointNetEncoder ──> object_feat (1, 512)
                                            │ repeat_interleave(K)
                                            ▼
noisy_grasps (K, 9) ──┐                object_feat (K, 512)
timestep (1,) ────────┼──> DiffusionHead ──> noise_pred (K, 9)
                      ┘
```

#### 3. GraspGenDiscriminatorONNX (`models/graspgen_onnx.py`)

```
pc (1, 2048, 3) ──> PointNetEncoder ──> object_feat (1, 512) ─┐ repeat(K)
                                                               ▼
grasps (1, K, 9) ──> sample_encoder ──> sample_feat (K, 512) ─┤
                                                               ▼
                              cat ──> prediction_head ──> sigmoid ──> scores (1, K)
```

---

## 导出产物

```
models/onnx/
├── graspgen_generator_pointnet.onnx        (176 KB + 18 MB .data)
└── graspgen_discriminator_pointnet.onnx    (152 KB + 6.4 MB .data)
```

输入/输出与 `configs/manifests/*.json` 契约一致：

**Generator**:
- 输入: `pc (1,2048,3)`, `noisy_grasps (20,9)`, `timestep (1,)`
- 输出: `noise_pred (20,9)`

**Discriminator**:
- 输入: `pc (1,2048,3)`, `grasps (1,20,9)`
- 输出: `scores (1,20)` ∈ [0,1]

---

## 重要说明：权重状态

⚠️ **当前 ONNX 模型使用随机初始化权重。**

**原因**: 上游预训练权重（`graspgen_franka_panda_gen.pth`）使用的架构是：
- `obs_backbone: ptv3`（PointTransformerV3 + spconv，**ONNX 不兼容**）
- `grasp_repr: r3_so3`（output_dim=6）
- `compositional_schedular: true`（位置/旋转分离调度器）
- `attention: cat_attn`（transformer attention）

而 S600 部署目标架构（契约定义）是：
- `obs_backbone: pointnet`（我们的 ONNX 兼容 PointNet++）
- `grasp_repr: r3_6d`（output_dim=9）
- 标准 DDPM 调度器
- 静态 MLP 头（"cat"）

两者 backbone 和表示不同，预训练权重**无法直接迁移**。

**因此精度恢复需要训练（Phase 4）**：
- Diffusion Head 的 MLP 权重可部分复用（如果切换到 r3_so3）
- PointNet encoder 需从头训练或蒸馏
- 当前导出验证的是**架构正确性和数值一致性**（PyTorch == ONNX）

---

## 遇到的问题与修复

### 问题 1: timestep 广播失败

**错误**:
```
RuntimeError: Sizes of tensors must match except in dimension 1.
Expected size 20 but got size 1 for tensor number 1 in the list.
```

**原因**: 传入的 timestep 形状为 `(1,)`，但 batch（K=20 个抓取）需要广播。
原代码只处理了标量（0 维）timestep。

**修复**:
```python
if torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
    timesteps = timesteps[None].to(device)
timesteps = timesteps.to(device)
if timesteps.shape[0] != batch:
    timesteps = timesteps.reshape(-1)[:1].expand(batch)
```

结果：误差降至 0.00000004 ✅

---

## 代码产出

```
src/python/graspgen_s600_tools/models/
├── pointnet_encoder.py      # Phase 2: ONNX 兼容 PointNet++
├── diffusion_head.py        # Phase 3: 噪声预测网络 (新增)
└── graspgen_onnx.py         # Phase 3: 完整 Generator/Discriminator (新增)

src/python/scripts/
├── test_pointnet_onnx.py    # Phase 2: encoder 测试
├── test_graspgen_onnx.py    # Phase 3: 完整模型测试 + DDPM 循环 (新增)
└── export_graspgen_onnx.py  # Phase 3: 导出到契约路径 (新增)
```

---

## 下一步：Phase 4

### 训练与精度恢复

1. **准备数据**: ACRONYM Franka Panda 子集（或合成数据）
2. **训练策略**:
   - 选项 A: 从头训练 PointNet encoder + Diffusion head
   - 选项 B: 知识蒸馏（PTV3 teacher → PointNet student）
3. **精度门控**: 抓取成功率 > 80%（初始目标）
4. **导出训练后权重**: `export_graspgen_onnx.py --generator-ckpt ...`

### Phase 5: HBM 编译与板端部署

1. 收集校准数据（点云 + 抓取样本）
2. INT16 量化编译（`hb_compile`）
3. S600 板端推理（Python DDPM 循环驱动 BPU 单步图）
4. 精度和性能优化

---

## 里程碑总结

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | 框架搭建，发现 PTV3/spconv 不兼容 | ✅ |
| Phase 2 | ONNX 兼容 PointNet++ encoder | ✅ |
| **Phase 3** | **完整模型集成 + ONNX 导出** | **✅** |
| Phase 4 | 训练与精度恢复 | ⏭️ |
| Phase 5 | HBM 编译与板端部署 | ⏭️ |

---

**报告生成时间**: 2026-06-21 14:30
**Phase 3 状态**: ✅ 完成
**精度门控**: ✅ 通过（Generator 4e-8, Discriminator 3e-8）
**下一阶段**: Phase 4 - 训练与精度恢复
**作者**: lvyufeng (with Claude Code)
