# GraspGen-s600 Phase 1 总结

## 已完成的工作

### 1. 项目结构搭建 ✅

创建了标准化的 s600 适配目录结构：

```
GraspGen-s600/
├── src/python/graspgen_s600_tools/
│   ├── export/          # ONNX 导出工具
│   ├── convert/         # 模型转换工具
│   ├── runtime/         # BPU 运行时适配器
│   └── debug/           # 验证和调试工具
├── models/
│   ├── onnx/            # ONNX 模型文件
│   ├── hbm/             # 编译后的 HBM 文件
│   └── upstream/        # 原始权重
├── configs/
│   ├── manifests/       # 模型契约定义
│   └── calibration/     # 量化校准数据
├── docs/                # 文档
├── benchmarks/          # 性能基准
└── third_party/
    └── GraspGen/        # 上游源码
```

### 2. Contract 生成模块 ✅

实现了 `graspgen_s600_tools/export/contract.py`，支持：

- **Generator Contract**: 
  - 输入: 点云 (1, 2048, 3)
  - 输出: 抓取姿态 (1, 20, 9) - r3_6d 表示
  - 编译提示: 双核, INT16, O2 优化

- **Discriminator Contract**:
  - 输入: 点云 (1, 2048, 3) + 抓取候选 (1, 20, 9)
  - 输出: 质量评分 (1, 20)
  - 编译提示: 单核, INT16, O2 优化

生成的 JSON 契约文件位于 `configs/manifests/`。

### 3. 上游项目集成 ✅

- 成功克隆 [NVlabs/GraspGen](https://github.com/NVlabs/GraspGen) 到 `third_party/GraspGen`
- 分析了核心模型结构：
  - `grasp_gen/models/generator.py` - Generator 模型
  - `grasp_gen/models/discriminator.py` - Discriminator 模型
  - 基于 Diffusion 的生成架构
  - 支持 PointNet++、PTV3、ViT 等多种骨干网络

### 4. Git 仓库配置 ✅

- 配置了 `.gitignore` 忽略大文件（模型、数据集）
- 设置了 git 用户信息
- 完成首次提交

## 上游 GraspGen 架构分析

### 核心组件

1. **Generator (生成器)**
   - 基于 DDPM Diffusion 模型
   - 默认 100 步训练，20 步推理（可优化）
   - 输入: 归一化点云 [-1, 1]
   - 输出: 6-DOF 抓取姿态
   - 支持多种表示: r3_6d (推荐), euler, so3

2. **Discriminator (判别器)**
   - 评估抓取质量
   - 输入: 点云 + 候选抓取
   - 输出: 质量评分 [0, 1]

3. **Backbone 选项**
   - `pointnet`: PointNet++ (默认，最兼容)
   - `ptv3`: PointTransformerV3 (高精度)
   - `vit`: Vision Transformer (图像输入)

### 推理流程

```python
# 伪代码
pc = normalize_point_cloud(input_pc)  # 归一化到 [-1, 1]
object_feat = object_encoder(pc)      # 提取物体特征

# Diffusion 去噪循环 (20 步)
noisy_grasps = torch.randn(batch_size, num_grasps, output_dim)
for t in timesteps:
    noise_pred = diffusion_head(noisy_grasps, object_feat, t)
    noisy_grasps = scheduler.step(noise_pred, t, noisy_grasps)

grasps_pred = noisy_grasps  # 最终预测

# 质量评分
scores = discriminator(pc, grasps_pred)
top_grasps = grasps_pred[scores.argsort(descending=True)]
```

## 关键发现

### 1. Diffusion 循环挑战

上游使用动态循环：
```python
for k in timesteps:  # 20 次迭代
    noise_pred = diffusion_head(noisy_grasps, object_embedding, k)
    noisy_grasps = scheduler.step(noise_pred, k, noisy_grasps)
```

**适配策略**:
- **方案 A**: 单步去噪函数导出到 ONNX，Python 层循环调用（推荐）
- **方案 B**: 展开固定 20 步静态图（ONNX 体积大）
- **方案 C**: 导出完整推理函数，接受预初始化噪声

### 2. PointNet++ 自定义算子

可能包含 CUDA 自定义算子 (Farthest Point Sampling)。

**缓解措施**:
- 优先使用纯 PyTorch PointNet 实现
- 检查 ONNX 算子支持
- 必要时预计算采样索引

### 3. 量化敏感性

Diffusion 模型对量化敏感，参考 FoundationPose-s600 经验：
- **INT16 全图量化** 是安全选择
- 保持输出层 FP32/FP16
- 充分的校准数据 (>100 样本)

## 🎯 核心策略：精度优先

**重要原则**: 在整个适配过程中，精度优先于速度。每个 Phase 必须通过精度验证门控才能进入下一阶段。

### 精度门控标准

- **ONNX 导出**: PyTorch vs ONNX 误差 < 1e-3
- **HBM 编译**: BPU vs CPU 抓取成功率 > 85%
- **姿态精度**: 位置误差 < 5mm，旋转误差 < 5°
- **量化策略**: 默认 INT16，仅在验证通过后考虑 INT8

### 失败回退策略

如果 INT16 出现精度下降：
1. 尝试 FP16 混合精度（关键层保持 FP16）
2. 识别敏感层并单独调优
3. 增加校准样本（>200）
4. 仅在充分验证后使用 INT8

详见：[Adaptation Strategy](adaptation_strategy.md)

## 下一步计划

### Phase 1 剩余任务

#### Task 1.1: 实现 ONNX 导出 🎯

创建 `src/python/graspgen_s600_tools/export/generator.py`:

```python
class GraspGenGeneratorONNXWrapper(nn.Module):
    """单步去噪函数，外部循环调用"""
    def __init__(self, generator):
        super().__init__()
        self.object_encoder = generator.object_encoder
        self.diffusion_head = generator.diffusion_head
        
    def forward(self, pc, noisy_grasps, timestep):
        """
        输入:
          pc: (1, 2048, 3)
          noisy_grasps: (1, 20, 9)
          timestep: (1,) - int tensor
        输出:
          noise_pred: (1, 20, 9)
        """
        object_feat = self.object_encoder(pc)
        noise_pred = self.diffusion_head(noisy_grasps, object_feat, timestep)
        return noise_pred
```

#### Task 1.2: 模型加载工具 🎯

创建 `src/python/graspgen_s600_tools/export/factories/graspgen.py`:

```python
def load_generator(checkpoint_path, config):
    """加载 Generator 模型"""
    from grasp_gen.models.generator import GraspGenGenerator
    
    model = GraspGenGenerator(**config)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model

def load_discriminator(checkpoint_path, config):
    """加载 Discriminator 模型"""
    # 类似实现
```

#### Task 1.3: 下载预训练权重 🎯

从 HuggingFace 下载：
```bash
# 使用 huggingface-cli 或手动下载
huggingface-cli download adithyamurali/GraspGenModels \
  franka_panda_pointnet_generator.pth \
  franka_panda_pointnet_discriminator.pth \
  --local-dir models/upstream/
```

#### Task 1.4: ONNX 导出脚本 🎯

创建 `src/python/scripts/export_onnx.py`:
```python
def export_generator():
    # 加载模型
    generator = load_generator(...)
    
    # 包装为 ONNX 友好形式
    wrapper = GraspGenGeneratorONNXWrapper(generator)
    
    # 导出
    torch.onnx.export(
        wrapper,
        (dummy_pc, dummy_noisy, dummy_t),
        "models/onnx/graspgen_generator_pointnet.onnx",
        opset_version=17,
        input_names=['pc', 'noisy_grasps', 'timestep'],
        output_names=['noise_pred'],
        dynamic_axes=None,  # 固定形状
    )
```

#### Task 1.5: ONNX 验证 🎯

创建 `src/python/graspgen_s600_tools/debug/validate_onnx.py`:
```python
def validate_onnx_export(pytorch_model, onnx_path):
    """对比 PyTorch vs ONNX 输出"""
    # PyTorch 推理
    pt_out = pytorch_model(dummy_input)
    
    # ONNX 推理
    ort_session = ort.InferenceSession(onnx_path)
    onnx_out = ort_session.run(None, {input_dict})
    
    # 精度检查
    error = np.abs(pt_out - onnx_out).max()
    assert error < 1e-3, f"Export error: {error}"
```

### Phase 1 验收标准

- [ ] ONNX 导出成功，通过 `onnx.checker.check_model()`
- [ ] PyTorch vs ONNX 精度误差 < 1e-3
- [ ] ONNX 模型形状与 contract 一致
- [ ] 文档化导出流程

## 参考资料

### 已分析的 s600 项目

1. **FoundationPose-s600**
   - 类似的基于 Diffusion 的姿态估计
   - 使用 INT16 量化 + Output FP16
   - 双核编译，达到实时性能

2. **SAM_s600**
   - 大规模 ViT 模型适配
   - 286ms → 143ms (双核加速)
   - 精度保持策略参考

### 工具链文档

- Horizon OpenExplorer Toolchain
- HBRT (Horizon Runtime)
- hb_compile 编译参数

### 上游 GraspGen

- GitHub: https://github.com/NVlabs/GraspGen
- Paper: arXiv 2507.13097
- Models: huggingface.co/adithyamurali/GraspGenModels

## 时间估算

- Task 1.1-1.2: 1-2 天（模型加载和包装）
- Task 1.3: 0.5 天（下载权重）
- Task 1.4: 2-3 天（ONNX 导出调试）
- Task 1.5: 1 天（验证）

**Phase 1 预计完成时间**: 1-1.5 周

---

**更新时间**: 2026-06-21  
**当前状态**: 结构搭建完成，准备开始 ONNX 导出实现  
**下一里程碑**: 成功导出 Generator 到 ONNX
