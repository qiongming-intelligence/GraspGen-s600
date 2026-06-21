# Phase 1 ONNX Export - 当前状态和挑战

## 当前状态 📊

### 已完成 ✅

1. **所有代码实现完成并推送**
   - Contract 生成器
   - 模型加载工具 (factories.py)
   - Generator ONNX 包装器
   - Discriminator ONNX 包装器
   - 导出脚本和测试工具

2. **预训练权重已下载（ws-wan）**
   - graspgen_franka_panda_gen.pth (866 MB) ✓
   - graspgen_franka_panda_dis.pth (159 MB) ✓
   - graspgen_franka_panda.yml (4.8 KB) ✓

3. **依赖环境搭建**
   - Python 3.12 venv 创建 ✓
   - PyTorch 2.12.1 安装 ✓
   - ONNX, ONNXRuntime 安装 ✓
   - GraspGen 基础依赖安装 ✓

### 当前挑战 🚧

#### 1. PointNet2 C++ 扩展依赖

**问题描述**:
```
RuntimeError: Ninja is required to load C++ extensions
Unable to load pointnet2_ops cpp extension. JIT Compiling.
```

**根本原因**:
- 上游 GraspGen 使用自定义 CUDA 算子 (Farthest Point Sampling 等)
- 需要编译 C++ 扩展，依赖 ninja, CUDA toolkit
- 即使编译成功，CUDA 自定义算子也无法导出到 ONNX

**配置分析**:
从 `graspgen_franka_panda.yml`:
```yaml
diffusion:
  obs_backbone: ptv3  # PointTransformerV3，不是 pointnet2！
discriminator:
  obs_backbone: ptv3
```

**关键发现**: 实际模型使用的是 `ptv3` (PointTransformerV3)，但代码导入路径仍然触发了 PointNet2 的加载。

#### 2. 架构兼容性问题

**版本冲突**:
- 原始模型: torch 2.1.0, diffusers 0.11.1, numpy 1.26.4
- 当前环境: torch 2.12.1, diffusers 0.38.0, numpy 2.4.6

虽然安装成功，但 API 变化可能导致模型加载失败。

## 解决方案路径 🛠️

### 方案 A: 绕过 PointNet2 依赖（推荐） ⭐

**思路**: 直接导出 PTV3 模型，避免触发 PointNet2 加载

**步骤**:
1. 修改 `factories.py`，延迟导入 GraspGenGenerator
2. 加载 checkpoint 后直接提取 PTV3 backbone
3. 跳过不需要的组件（如训练用的 metrics）

**优势**:
- 无需编译 C++ 扩展
- 专注于实际使用的 PTV3 模型
- 适合 ONNX 导出（PTV3 是纯 PyTorch）

**实施**:
```python
# 直接加载 state_dict，避免触发 PointNet2 导入
checkpoint = torch.load(checkpoint_path, map_location='cpu')
state_dict = checkpoint['model']

# 手动构建模型（仅包含 PTV3）
from grasp_gen.models.ptv3.ptv3 import PointTransformerV3
encoder = PointTransformerV3(...)
encoder.load_state_dict({k: v for k, v in state_dict.items() if 'encoder' in k})
```

### 方案 B: 安装 PointNet2++ 扩展

**步骤**:
1. 安装 CUDA development toolkit
2. 编译 `third_party/GraspGen/pointnet2_ops`
3. 确保自定义 CUDA 算子可加载

**挑战**:
- 编译环境要求高（CUDA, gcc 兼容性）
- 自定义 CUDA 算子可能无法导出到 ONNX
- 即使导出成功，BPU 不支持 CUDA 算子

**不推荐原因**: 最终模型使用 PTV3，PointNet2 是无用依赖

### 方案 C: 切换到纯 PyTorch PointNet（备选）

如果 PTV3 也有问题：
1. 使用第三方纯 PyTorch PointNet 实现
2. 重新训练或微调适配器
3. 确保完全兼容 ONNX

**时间成本**: 高（需要重新训练）

## 下一步行动 📋

### 优先级 1: 实施方案 A（今日完成目标）

1. **修改模型加载逻辑** (30分钟)
   - 延迟导入，避免触发 PointNet2
   - 直接加载 state_dict
   - 仅构建 PTV3 encoder 和 diffusion head

2. **测试 PTV3 导出** (1小时)
   - 验证 PTV3 可以导出到 ONNX
   - 检查算子兼容性
   - 如果失败，回退到标准 PointNet

3. **精度验证** (1小时)
   - PyTorch vs ONNX 输出对比
   - 确保误差 < 1e-3

4. **文档和提交** (30分钟)
   - 记录实施细节
   - 更新 Phase 1 总结
   - 推送到远程

### 优先级 2: 如果方案 A 失败

- 调研 PTV3 ONNX 兼容性
- 评估切换到纯 PyTorch PointNet 的成本
- 与团队讨论是否接受额外的适配工作

## 精度优先原则坚持 🎯

虽然遇到依赖问题，但我们保持：
- ✅ 不妥协精度要求（误差 < 1e-3）
- ✅ 选择兼容 ONNX 的架构
- ✅ 避免引入无法在 BPU 上运行的算子
- ✅ 充分测试每个组件

## 时间估算 ⏰

- 方案 A 实施: 2-3 小时
- 如需方案 B: +4-6 小时（编译调试）
- 如需方案 C: +2-3 天（重新训练）

**当前最佳路径**: 方案 A
**预计今日完成**: 有一定挑战，但可行

---

**更新时间**: 2026-06-21 13:00  
**下一步**: 实施方案 A - 绕过 PointNet2，直接导出 PTV3
