# Phase 1: 基础适配

## 当前状态

✅ 目录结构已创建  
✅ Contract 生成模块完成  
✅ Generator 和 Discriminator contracts 已生成  

## 下一步：克隆上游 GraspGen

```bash
# 1. 克隆上游项目
cd /home/sunrise/Projects/GraspGen-s600
git clone https://github.com/NVlabs/GraspGen.git third_party/GraspGen

# 2. 查看上游项目结构
cd third_party/GraspGen
ls -la
```

## 需要的模型权重

上游 GraspGen 的预训练权重需要从 HuggingFace 下载：

- **Repository**: `adithyamurali/GraspGenModels`
- **推荐模型**:
  - `franka_panda_pointnet_generator.pth`
  - `franka_panda_pointnet_discriminator.pth`

下载后放置到：
```
models/upstream/
├── franka_panda_pointnet_generator.pth
└── franka_panda_pointnet_discriminator.pth
```

## 环境准备

### 导出环境 (x86_64 主机)

```bash
conda create -n graspgen-export python=3.11
conda activate graspgen-export
conda install pytorch torchvision onnx onnxruntime -c pytorch
pip install diffusers omegaconf hydra-core transformers trimesh scipy
```

### 编译环境 (x86_64 主机)

```bash
conda create -n graspgen-compile python=3.10
conda activate graspgen-compile
conda install numpy=1.23.0
# 安装 Horizon OpenExplorer Toolchain
pip install horizon_tc_ui-<version>.whl
```

## Phase 1 目标

- [ ] 克隆上游 GraspGen 项目
- [ ] 下载预训练权重
- [ ] 实现 ONNX 导出模块
- [ ] 验证 ONNX 导出成功
- [ ] 编译 HBM 文件
- [ ] 板端加载验证

## 文件结构

```
GraspGen-s600/
├── configs/
│   └── manifests/
│       ├── graspgen_generator.json       ✅ 已生成
│       └── graspgen_discriminator.json   ✅ 已生成
├── src/python/graspgen_s600_tools/
│   ├── export/
│   │   ├── __init__.py                   ✅ 已创建
│   │   └── contract.py                   ✅ 已创建
│   ├── convert/                          ⏳ 待开发
│   ├── runtime/                          ⏳ 待开发
│   └── debug/                            ⏳ 待开发
└── third_party/
    └── GraspGen/                         ⏳ 待克隆
```
