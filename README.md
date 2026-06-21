# GraspGen-s600

[English](#english) | [中文](#chinese)

---

<a name="english"></a>
## English

This repository is an adaptation of [NVlabs/GraspGen](https://github.com/NVlabs/GraspGen) for the Horizon Sunrise 6 (S600) platform.

### Original Project

GraspGen is a generative grasp pose synthesis system developed by NVIDIA Labs. It enables robust 6-DOF grasp generation for robotic manipulation tasks.

### Adaptation Overview

This project adapts GraspGen to run on the Horizon Sunrise 6 (S600) embedded AI platform, leveraging the BPU (Brain Processing Unit) for efficient inference on edge devices.

### Key Adaptation Goals

- Convert models to Horizon-compatible format (ONNX → .bin)
- Optimize inference pipeline for BPU acceleration
- Implement quantization (INT8/INT16) for efficient edge deployment
- Maintain grasp generation quality while reducing latency
- Minimize memory footprint for embedded systems

### Status

🚧 **Work in Progress** - Currently adapting the original implementation

### Platform Requirements

- Horizon Sunrise 6 (S600) development board
- Horizon OpenExplorer toolchain
- Horizon Runtime Library
- Python 3.x

### Reference Repositories

- Original Project: [NVlabs/GraspGen](https://github.com/NVlabs/GraspGen)
- Horizon Developer Portal: [developer.horizon.cc](https://developer.horizon.cc/)
- Related S600 Adaptations:
  - [FoundationPose-s600](../FoundationPose-s600)
  - [SAM-s600](../SAM_s600)

### License

This adaptation follows the license terms of the original GraspGen project.

---

<a name="chinese"></a>
## 中文

本仓库是 [NVlabs/GraspGen](https://github.com/NVlabs/GraspGen) 在地平线 Sunrise 6（S600）平台上的适配版本。

### 原始项目介绍

GraspGen 是由 NVIDIA 实验室开发的生成式抓取姿态合成系统，可为机器人操作任务生成鲁棒的 6 自由度抓取姿态。

### 适配说明

本项目将 GraspGen 适配到地平线 Sunrise 6（S600）嵌入式 AI 平台，利用 BPU（Brain Processing Unit）实现在边缘设备上的高效推理。

### 主要适配目标

- 将模型转换为地平线兼容格式（ONNX → .bin）
- 针对 BPU 加速优化推理流程
- 实现量化（INT8/INT16）以支持高效边缘部署
- 在降低延迟的同时保持抓取生成质量
- 优化内存占用以适应嵌入式系统

### 项目状态

🚧 **开发中** - 正在进行原始实现的适配工作

### 平台要求

- 地平线 Sunrise 6（S600）开发板
- 地平线 OpenExplorer 工具链
- Horizon Runtime Library
- Python 3.x

### 参考资料

- 原始项目：[NVlabs/GraspGen](https://github.com/NVlabs/GraspGen)
- 地平线开发者平台：[developer.horizon.cc](https://developer.horizon.cc/)
- 相关 S600 适配项目：
  - [FoundationPose-s600](../FoundationPose-s600)
  - [SAM-s600](../SAM_s600)

### 许可证

本适配项目遵循原始 GraspGen 项目的许可证条款。
