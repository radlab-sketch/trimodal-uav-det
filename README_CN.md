# TriModalDet — 三模态无人机目标检测

## 项目概述

TriModalDet 是一个基于 RGB、热红外（Thermal）和事件相机（Event）的三模态目标检测框架。通过 MAGE（Modality-Aware Gated Exchange）和 BiTE（Bidirectional Token Exchange）融合机制，在不同环境条件下实现鲁棒的目标检测。

## 架构设计

### 总体架构

```
输入 (B, 5, H, W)
├── RGB 分支 (3 ch)  → Patch Embed → MiT Transformer Stages 1-4
├── X 分支 (2 ch: Thermal+Event) → Patch Embed → MiT Transformer Stages 1-4
│   ├── MAGE（通道+空间门控） ← 自适应模态加权
│   └── BiTE（跨模态注意力） ← 双向特征交换
└── FPN (4 stages) → Faster R-CNN 检测头
```

### 核心组件

**1. 跨模态骨干网络（Inter-Modal Backbone）**
- 基于 SegFormer MiT（Mix Transformer）的双分支架构
- RGB 分支：处理 3 通道 RGB 图像
- 辅助模态分支（X）：处理 2 通道热红外+事件相机数据
- 支持 5 种变体（mit_b0 ~ mit_b4），平衡速度/精度

**2. 三模态融合模块**
- **MAGE（Modality-Aware Gated Exchange）**：通道门控 + 空间门控，自适应加权各模态贡献
- **BiTE（Bidirectional Token Exchange）**：跨模态注意力，实现双向信息流动
- 阶段级融合：在每个 Transformer Stage 后进行融合

**3. 检测头**
- **FPN（Feature Pyramid Network）**：多尺度特征聚合
- **Faster R-CNN**：分类 + 定位（RPN 提案 + RoI Align + 分类/回归）

## 数据格式

### 输入图像

图像为 `.npy` 格式的 NumPy 数组，shape 为 `(H, W, 5)`：
- **通道 0-2**：RGB
- **通道 3**：热红外（Thermal）
- **通道 4**：事件相机（Event）

```python
import numpy as np
image = np.zeros((480, 640, 5), dtype=np.uint8)
image[:, :, 0:3] = rgb_data
image[:, :, 3] = thermal_data
image[:, :, 4] = event_data
np.save('data/images/frame_001.npy', image)
```

### 标注格式

使用 YOLO 格式（每行一个目标）：
```
class_id x_center y_center width height
```
所有坐标归一化到 `[0, 1]`。

### 目录结构

```
data/
├── images/    (放置 .npy 图像文件)
└── labels/    (放置 .txt 标注文件)
```

## 环境配置

### 环境信息

- **Python**：3.12
- **PyTorch**：2.9.1 + CUDA 12.8
- **GPU**：RTX 5080（Blackwell 架构）
- **Conda 环境**：`triair`

### 安装步骤

```bash
conda create -n triair python=3.12 -y
conda activate triair

# 安装 PyTorch CUDA 12.8 版本
pip install torch==2.9.1+cu128 torchvision==0.24.1+cu128 torchaudio==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128

# 安装其余依赖
pip install timm matplotlib scikit-learn torchmetrics pycocotools

# 或直接使用 requirements2.txt
pip install -r requirements2.txt --index-url https://download.pytorch.org/whl/cu128
```

### 验证环境

```python
import torch
print(torch.__version__)          # 2.9.1+cu128
print(torch.cuda.is_available())  # True
print(torch.version.cuda)         # 12.8
```

## 使用指南

### 训练

#### 主模型（全模态：RGB + Thermal + Event）

```bash
# 默认配置（mit_b1 backbone，15 epochs，batch_size=16）
python scripts/train.py --data E:\dataset\CV\triair\data --epochs 15 --batch-size 16

# 快速原型验证（mit_b0，最小模型）
python scripts/train.py --data E:\dataset\CV\triair\data --epochs 15 --backbone mit_b0 --batch-size 32

# 高精度配置（mit_b4，最大模型）
python scripts/train.py --data E:\dataset\CV\triair\data --epochs 15 --backbone mit_b4 --batch-size 4
```

#### 模态消融实验（无事件相机）

```bash
python trimodaldet/ablations/scripts/train_modality_ablation.py \
    --data E:\dataset\CV\triair\data\images \
    --labels E:\dataset\CV\triair\data\labels \
    --epochs 15 \
    --backbone mit_b1 \
    --modalities rgb,thermal \
    --output-dir results/ablation_no_event
```

### 测试

```bash
# 单模型评估
python scripts/test.py --data E:\dataset\CV\triair\data --model trimodaldet.pth
```

### 可视化

```bash
# 可视化数据集样本
python scripts/visualize.py --vis 0 --data E:\dataset\CV\triair\data
```

## 配置参数

在 `trimodaldet/config.py` 中可修改以下配置：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `backbone_type` | `'mit_b1'` | 骨干网络变体 |
| `num_epochs` | `15` | 训练轮数 |
| `batch_size` | `16` | 批次大小 |
| `learning_rate` | `0.02` | 学习率 |
| `momentum` | `0.9` | SGD 动量 |
| `weight_decay` | `0.0001` | 权重衰减 |
| `test_size` | `0.2` | 测试集比例 |
| `score_threshold` | `0.5` | 置信度阈值 |

### 骨干网络对比

| 变体 | 参数量 | 深度 | 嵌入维度 | 适用场景 |
|------|--------|------|----------|----------|
| mit_b0 | ~3.7M | [2,2,2,2] | [32,64,160,256] | 快速原型、边缘设备 |
| mit_b1 | ~13.5M | [2,2,2,2] | [64,128,320,512] | 默认均衡配置 |
| mit_b2 | ~24.7M | [3,4,6,3] | [64,128,320,512] | 更高精度 |
| mit_b3 | ~44M | [3,4,18,3] | [64,128,320,512] | 中大型模型 |
| mit_b4 | ~61.4M | [3,8,27,3] | [64,128,320,512] | 最大精度 |

## 项目结构

```
trimodal-uav-det/
├── scripts/                    # 入口脚本
│   ├── train.py               # 主训练脚本
│   ├── test.py                # 测试评估脚本
│   └── visualize.py           # 可视化脚本
├── trimodaldet/               # 主包
│   ├── models/                # 模型架构
│   │   ├── encoder.py         # InterModalBackbone (MiT 双分支)
│   │   ├── fusion.py          # MAGE + BiTE 融合模块
│   │   ├── backbone.py        # FPN 包装器
│   │   └── transformer.py     # Transformer 基础模块
│   ├── training/              # 训练与评估
│   │   ├── trainer.py         # 训练循环
│   │   └── evaluator.py       # 评估器
│   ├── data/                  # 数据加载
│   │   ├── dataset.py         # NpyYoloDataset (五通道 .npy)
│   │   └── transforms.py      # YOLO/COCO 格式转换
│   ├── utils/                 # 工具函数
│   │   ├── metrics.py         # mAP 评估指标
│   │   ├── timm_compat.py     # Timm 兼容性模块
│   │   └── visualization.py   # 可视化
│   ├── ablations/             # 消融实验
│   │   ├── backbone_modality.py          # 模态可配置 Backbone
│   │   └── scripts/            # 消融训练脚本
│   │       └── train_modality_ablation.py  # 模态消融
│   └── config.py              # 配置管理
├── requirements2.txt          # 依赖清单
├── README.md                   # 英文 README
├── README_CN.md              # 中文 README
├── CLAUDE.md                 # 项目速览
└── data/                      # 数据目录（用户创建）
    ├── images/
    └── labels/
```

## 训练详情

- **优化器**：SGD + Momentum
- **损失函数**：Faster R-CNN 分类 + 边界框回归
- **数据划分**：自动 80/20 训练/测试划分
- **学习率调度**：Linear Warmup（500 steps）+ Cosine Annealing
- **设备**：自动检测 GPU，否则回退到 CPU

## 许可证

MIT License
