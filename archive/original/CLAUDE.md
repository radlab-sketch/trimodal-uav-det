# TriModalDet — CLAUDE.md

## 项目速览

TriModalDet 是一个三模态（RGB + 热红外 + 事件相机）无人机目标检测框架。

- **核心架构**：SegFormer MiT 双分支 Backbone + MAGE/BiTE 融合模块 + Faster R-CNN 检测头
- **数据集**：`E:\dataset\CV\triair\data`（.npy 五通道图像 + YOLO 格式标签）
- **环境**：`triair` conda 环境（Python 3.12 + torch 2.9.1+cu128 + CUDA 12.8）
- **GitHub**：https://github.com/xcs-336/trimodal-uav-det.git

## 关键文件

| 文件 | 说明 |
|------|------|
| `scripts/train.py` | 主训练入口 |
| `scripts/test.py` | 测试评估入口 |
| `scripts/visualize.py` | 数据集可视化 |
| `trimodaldet/config.py` | 配置管理（路径、超参数、命令行解析） |
| `trimodaldet/models/encoder.py` | InterModalBackbone（MiT 双分支编码器） |
| `trimodaldet/models/fusion.py` | MAGE + BiTE 融合模块 |
| `trimodaldet/models/backbone.py` | FPN 包装器（适配 torchvision Faster R-CNN） |
| `trimodaldet/training/trainer.py` | 训练循环（SGD + warmup + cosine annealing） |
| `trimodaldet/training/evaluator.py` | 评估器（加载 checkpoint + 测试） |
| `trimodaldet/data/dataset.py` | NpyYoloDataset（五通道 .npy + YOLO 标签） |
| `trimodaldet/utils/timm_compat.py` | Timm 兼容性模块（DropPath / to_2tuple / trunc_normal_） |
| `trimodaldet/ablations/backbone_modality.py` | 模态可配置 Backbone（消融实验） |
| `trimodaldet/ablations/scripts/train_modality_ablation.py` | 模态消融训练脚本 |

## 模型架构

```
Input (B, 5, H, W)
├── RGB 分支 (3 ch) → Patch Embed → Transformer Stages 1-4
├── X 分支 (2 ch: Thermal+Event) → Patch Embed → Transformer Stages 1-4
│   ├── MAGE（通道+空间门控）  ← 自适应模态加权
│   └── BiTE（跨模态注意力）  ← 双向特征交换
└── FPN (4 stages) → Faster R-CNN Head
```

## 用户偏好

- **系统**：Windows 11，PowerShell 终端
- **环境**：conda 环境管理，新建独立 `triair` 环境
- **Git**：每次提交到 GitHub 必须创建新分支并提交 PR，禁止直接合并到 main
- **Python**：3.12，utf-8 编码
- **GPU**：RTX 5080，CUDA 12.8，torch 2.9.1+cu128

## 数据格式

- **图像**：.npy 文件，shape (H, W, 5)，通道顺序：RGB(0-2) + Thermal(3) + Event(4)
- **标签**：.txt 文件，YOLO 格式（`class_id x_center y_center width height`，归一化 0-1）
- **目录结构**：
  ```
  data/
  ├── images/    (frame_00000.npy, ...)
  └── labels/    (frame_00000.txt, ...)
  ```

## 训练命令

```bash
# 主模型（全模态）
python scripts/train.py --data E:\dataset\CV\triair\data --epochs 15 --batch-size 16 --backbone mit_b1

# 模态消融（无事件相机）
python trimodaldet/ablations/scripts/train_modality_ablation.py \
    --data E:\dataset\CV\triair\data\images \
    --labels E:\dataset\CV\triair\data\labels \
    --epochs 15 --backbone mit_b1 --modalities rgb,thermal \
    --output-dir results/ablation_no_event

# 测试
python scripts/test.py --data E:\dataset\CV\triair\data --model trimodaldet.pth

# 可视化
python scripts/visualize.py --vis 0 --data E:\dataset\CV\triair\data
```

## 环境

```bash
conda activate triair
# 依赖：见 requirements2.txt
```

## 兼容性问题

- `timm>=1.0` 中 `timm.models.layers` 已弃用，改为 `timm.layers`
- `torchmetrics>=1.4` 中 `MeanAveragePrecision` API 路径不变
