#!/usr/bin/env python
"""
快速训练验证 + 资源监控脚本。
只跑前 N 个 batch，采集资源占用数据，输出优化建议。
支持显存超过阈值自动终止。
"""
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trimodaldet.config import Config
from trimodaldet.training.trainer import Trainer
from trimodaldet.training.monitor_utils import get_gpu_memory_stats
from scripts.monitor import ResourceMonitor


def quick_train_with_monitor(num_batches=5, batch_size=16, max_gpu_mem_pct=85.0):
    """
    快速训练验证：只跑前 num_batches 个 batch，同时监控资源。
    当 GPU 显存超过阈值时自动终止。
    """
    config = Config()
    # 手动设置数据路径
    config.data_root = 'E:/dataset/CV/triair/data'
    config.image_dir = 'E:/dataset/CV/triair/data/images'
    config.label_dir = 'E:/dataset/CV/triair/data/labels'
    config.num_epochs = 1
    config.batch_size = batch_size
    config.backbone_type = 'mit_b1'
    config.model_path = 'trimodaldet_quick.pth'

    print("=== 快速训练验证 + 资源监控 ===")
    print(f"  batch_size: {batch_size}")
    print(f"  max_gpu_mem_pct: {max_gpu_mem_pct}%")

    # 启动资源监控
    monitor = ResourceMonitor(interval=1.0, output='monitor_quick.json')
    monitor.start()

    try:
        # 初始化 Trainer
        trainer = Trainer(config)
        trainer.model.train()
        total_loss = 0
        terminated = False

        for i, (images, targets) in enumerate(trainer.train_loader):
            if i >= num_batches:
                break

            images = list(image.to(config.device) for image in images)
            targets = [{k: v.to(config.device) for k, v in t.items()} for t in targets]

            loss_dict = trainer.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            trainer.optimizer.zero_grad()
            losses.backward()
            trainer.optimizer.step()

            total_loss += losses.item()
            print(f"  Batch [{i+1}/{num_batches}] Loss: {losses.item():.4f}")

            # 每 batch 检查显存占用
            stats = get_gpu_memory_stats()
            if stats and stats['util_pct'] > max_gpu_mem_pct:
                print(f"\n[TERMINATED] GPU 显存占用 {stats['util_pct']:.1f}% 超过阈值 {max_gpu_mem_pct}%，"
                      f"自动终止训练以避免 OOM。")
                terminated = True
                break

        if not terminated:
            avg_loss = total_loss / min(num_batches, len(trainer.train_loader))
            print(f"\n快速验证完成。平均 Loss: {avg_loss:.4f}")
        else:
            print(f"\n快速验证已终止（显存超限）。已跑 {i+1} batch，平均 Loss: {total_loss / (i+1):.4f}")

    except Exception as e:
        print(f"\n训练过程中出错: {e}")
        raise
    finally:
        monitor.stop()
        monitor.print_summary()

        # 分析监控结果并给出优化建议
        if monitor.records:
            analyze_and_recommend(monitor.records, batch_size=batch_size)


def analyze_and_recommend(records, batch_size=None):
    """根据监控记录分析资源占用并给出优化建议。"""
    gpu_mem_max = max(r.get('gpu_mem_used_mb', 0) for r in records)
    gpu_mem_total = max(r.get('gpu_mem_total_mb', 1) for r in records)
    gpu_mem_pct = gpu_mem_max / gpu_mem_total * 100
    gpu_util_max = max(r.get('gpu_util_pct', 0) for r in records)
    mem_max = max(r.get('mem_percent', 0) for r in records)
    cpu_max = max(r.get('cpu_percent', 0) for r in records)

    print("\n=== 优化建议 ===")

    # GPU 显存分析
    if gpu_mem_pct > 90:
        print(f"  [警告] GPU 显存占用过高: {gpu_mem_max:.0f} / {gpu_mem_total:.0f} MB ({gpu_mem_pct:.1f}%)")
        print("  建议：")
        if batch_size and batch_size > 1:
            print(f"    - 减小 batch_size（当前 {batch_size} -> {batch_size // 2} 或更小）")
        else:
            print("    - 减小 batch_size（建议从 16 -> 8 -> 4 逐步尝试）")
        print("    - 使用更小的 backbone（mit_b1 -> mit_b0）")
        print("    - 启用 AMP（--use-amp）可减少 40-50% 显存占用")
        print("    - 启用梯度累积（--grad-accumulation-steps）保持等效 batch_size")
    elif gpu_mem_pct > 70:
        print(f"  [注意] GPU 显存占用较高: {gpu_mem_max:.0f} / {gpu_mem_total:.0f} MB ({gpu_mem_pct:.1f}%)")
        print("  建议：")
        print("    - 可略微减小 batch_size 以留余量")
        print("    - 监控后续训练是否稳定")
    else:
        print(f"  [正常] GPU 显存占用: {gpu_mem_max:.0f} / {gpu_mem_total:.0f} MB ({gpu_mem_pct:.1f}%)")
        print("  建议: 显存充足，可尝试增大 batch_size 提升训练效率")

    # GPU 利用率分析
    if gpu_util_max < 50:
        print(f"  [警告] GPU 利用率偏低: 峰值 {gpu_util_max:.1f}%")
        print("  建议：")
        if batch_size:
            print(f"    - 增大 batch_size（当前 {batch_size}）")
        else:
            print("    - 增大 batch_size")
        print("    - 检查数据加载是否为瓶颈")
        print("    - 使用 torch.backends.cudnn.benchmark = True")
    elif gpu_util_max < 80:
        print(f"  [注意] GPU 利用率一般: 峰值 {gpu_util_max:.1f}%")
        print("  建议: 可尝试微调 batch_size 或优化数据加载")
    else:
        print(f"  [正常] GPU 利用率良好: 峰值 {gpu_util_max:.1f}%")

    # 系统内存分析
    if mem_max > 90:
        print(f"  [警告] 系统内存占用过高: {mem_max:.1f}%")
        print("  建议：")
        print("    - 检查数据加载是否一次性加载过多图像到内存")
        print("    - 减小 DataLoader 的 num_workers（当前为 0）")
        print("    - 使用内存映射加载 .npy 文件（np.load 的 mmap_mode='r'）")
    else:
        print(f"  [正常] 系统内存占用: {mem_max:.1f}%")

    # CPU 分析
    if cpu_max > 90:
        print(f"  [警告] CPU 占用过高: {cpu_max:.1f}%")
        print("  建议: 数据预处理可能过重，考虑简化数据增强")
    else:
        print(f"  [正常] CPU 占用: {cpu_max:.1f}%")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batches', type=int, default=5, help='验证 batch 数量')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--max-gpu-mem-pct', type=float, default=85.0, help='显存占用阈值（%）')
    args = parser.parse_args()
    quick_train_with_monitor(num_batches=args.batches, batch_size=args.batch_size,
                            max_gpu_mem_pct=args.max_gpu_mem_pct)
