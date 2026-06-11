"""
Monitor utilities for training diagnostics and optimization recommendations.
"""
import os


def analyze_and_recommend(records, batch_size=None, backbone=None):
    """
    Analyze monitor records and print optimization recommendations.

    Args:
        records: List of resource monitor records
        batch_size: Current batch size for contextual recommendations
        backbone: Current backbone type for contextual recommendations
    """
    if not records:
        print("[Monitor] No records available for analysis.")
        return

    gpu_mem_max = max(r.get('gpu_mem_used_mb', 0) for r in records)
    gpu_mem_total = max(r.get('gpu_mem_total_mb', 1) for r in records)
    gpu_mem_pct = gpu_mem_max / gpu_mem_total * 100 if gpu_mem_total > 0 else 0
    gpu_util_max = max(r.get('gpu_util_pct', 0) for r in records)
    gpu_util_avg = sum(r.get('gpu_util_pct', 0) for r in records) / len(records)
    mem_max = max(r.get('mem_percent', 0) for r in records)
    cpu_max = max(r.get('cpu_percent', 0) for r in records)

    print("\n=== 监控摘要 ===")
    print(f"  采样点数: {len(records)}")
    print(f"  GPU 显存: 峰值 {gpu_mem_max:.0f} / {gpu_mem_total:.0f} MB ({gpu_mem_pct:.1f}%)")
    print(f"  GPU 利用率: 峰值 {gpu_util_max:.1f}%, 平均 {gpu_util_avg:.1f}%")
    print(f"  系统内存: 峰值 {mem_max:.1f}%")
    print(f"  CPU 占用: 峰值 {cpu_max:.1f}%")

    print("\n=== 优化建议 ===")

    # GPU 显存分析
    if gpu_mem_pct > 85:
        print(f"  [警告] GPU 显存占用过高: {gpu_mem_max:.0f} / {gpu_mem_total:.0f} MB ({gpu_mem_pct:.1f}%)")
        print("  建议:")
        if batch_size and batch_size > 1:
            print(f"    - 减小 batch_size（当前 {batch_size} -> {batch_size // 2} 或更小）")
        else:
            print("    - 减小 batch_size（建议从 16 -> 8 -> 4 逐步尝试）")
        if backbone and backbone != 'mit_b0':
            print(f"    - 使用更小的 backbone（当前 {backbone} -> mit_b0）")
        print("    - 启用梯度累积（gradient accumulation）保持等效 batch size")
        print("    - 在训练脚本中加入 torch.cuda.empty_cache() 定期清理")
    elif gpu_mem_pct > 70:
        print(f"  [注意] GPU 显存占用较高: {gpu_mem_max:.0f} / {gpu_mem_total:.0f} MB ({gpu_mem_pct:.1f}%)")
        print("  建议:")
        print("    - 可略微减小 batch_size 以留余量")
        print("    - 监控后续训练是否稳定")
    else:
        print(f"  [正常] GPU 显存占用: {gpu_mem_max:.0f} / {gpu_mem_total:.0f} MB ({gpu_mem_pct:.1f}%)")
        print("  建议: 显存充足，可尝试增大 batch_size 提升训练效率")

    # GPU 利用率分析
    if gpu_util_avg < 50:
        print(f"  [警告] GPU 利用率偏低: 平均 {gpu_util_avg:.1f}%")
        print("  建议:")
        if batch_size:
            print(f"    - 增大 batch_size（当前 {batch_size}）")
        else:
            print("    - 增大 batch_size")
        print("    - 检查数据加载是否为瓶颈（考虑增加 num_workers）")
        print("    - 使用 torch.backends.cudnn.benchmark = True")
    elif gpu_util_avg < 80:
        print(f"  [注意] GPU 利用率一般: 平均 {gpu_util_avg:.1f}%")
        print("  建议: 可尝试微调 batch_size 或优化数据加载")
    else:
        print(f"  [正常] GPU 利用率良好: 平均 {gpu_util_avg:.1f}%")

    # 系统内存分析
    if mem_max > 90:
        print(f"  [警告] 系统内存占用过高: {mem_max:.1f}%")
        print("  建议:")
        print("    - 检查数据加载是否一次性加载过多图像到内存")
        print("    - 减小 DataLoader 的 num_workers")
        print("    - 使用内存映射加载 .npy 文件（np.load 的 mmap_mode='r'）")
    else:
        print(f"  [正常] 系统内存占用: {mem_max:.1f}%")

    # CPU 分析
    if cpu_max > 90:
        print(f"  [警告] CPU 占用过高: {cpu_max:.1f}%")
        print("  建议: 数据预处理可能过重，考虑简化数据增强")
    else:
        print(f"  [正常] CPU 占用: {cpu_max:.1f}%")


def get_gpu_memory_stats():
    """
    Get current GPU memory statistics using torch.cuda.

    Returns:
        dict with 'allocated_mb', 'reserved_mb', 'total_mb', 'util_pct'
    """
    import torch
    if not torch.cuda.is_available():
        return None
    allocated = torch.cuda.memory_allocated() / 1024 / 1024
    reserved = torch.cuda.memory_reserved() / 1024 / 1024
    total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
    return {
        'allocated_mb': allocated,
        'reserved_mb': reserved,
        'total_mb': total,
        'util_pct': allocated / total * 100,
    }


__all__ = ['analyze_and_recommend', 'get_gpu_memory_stats']
