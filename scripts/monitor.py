#!/usr/bin/env python
"""
资源监控脚本：在训练运行时监控 GPU 显存、GPU 利用率、CPU 占用率、内存占用。

用法：
    python scripts/monitor.py --pid <训练进程PID> --interval 1 --output monitor_log.json
    # 或直接在训练脚本中集成：
    from trimodaldet.utils.monitor import ResourceMonitor
    monitor = ResourceMonitor(interval=1, output='monitor_log.json')
    monitor.start()
    # 训练结束后
    monitor.stop()
    monitor.print_summary()
"""
import time
import sys
import os
import json
import argparse
import psutil
import threading


class ResourceMonitor:
    """资源监控器，采集 GPU 显存、GPU 利用率、CPU 占用率、内存占用。"""

    def __init__(self, interval=1.0, output='monitor_log.json', pid=None):
        self.interval = interval
        self.output = output
        self.pid = pid
        self.running = False
        self.thread = None
        self.records = []
        self._nvml = None
        self._handle = None

    def _init_gpu(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            # 默认监控 GPU 0
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return True
        except Exception as e:
            print(f"[Monitor] GPU 监控初始化失败: {e}")
            return False

    def _get_gpu_stats(self):
        if self._nvml is None:
            return None
        try:
            mem_info = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
            util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
            return {
                'gpu_mem_used_mb': mem_info.used / 1024 / 1024,
                'gpu_mem_total_mb': mem_info.total / 1024 / 1024,
                'gpu_mem_util_pct': mem_info.used / mem_info.total * 100,
                'gpu_util_pct': util.gpu,
            }
        except Exception:
            return None

    def _get_system_stats(self):
        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        return {
            'cpu_percent': cpu_percent,
            'mem_used_mb': mem.used / 1024 / 1024,
            'mem_total_mb': mem.total / 1024 / 1024,
            'mem_percent': mem.percent,
        }

    def _get_process_stats(self):
        if self.pid is None:
            return None
        try:
            proc = psutil.Process(self.pid)
            mem_info = proc.memory_info()
            return {
                'proc_mem_rss_mb': mem_info.rss / 1024 / 1024,
                'proc_mem_vms_mb': mem_info.vms / 1024 / 1024,
                'proc_cpu_percent': proc.cpu_percent(interval=None),
            }
        except Exception:
            return None

    def _monitor_loop(self):
        self._init_gpu()
        while self.running:
            record = {'timestamp': time.time()}
            gpu_stats = self._get_gpu_stats()
            if gpu_stats:
                record.update(gpu_stats)
            record.update(self._get_system_stats())
            proc_stats = self._get_process_stats()
            if proc_stats:
                record.update(proc_stats)
            self.records.append(record)
            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print(f"[Monitor] 资源监控已启动，采样间隔 {self.interval}s，输出文件: {self.output}")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        self._save()
        if self._nvml:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
        print(f"[Monitor] 资源监控已停止，共采集 {len(self.records)} 条记录")

    def _save(self):
        output_dir = os.path.dirname(self.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(self.output, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, indent=2)

    def print_summary(self):
        if not self.records:
            print("[Monitor] 无记录")
            return

        gpu_mem_max = max(r.get('gpu_mem_used_mb', 0) for r in self.records)
        gpu_mem_avg = sum(r.get('gpu_mem_used_mb', 0) for r in self.records) / len(self.records)
        gpu_util_max = max(r.get('gpu_util_pct', 0) for r in self.records)
        gpu_util_avg = sum(r.get('gpu_util_pct', 0) for r in self.records) / len(self.records)
        mem_max = max(r.get('mem_percent', 0) for r in self.records)
        mem_avg = sum(r.get('mem_percent', 0) for r in self.records) / len(self.records)
        cpu_max = max(r.get('cpu_percent', 0) for r in self.records)
        cpu_avg = sum(r.get('cpu_percent', 0) for r in self.records) / len(self.records)

        print("\n[Monitor] 资源使用摘要")
        print(f"  采样时长: {len(self.records) * self.interval:.1f}s")
        print(f"  GPU 显存: 峰值 {gpu_mem_max:.1f} MB, 平均 {gpu_mem_avg:.1f} MB")
        print(f"  GPU 利用率: 峰值 {gpu_util_max:.1f}%, 平均 {gpu_util_avg:.1f}%")
        print(f"  系统内存: 峰值 {mem_max:.1f}%, 平均 {mem_avg:.1f}%")
        print(f"  CPU 占用: 峰值 {cpu_max:.1f}%, 平均 {cpu_avg:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='资源监控脚本')
    parser.add_argument('--pid', type=int, required=False, help='要监控的进程 PID')
    parser.add_argument('--interval', type=float, default=1.0, help='采样间隔（秒）')
    parser.add_argument('--output', type=str, default='monitor_log.json', help='输出 JSON 文件')
    parser.add_argument('--duration', type=int, default=60, help='监控时长（秒）')
    args = parser.parse_args()

    monitor = ResourceMonitor(interval=args.interval, output=args.output, pid=args.pid)
    monitor.start()
    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        pass
    monitor.stop()
    monitor.print_summary()


if __name__ == '__main__':
    main()
