import os
import re
import matplotlib.pyplot as plt
import numpy as np

# ================= 配置区域 =================
# 日志根目录
BASE_DIR = "/home/shihan/Moal/Moal/logs/adapt_ac_com_sdc_ema_auto"

# 数据集配置：名称, 总类别数, 任务数列表(T), Hidden层大小
DATASET_CONFIGS = {
    "cifar224": {
        "total_cls": 100,
        "tasks": [20, 10, 5], # 步长: 5, 10, 20
        "label": "CIFAR-100",
        "hidden": 5000
    },
    "imageneta": {
        "total_cls": 200,
        "tasks": [20, 10, 5], # 步长: 10, 20, 40
        "label": "ImageNet-A",
        "hidden": 15000
    },
    "imagenetr": {
        "total_cls": 200,
        "tasks": [20, 10, 5], # 步长: 10, 20, 40
        "label": "ImageNet-R",
        "hidden": 20000
    },
    "omnibenchmark": {
        "total_cls": 300,
        "tasks": [30, 15, 10], # 步长: 10, 20, 30
        "label": "OmniBenchmark",
        "hidden": 10000
    }
}
# ===========================================

def parse_log_file(file_path):
    """从日志文件中解析准确率曲线"""
    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            # 倒序查找最后一次出现的完整曲线
            for line in reversed(lines):
                if "CNN top1 curve:" in line:
                    match = re.search(r"CNN top1 curve:\s*(\[.*?\])", line)
                    if match:
                        return eval(match.group(1))
    except Exception as e:
        print(f"解析错误 {file_path}: {e}")
    return None

def find_log_path(dataset_name, hidden_size, step_size):
    """
    根据指定的 hidden_size 和 step_size 寻找日志文件
    路径结构: BASE_DIR / dataset / hidden_size / 0 / step_size
    """
    # 构建精确的目标目录
    target_dir = os.path.join(BASE_DIR, dataset_name, str(hidden_size), "0", str(step_size))
    
    if not os.path.exists(target_dir):
        # 尝试打印一下，方便调试
        # print(f"目录不存在: {target_dir}")
        return None

    # 1. 优先尝试直接以 step_size 命名的文件 (例如 '40')
    target_file_by_name = os.path.join(target_dir, str(step_size))
    if os.path.exists(target_file_by_name) and os.path.isfile(target_file_by_name):
        return target_file_by_name
        
    # 2. 如果没有，遍历目录下任何非 .py/.json 的文件作为日志
    for f in os.listdir(target_dir):
        full_path = os.path.join(target_dir, f)
        if os.path.isfile(full_path):
            if not f.endswith('.py') and not f.endswith('.json') and not f.endswith('.pyc'):
                return full_path
                
    return None

def main():
    # 创建 2x2 的子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # 蓝, 橙, 绿
    markers = ['o', 's', '^']

    for idx, (ds_name, config) in enumerate(DATASET_CONFIGS.items()):
        ax = axes[idx]
        total_cls = config['total_cls']
        tasks = config['tasks']
        label_name = config['label']
        hidden_size = config['hidden']
        
        print(f"正在处理数据集: {label_name} (Hidden={hidden_size})...")
        
        has_data_in_subplot = False
        
        for t_idx, t in enumerate(tasks):
            # 计算步长 (Increment) = Total / T
            # 例如 ImageNet-R (200类) / 20个任务 = 每次10类
            step_size = int(total_cls / t)
            
            log_file = find_log_path(ds_name, hidden_size, step_size)
            
            if log_file:
                accuracies = parse_log_file(log_file)
                if accuracies:
                    has_data_in_subplot = True
                    # 生成 x 轴 (0 到 T-1)
                    x_axis = range(len(accuracies))
                    avg_acc = np.mean(accuracies)
                    
                    print(f"  - [T={t}, Step={step_size}] 找到日志: {os.path.basename(log_file)}, Avg Acc={avg_acc:.2f}%")
                    
                    ax.plot(x_axis, accuracies, 
                            label=f"T={t} (Avg: {avg_acc:.2f}%)",
                            color=colors[t_idx % 3],
                            marker=markers[t_idx % 3],
                            linewidth=2,
                            markersize=6)
                else:
                    print(f"  - [T={t}, Step={step_size}] 找到文件但解析失败: {log_file}")
            else:
                print(f"  - [T={t}, Step={step_size}] 未找到日志目录或文件")

        # 子图设置
        ax.set_title(label_name, fontsize=14, fontweight='bold')
        ax.set_xlabel("Task ID", fontsize=12)
        ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        if has_data_in_subplot:
            ax.legend(fontsize=10)
        else:
            ax.text(0.5, 0.5, "No Data Found", ha='center', va='center', transform=ax.transAxes)

    plt.suptitle("MoAL Reproduction Results (Accuracy Curves)", fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    
    output_filename = 'moal_reproduce_results_fixed.png'
    plt.savefig(output_filename, dpi=300)
    print(f"\n绘图完成，已保存为: {output_filename}")
    # plt.show() 

if __name__ == "__main__":
    main()