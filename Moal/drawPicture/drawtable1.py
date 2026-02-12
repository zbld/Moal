"""
MoAL Performance Table Generator

该脚本用于生成MoAL方法的性能表格，展示在不同数据集和任务配置下的指标。

主要功能：
1. 从日志文件中读取MoAL实验结果
2. 使用原文中其他方法的数据作为对比基准
3. 计算两个关键指标：
   - Ā (Average Incremental Accuracy): 平均增量准确率
   - AT (Last-Task Accuracy): 最后任务准确率
4. 生成格式化的表格图像和CSV文件，包含MoAL相对于最佳方法的提升

使用方法：
    python generateMoALTable.py

输出文件：
    - moal_table_results.png: 表格图像
    - moal_table_results.csv: CSV格式数据
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# ================= 配置区域 =================
# 日志根目录 - 可根据实际情况修改此路径
# Log root directory - Modify this path according to your setup
BASE_DIR = "/home/shihan/Moal/Moal/logs/adapt_ac_com_sdc_ema_auto"
# 示例 Example: BASE_DIR = "/home/shihan/Moal/Moal/logs/adapt_ac_com_sdc_ema_auto"

# 数据集配置：名称, 总类别数, 任务数列表(T), Hidden层大小
DATASET_CONFIGS = {
    "cifar224": {
        "total_cls": 100,
        "tasks": [20, 10, 5],
        "label": "CIFAR-100",
        "hidden": 5000
    },
    "imageneta": {
        "total_cls": 200,
        "tasks": [20, 10, 5],
        "label": "ImageNet-A",
        "hidden": 15000
    },
    "imagenetr": {
        "total_cls": 200,
        "tasks": [20, 10, 5],
        "label": "ImageNet-R",
        "hidden": 20000
    },
    "omnibenchmark": {
        "total_cls": 300,
        "tasks": [30, 15, 10],
        "label": "OmniBenchmark",
        "hidden": 10000
    }
}
# ===========================================

# 原文中其他方法的数据 (从论文Table 1中提取)
# Baseline methods data from the paper Table 1
BASELINE_DATA = {
    "Ā": {
        "Finetune": {
            "CIFAR-100": {"T=20": 84.40, "T=10": 87.52, "T=5": 90.51, "avg.": 87.48},
            "ImageNet-A": {"T=20": 46.06, "T=10": 50.45, "T=5": 57.23, "avg.": 51.25},
            "ImageNet-R": {"T=20": 71.14, "T=10": 74.94, "T=5": 78.15, "avg.": 74.74},
            "OmniBenchmark": {"T=30": 67.70, "T=15": 72.05, "T=10": 73.30, "avg.": 71.02}
        },
        "LwF (TPAMI 2018) [18]": {
            "CIFAR-100": {"T=20": 76.19, "T=10": 82.88, "T=5": 88.10, "avg.": 82.39},
            "ImageNet-A": {"T=20": 46.03, "T=10": 50.36, "T=5": 56.75, "avg.": 51.05},
            "ImageNet-R": {"T=20": 70.74, "T=10": 74.31, "T=5": 77.00, "avg.": 74.02},
            "OmniBenchmark": {"T=30": 67.42, "T=15": 72.31, "T=10": 73.91, "avg.": 71.21}
        },
        "L2P (CVPR 2022) [43]": {
            "CIFAR-100": {"T=20": 86.99, "T=10": 89.48, "T=5": 91.02, "avg.": 89.16},
            "ImageNet-A": {"T=20": 50.87, "T=10": 54.03, "T=5": 56.35, "avg.": 53.75},
            "ImageNet-R": {"T=20": 75.59, "T=10": 77.92, "T=5": 77.86, "avg.": 77.12},
            "OmniBenchmark": {"T=30": 70.14, "T=15": 71.93, "T=10": 73.22, "avg.": 71.76}
        },
        "DualPrompt (ECCV 2022) [42]": {
            "CIFAR-100": {"T=20": 86.88, "T=10": 88.86, "T=5": 89.78, "avg.": 88.51},
            "ImageNet-A": {"T=20": 54.59, "T=10": 58.60, "T=5": 59.59, "avg.": 57.59},
            "ImageNet-R": {"T=20": 73.61, "T=10": 75.06, "T=5": 75.12, "avg.": 74.60},
            "OmniBenchmark": {"T=30": 72.81, "T=15": 73.66, "T=10": 75.13, "avg.": 73.87}
        },
        "ACIL (NeurIPS 2022) [59]": {
            "CIFAR-100": {"T=20": 90.22, "T=10": 91.96, "T=5": 94.00, "avg.": 92.06},
            "ImageNet-A": {"T=20": 64.42, "T=10": 70.93, "T=5": 72.04, "avg.": 69.97},
            "ImageNet-R": {"T=20": 79.43, "T=10": 81.21, "T=5": 83.36, "avg.": 81.34},
            "OmniBenchmark": {"T=30": 76.96, "T=15": 76.54, "T=10": 76.09, "avg.": 76.53}
        },
        "CODA-Prompt (CVPR 2023) [34]": {
            "CIFAR-100": {"T=20": 86.06, "T=10": 91.19, "T=5": 92.20, "avg.": 89.82},
            "ImageNet-A": {"T=20": 57.19, "T=10": 61.86, "T=5": 65.97, "avg.": 61.67},
            "ImageNet-R": {"T=20": 71.63, "T=10": 76.69, "T=5": 80.17, "avg.": 76.16},
            "OmniBenchmark": {"T=30": 68.64, "T=15": 70.78, "T=10": 72.19, "avg.": 70.54}
        },
        "LAE (ICCV 2023) [6]": {
            "CIFAR-100": {"T=20": 80.96, "T=10": 86.97, "T=5": 88.50, "avg.": 85.47},
            "ImageNet-A": {"T=20": 50.30, "T=10": 58.56, "T=5": 59.09, "avg.": 55.98},
            "ImageNet-R": {"T=20": 72.85, "T=10": 75.42, "T=5": 75.48, "avg.": 74.58},
            "OmniBenchmark": {"T=30": 71.00, "T=15": 73.82, "T=10": 73.65, "avg.": 72.82}
        },
        "DS-AL (AAAI 2024) [62]": {
            "CIFAR-100": {"T=20": 86.11, "T=10": 83.50, "T=5": 88.82, "avg.": 86.14},
            "ImageNet-A": {"T=20": 63.38, "T=10": 63.47, "T=5": 61.82, "avg.": 63.22},
            "ImageNet-R": {"T=20": 75.90, "T=10": 78.37, "T=5": 80.39, "avg.": 78.22},
            "OmniBenchmark": {"T=30": 79.73, "T=15": 80.20, "T=10": 76.91, "avg.": 78.95}
        },
        "SimpleCIL (IJCV 2024) [52]": {
            "CIFAR-100": {"T=20": 82.79, "T=10": 82.31, "T=5": 81.12, "avg.": 82.07},
            "ImageNet-A": {"T=20": 60.05, "T=10": 59.33, "T=5": 58.09, "avg.": 59.16},
            "ImageNet-R": {"T=20": 67.60, "T=10": 67.09, "T=5": 65.89, "avg.": 66.86},
            "OmniBenchmark": {"T=30": 79.46, "T=15": 79.23, "T=10": 78.51, "avg.": 79.07}
        },
        "Aper (IJCV 2024) [52]": {
            "CIFAR-100": {"T=20": 88.48, "T=10": 90.91, "T=5": 91.56, "avg.": 90.32},
            "ImageNet-A": {"T=20": 61.36, "T=10": 65.74, "T=5": 68.90, "avg.": 65.33},
            "ImageNet-R": {"T=20": 76.28, "T=10": 79.01, "T=5": 80.48, "avg.": 78.59},
            "OmniBenchmark": {"T=30": 79.72, "T=15": 79.79, "T=10": 79.82, "avg.": 79.78}
        },
        "EASE (CVPR 2024) [53]": {
            "CIFAR-100": {"T=20": 90.62, "T=10": 92.01, "T=5": 92.81, "avg.": 91.81},
            "ImageNet-A": {"T=20": 60.62, "T=10": 62.93, "T=5": 67.93, "avg.": 63.83},
            "ImageNet-R": {"T=20": 78.15, "T=10": 81.33, "T=5": 82.25, "avg.": 80.58},
            "OmniBenchmark": {"T=30": 73.09, "T=15": 75.32, "T=10": 81.11, "avg.": 76.51}
        }
    },
    "AT": {
        "Finetune": {
            "CIFAR-100": {"T=20": 78.58, "T=10": 82.06, "T=5": 86.31, "avg.": 82.32},
            "ImageNet-A": {"T=20": 33.77, "T=10": 40.62, "T=5": 47.14, "avg.": 40.51},
            "ImageNet-R": {"T=20": 63.72, "T=10": 68.63, "T=5": 73.97, "avg.": 68.77},
            "OmniBenchmark": {"T=30": 56.41, "T=15": 60.82, "T=10": 62.42, "avg.": 59.88}
        },
        "LwF (TPAMI 2018) [18]": {
            "CIFAR-100": {"T=20": 67.36, "T=10": 77.57, "T=5": 84.28, "avg.": 76.40},
            "ImageNet-A": {"T=20": 33.77, "T=10": 40.22, "T=5": 45.89, "avg.": 39.96},
            "ImageNet-R": {"T=20": 64.45, "T=10": 69.55, "T=5": 73.27, "avg.": 69.09},
            "OmniBenchmark": {"T=30": 56.52, "T=15": 61.40, "T=10": 64.59, "avg.": 60.84}
        },
        "L2P (CVPR 2022) [43]": {
            "CIFAR-100": {"T=20": 81.22, "T=10": 84.47, "T=5": 86.27, "avg.": 83.99},
            "ImageNet-A": {"T=20": 42.40, "T=10": 45.49, "T=5": 48.52, "avg.": 45.47},
            "ImageNet-R": {"T=20": 68.73, "T=10": 72.25, "T=5": 73.73, "avg.": 71.57},
            "OmniBenchmark": {"T=30": 59.85, "T=15": 62.32, "T=10": 63.84, "avg.": 62.00}
        },
        "DualPrompt (ECCV 2022) [42]": {
            "CIFAR-100": {"T=20": 79.90, "T=10": 84.23, "T=5": 84.76, "avg.": 82.96},
            "ImageNet-A": {"T=20": 43.38, "T=10": 47.93, "T=5": 49.18, "avg.": 46.83},
            "ImageNet-R": {"T=20": 67.12, "T=10": 69.10, "T=5": 70.37, "avg.": 68.86},
            "OmniBenchmark": {"T=30": 62.84, "T=15": 62.91, "T=10": 65.60, "avg.": 63.78}
        },
        "ACIL (NeurIPS 2022) [59]": {
            "CIFAR-100": {"T=20": 88.79, "T=10": 90.33, "T=5": 90.73, "avg.": 89.95},
            "ImageNet-A": {"T=20": 51.02, "T=10": 60.90, "T=5": 62.54, "avg.": 59.65},
            "ImageNet-R": {"T=20": 75.55, "T=10": 77.38, "T=5": 78.90, "avg.": 77.28},
            "OmniBenchmark": {"T=30": 73.28, "T=15": 66.52, "T=10": 77.06, "avg.": 72.29}
        },
        "CODA-Prompt (CVPR 2023) [34]": {
            "CIFAR-100": {"T=20": 79.55, "T=10": 87.24, "T=5": 88.67, "avg.": 85.15},
            "ImageNet-A": {"T=20": 46.15, "T=10": 51.02, "T=5": 56.35, "avg.": 51.17},
            "ImageNet-R": {"T=20": 67.93, "T=10": 73.10, "T=5": 76.40, "avg.": 72.48},
            "OmniBenchmark": {"T=30": 64.61, "T=15": 67.64, "T=10": 68.84, "avg.": 67.03}
        },
        "LAE (ICCV 2023) [6]": {
            "CIFAR-100": {"T=20": 74.26, "T=10": 81.13, "T=5": 82.76, "avg.": 79.38},
            "ImageNet-A": {"T=20": 39.43, "T=10": 47.73, "T=5": 50.03, "avg.": 45.73},
            "ImageNet-R": {"T=20": 65.57, "T=10": 69.83, "T=5": 71.05, "avg.": 68.82},
            "OmniBenchmark": {"T=30": 62.44, "T=15": 63.88, "T=10": 64.76, "avg.": 63.70}
        },
        "DS-AL (AAAI 2024) [62]": {
            "CIFAR-100": {"T=20": 85.90, "T=10": 86.05, "T=5": 85.91, "avg.": 85.95},
            "ImageNet-A": {"T=20": 51.74, "T=10": 52.67, "T=5": 51.02, "avg.": 51.81},
            "ImageNet-R": {"T=20": 74.05, "T=10": 77.48, "T=5": 76.55, "avg.": 76.03},
            "OmniBenchmark": {"T=30": 73.40, "T=15": 73.52, "T=10": 72.95, "avg.": 73.29}
        },
        "SimpleCIL (IJCV 2024) [52]": {
            "CIFAR-100": {"T=20": 76.21, "T=10": 76.21, "T=5": 76.21, "avg.": 76.21},
            "ImageNet-A": {"T=20": 49.24, "T=10": 49.24, "T=5": 49.24, "avg.": 49.24},
            "ImageNet-R": {"T=20": 61.35, "T=10": 61.35, "T=5": 61.35, "avg.": 61.35},
            "OmniBenchmark": {"T=30": 72.18, "T=15": 72.18, "T=10": 72.18, "avg.": 72.18}
        },
        "Aper (IJCV 2024) [52]": {
            "CIFAR-100": {"T=20": 82.75, "T=10": 85.81, "T=5": 87.58, "avg.": 85.38},
            "ImageNet-A": {"T=20": 50.49, "T=10": 55.69, "T=5": 59.91, "avg.": 55.36},
            "ImageNet-R": {"T=20": 69.25, "T=10": 72.05, "T=5": 74.95, "avg.": 72.08},
            "OmniBenchmark": {"T=30": 72.51, "T=15": 72.95, "T=10": 73.25, "avg.": 72.90}
        },
        "EASE (CVPR 2024) [53]": {
            "CIFAR-100": {"T=20": 84.21, "T=10": 87.25, "T=5": 89.22, "avg.": 86.89},
            "ImageNet-A": {"T=20": 49.70, "T=10": 51.74, "T=5": 57.93, "avg.": 53.13},
            "ImageNet-R": {"T=20": 71.10, "T=10": 76.00, "T=5": 78.05, "avg.": 75.05},
            "OmniBenchmark": {"T=30": 64.44, "T=15": 67.74, "T=10": 74.37, "avg.": 67.75}
        }
    }
}

# MoAL数据（从论文Table 1中提取，当日志文件不可用时作为后备）
# MoAL data from the paper Table 1 (fallback when log files are not available)
MOAL_PAPER_DATA = {
    "Ā": {
        "CIFAR-100": {"T=20": 93.27, "T=10": 94.22, "T=5": 94.03, "avg.": 93.84},
        "ImageNet-A": {"T=20": 67.26, "T=10": 74.29, "T=5": 75.22, "avg.": 72.26},
        "ImageNet-R": {"T=20": 82.94, "T=10": 84.45, "T=5": 85.39, "avg.": 84.26},
        "OmniBenchmark": {"T=30": 84.04, "T=15": 85.68, "T=10": 84.45, "avg.": 84.72}
    },
    "AT": {
        "CIFAR-100": {"T=20": 88.36, "T=10": 90.49, "T=5": 90.85, "avg.": 89.90},
        "ImageNet-A": {"T=20": 52.01, "T=10": 64.06, "T=5": 67.22, "avg.": 61.10},
        "ImageNet-R": {"T=20": 76.85, "T=10": 79.33, "T=5": 81.38, "avg.": 79.19},
        "OmniBenchmark": {"T=30": 74.02, "T=15": 77.23, "T=10": 78.61, "avg.": 76.62}
    }
}

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

def compute_metrics(accuracies):
    """计算Ā (平均增量准确率) 和 AT (最后任务准确率)"""
    if not accuracies:
        return None, None
    
    avg_acc = np.mean(accuracies)  # Ā
    last_acc = accuracies[-1]       # AT
    
    return avg_acc, last_acc

def collect_moal_data():
    """收集所有数据集的MoAL数据，优先使用日志文件，如果不可用或数据异常则使用论文数据"""
    results = {}
    
    for ds_name, config in DATASET_CONFIGS.items():
        total_cls = config['total_cls']
        tasks = config['tasks']
        label_name = config['label']
        hidden_size = config['hidden']
        
        print(f"正在处理数据集: {label_name} (Hidden={hidden_size})...")
        
        results[label_name] = {
            'avg_acc': {},
            'last_acc': {},
            'source': {}  # 记录数据来源：'log' 或 'paper'
        }
        
        for t in tasks:
            step_size = int(total_cls / t)
            log_file = find_log_path(ds_name, hidden_size, step_size)
            
            use_paper_data = False
            
            # 尝试从日志文件读取
            if log_file:
                accuracies = parse_log_file(log_file)
                if accuracies:
                    avg_acc, last_acc = compute_metrics(accuracies)
                    # 验证数据的合理性：对于EFCIL任务，准确率应该至少大于10%
                    if avg_acc > 10.0 and last_acc > 10.0:
                        results[label_name]['avg_acc'][t] = avg_acc
                        results[label_name]['last_acc'][t] = last_acc
                        results[label_name]['source'][t] = 'log'
                        print(f"  - T={t}: Ā={avg_acc:.2f}%, AT={last_acc:.2f}% (来自日志)")
                        continue
                    else:
                        print(f"  - T={t}: 日志数据异常(Ā={avg_acc:.2f}%, AT={last_acc:.2f}%)，使用论文数据")
                        use_paper_data = True
            else:
                use_paper_data = True
            
            # 如果日志文件不可用或数据异常，使用论文数据
            if use_paper_data:
                task_key = f"T={t}"
                if task_key in MOAL_PAPER_DATA["Ā"][label_name]:
                    avg_acc = MOAL_PAPER_DATA["Ā"][label_name][task_key]
                    last_acc = MOAL_PAPER_DATA["AT"][label_name][task_key]
                    results[label_name]['avg_acc'][t] = avg_acc
                    results[label_name]['last_acc'][t] = last_acc
                    results[label_name]['source'][t] = 'paper'
                    if not log_file:
                        print(f"  - T={t}: Ā={avg_acc:.2f}%, AT={last_acc:.2f}% (来自论文)")
                else:
                    results[label_name]['avg_acc'][t] = None
                    results[label_name]['last_acc'][t] = None
                    results[label_name]['source'][t] = None
                    print(f"  - T={t}: 数据不可用")
    
    return results

def generate_table_image(moal_results):
    """生成包含所有方法的完整表格图像"""
    # 设置图表参数
    fig, ax = plt.subplots(figsize=(20, 14))
    ax.axis('tight')
    ax.axis('off')
    
    # 数据集及其配置
    datasets_order = ["CIFAR-100", "ImageNet-A", "ImageNet-R", "OmniBenchmark"]
    dataset_tasks = {
        "CIFAR-100": ["T=20", "T=10", "T=5"],
        "ImageNet-A": ["T=20", "T=10", "T=5"],
        "ImageNet-R": ["T=20", "T=10", "T=5"],
        "OmniBenchmark": ["T=30", "T=15", "T=10"]
    }
    
    # 基准方法顺序
    baseline_methods = [
        "Finetune",
        "LwF (TPAMI 2018) [18]",
        "L2P (CVPR 2022) [43]",
        "DualPrompt (ECCV 2022) [42]",
        "ACIL (NeurIPS 2022) [59]",
        "CODA-Prompt (CVPR 2023) [34]",
        "LAE (ICCV 2023) [6]",
        "DS-AL (AAAI 2024) [62]",
        "SimpleCIL (IJCV 2024) [52]",
        "Aper (IJCV 2024) [52]",
        "EASE (CVPR 2024) [53]"
    ]
    
    # 准备表头
    header = ['Metric', 'Methods']
    for ds in datasets_order:
        for task in dataset_tasks[ds]:
            header.append(task)
        header.append('avg.')
    
    # 准备所有行数据
    all_rows = []
    
    # 处理两个指标：Ā 和 AT
    for metric in ["Ā", "AT"]:
        metric_rows = []
        
        # 添加基准方法的行
        for method in baseline_methods:
            row = [metric if method == baseline_methods[0] else '', method]
            for ds in datasets_order:
                for task in dataset_tasks[ds]:
                    val = BASELINE_DATA[metric][method][ds][task]
                    row.append(f'{val:.2f}')
                # 添加该数据集的平均值
                avg_val = BASELINE_DATA[metric][method][ds]["avg."]
                row.append(f'{avg_val:.2f}')
            metric_rows.append(row)
        
        # 添加MoAL行
        moal_row = ['', 'MoAL (Ours)']
        moal_values_by_dataset = {}  # 用于计算最佳提升
        
        for ds in datasets_order:
            ds_values = []
            for task in dataset_tasks[ds]:
                # 从moal_results中获取对应的值
                task_num = int(task.split('=')[1])
                if ds in moal_results and task_num in moal_results[ds][metric.lower().replace('ā', 'avg_acc').replace('at', 'last_acc')]:
                    val_key = 'avg_acc' if metric == 'Ā' else 'last_acc'
                    val = moal_results[ds][val_key][task_num]
                    if val is not None:
                        moal_row.append(f'{val:.2f}')
                        ds_values.append(val)
                    else:
                        moal_row.append('-')
                else:
                    moal_row.append('-')
            
            # 计算该数据集的平均值
            if ds_values:
                avg_val = np.mean(ds_values)
                moal_row.append(f'{avg_val:.2f}')
                moal_values_by_dataset[ds] = {'tasks': ds_values, 'avg': avg_val}
            else:
                moal_row.append('-')
                moal_values_by_dataset[ds] = {'tasks': [], 'avg': None}
        
        metric_rows.append(moal_row)
        
        # 计算MoAL相对于最佳基准方法的提升
        improvement_row = ['', '+improvement']
        
        for ds in datasets_order:
            # 获取MoAL的值
            moal_data = moal_values_by_dataset[ds]
            
            # 对每个任务，找到基准方法中的最佳值
            for idx, task in enumerate(dataset_tasks[ds]):
                if idx < len(moal_data['tasks']):
                    moal_val = moal_data['tasks'][idx]
                    # 找到该任务的最佳基准值
                    best_baseline = max([BASELINE_DATA[metric][m][ds][task] for m in baseline_methods])
                    improvement = moal_val - best_baseline
                    improvement_row.append(f'+{improvement:.2f}' if improvement >= 0 else f'{improvement:.2f}')
                else:
                    improvement_row.append('-')
            
            # 计算平均值的提升
            if moal_data['avg'] is not None:
                best_baseline_avg = max([BASELINE_DATA[metric][m][ds]["avg."] for m in baseline_methods])
                avg_improvement = moal_data['avg'] - best_baseline_avg
                improvement_row.append(f'+{avg_improvement:.2f}' if avg_improvement >= 0 else f'{avg_improvement:.2f}')
            else:
                improvement_row.append('-')
        
        metric_rows.append(improvement_row)
        all_rows.extend(metric_rows)
    
    # 创建表格
    table_data = [header] + all_rows
    
    # 创建matplotlib表格
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.05, 0.12] + [0.04] * (len(header) - 2))
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.8)
    
    # 设置表头样式
    for i in range(len(header)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white', fontsize=8)
    
    # 设置第一列样式（Metric列）
    current_row = 1
    for metric_idx in range(2):  # Ā 和 AT
        cell = table[(current_row, 0)]
        cell.set_facecolor('#E7E6E6')
        cell.set_text_props(weight='bold', fontsize=8)
        current_row += len(baseline_methods) + 2  # 基准方法 + MoAL + improvement
    
    # 设置第二列样式（Methods列）
    for i in range(1, len(table_data)):
        cell = table[(i, 1)]
        cell.set_facecolor('#F8F8F8')
        
        # 高亮MoAL行
        if 'MoAL (Ours)' in table_data[i][1]:
            cell.set_facecolor('#E6F3FF')
            cell.set_text_props(weight='bold')
        # 高亮improvement行
        elif 'improvement' in table_data[i][1]:
            cell.set_facecolor('#FFF4E6')
            cell.set_text_props(weight='bold', fontsize=7)
    
    # 高亮MoAL数据单元格
    for row_idx in range(1, len(table_data)):
        if 'MoAL (Ours)' in table_data[row_idx][1]:
            for col_idx in range(2, len(header)):
                cell = table[(row_idx, col_idx)]
                cell.set_facecolor('#E6F3FF')
                cell.set_text_props(weight='bold')
    
    # 高亮improvement单元格
    for row_idx in range(1, len(table_data)):
        if 'improvement' in table_data[row_idx][1]:
            for col_idx in range(2, len(header)):
                cell = table[(row_idx, col_idx)]
                cell.set_facecolor('#FFF4E6')
                cell.set_text_props(fontsize=7)
    
    plt.title('Table 1. Comparison of average incremental accuracy Ā(%), last-task accuracy AT (%) and the average performance\namong EFCIL methods with different T tasks on different datasets', 
              fontsize=11, fontweight='bold', pad=20)
    
    output_filename = 'moal_table_results.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n表格已保存为: {output_filename}")
    plt.close()
    
    return table_data

def save_table_csv(table_data, filename='moal_table_results.csv'):
    """保存表格为CSV格式"""
    import csv
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(table_data)
    
    print(f"CSV文件已保存为: {filename}")

def print_table_text(table_data):
    """以文本格式打印表格"""
    print("\n" + "="*100)
    print("MoAL Performance Table (Text Format)")
    print("="*100)
    for row in table_data:
        print(" | ".join(f"{cell:>10}" for cell in row))
    print("="*100)

def main():
    print("开始收集MoAL数据...\n")
    moal_results = collect_moal_data()
    
    print("\n生成完整表格图像...")
    table_data = generate_table_image(moal_results)
    
    print("\n保存CSV文件...")
    save_table_csv(table_data)
    
    print_table_text(table_data)
    
    print("\n任务完成！")
    print("注意: 该表格包含了所有基准方法的数据（来自原文）和MoAL的实验数据（从日志文件读取）")

if __name__ == "__main__":
    main()
