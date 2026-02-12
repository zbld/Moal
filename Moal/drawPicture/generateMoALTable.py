import os
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# ================= 配置区域 =================
# 日志根目录
BASE_DIR = "/home/runner/work/Moal/Moal/logs/adapt_ac_com_sdc_ema_auto"

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
    """收集所有数据集的MoAL数据"""
    results = {}
    
    for ds_name, config in DATASET_CONFIGS.items():
        total_cls = config['total_cls']
        tasks = config['tasks']
        label_name = config['label']
        hidden_size = config['hidden']
        
        print(f"正在处理数据集: {label_name} (Hidden={hidden_size})...")
        
        results[label_name] = {
            'avg_acc': {},
            'last_acc': {}
        }
        
        for t in tasks:
            step_size = int(total_cls / t)
            log_file = find_log_path(ds_name, hidden_size, step_size)
            
            if log_file:
                accuracies = parse_log_file(log_file)
                if accuracies:
                    avg_acc, last_acc = compute_metrics(accuracies)
                    results[label_name]['avg_acc'][t] = avg_acc
                    results[label_name]['last_acc'][t] = last_acc
                    print(f"  - T={t}: Ā={avg_acc:.2f}%, AT={last_acc:.2f}%")
                else:
                    results[label_name]['avg_acc'][t] = None
                    results[label_name]['last_acc'][t] = None
                    print(f"  - T={t}: 解析失败")
            else:
                results[label_name]['avg_acc'][t] = None
                results[label_name]['last_acc'][t] = None
                print(f"  - T={t}: 未找到日志")
    
    return results

def generate_table_image(results):
    """生成表格图像"""
    # 设置图表参数
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # 准备表格数据
    datasets = list(results.keys())
    
    # 表头
    header = ['Metric', 'Methods']
    for ds in datasets:
        # 获取该数据集的任务数
        tasks_list = list(results[ds]['avg_acc'].keys())
        for t in sorted(tasks_list, reverse=True):  # 降序排列: 20, 10, 5 或 30, 15, 10
            header.append(f'T={t}')
        header.append('avg.')
    
    # 数据行
    rows = []
    
    # Ā (平均增量准确率) 行
    avg_row = ['Ā', 'MoAL (Ours)']
    for ds in datasets:
        tasks_list = sorted(results[ds]['avg_acc'].keys(), reverse=True)
        ds_values = []
        for t in tasks_list:
            val = results[ds]['avg_acc'][t]
            if val is not None:
                avg_row.append(f'{val:.2f}')
                ds_values.append(val)
            else:
                avg_row.append('-')
        # 计算该数据集的平均值
        if ds_values:
            avg_row.append(f'{np.mean(ds_values):.2f}')
        else:
            avg_row.append('-')
    rows.append(avg_row)
    
    # AT (最后任务准确率) 行
    last_row = ['AT', 'MoAL (Ours)']
    for ds in datasets:
        tasks_list = sorted(results[ds]['last_acc'].keys(), reverse=True)
        ds_values = []
        for t in tasks_list:
            val = results[ds]['last_acc'][t]
            if val is not None:
                last_row.append(f'{val:.2f}')
                ds_values.append(val)
            else:
                last_row.append('-')
        # 计算该数据集的平均值
        if ds_values:
            last_row.append(f'{np.mean(ds_values):.2f}')
        else:
            last_row.append('-')
    rows.append(last_row)
    
    # 创建表格
    table_data = [header] + rows
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.08, 0.12] + [0.06] * (len(header) - 2))
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # 设置表头样式
    for i in range(len(header)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
    
    # 设置第一列样式（Metric列）
    for i in range(1, len(table_data)):
        cell = table[(i, 0)]
        cell.set_facecolor('#E7E6E6')
        cell.set_text_props(weight='bold')
    
    # 设置第二列样式（Methods列）
    for i in range(1, len(table_data)):
        cell = table[(i, 1)]
        cell.set_facecolor('#F2F2F2')
    
    plt.title('Table 1. MoAL Performance Metrics', fontsize=14, fontweight='bold', pad=20)
    
    output_filename = 'moal_table_results.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n表格已保存为: {output_filename}")
    
    return table_data

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
    results = collect_moal_data()
    
    print("\n生成表格图像...")
    table_data = generate_table_image(results)
    
    print_table_text(table_data)
    
    print("\n任务完成！")

if __name__ == "__main__":
    main()
