import os
import re
import matplotlib.pyplot as plt
import numpy as np

# ================= 配置区域 =================
# 日志根目录
BASE_DIR = "/home/shihan/Moal/Moal/logs/adapt_ac_com_sdc_ema_auto"

# 需要测试的 Hidden (Buffer Layer) 维度
HIDDEN_DIMS = [3000, 5000, 10000, 15000, 20000]

# 子图实验配置
# 根据图片顺序配置: CIFAR-100(T=5), CIFAR-100(T=10), ImageNet-R(T=10), ImageNet-R(T=5)
EXPERIMENTS = [
    {
        "title": "CIFAR-100 (5 tasks)",
        "dataset": "cifar224",
        "total_cls": 100,
        "tasks": 5,     # Step = 20
        "baseline": 94.00 # 黑色虚线参考值 (根据图片目测)
    },
    {
        "title": "CIFAR-100 (10 tasks)",
        "dataset": "cifar224",
        "total_cls": 100,
        "tasks": 10,    # Step = 10
        "baseline": 92.06 # 参考值
    },
    {
        "title": "ImageNet-R (10 tasks)",
        "dataset": "imagenetr",
        "total_cls": 200,
        "tasks": 10,    # Step = 20
        "baseline": 81.34 # 参考值
    },
    {
        "title": "ImageNet-R (5 tasks)",
        "dataset": "imagenetr",
        "total_cls": 200,
        "tasks": 5,     # Step = 40
        "baseline": 83.36 # 参考值
    }
]
# ===========================================

def parse_log_avg_acc(file_path):
    """解析日志并返回平均准确率"""
    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            for line in reversed(lines):
                if "CNN top1 curve:" in line:
                    match = re.search(r"CNN top1 curve:\s*(\[.*?\])", line)
                    if match:
                        acc_list = eval(match.group(1))
                        return np.mean(acc_list) # 返回平均值
    except Exception as e:
        pass
    return None

def find_log_file(dataset, hidden_dim, step_size):
    """
    构建路径: BASE_DIR / dataset / hidden_dim / 0 / step_size
    """
    target_dir = os.path.join(BASE_DIR, dataset, str(hidden_dim), "0", str(step_size))
    
    if not os.path.exists(target_dir):
        return None

    # 1. 尝试直接找名为 step_size 的文件
    target_file = os.path.join(target_dir, str(step_size))
    if os.path.isfile(target_file):
        return target_file
        
    # 2. 否则找第一个非代码文件
    for f in os.listdir(target_dir):
        full_path = os.path.join(target_dir, f)
        if os.path.isfile(full_path):
            if not f.endswith('.py') and not f.endswith('.json'):
                return full_path
    return None

def main():
    # 创建 1行4列 的画布
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # X轴刻度标签
    x_ticks_labels = [f"{int(d/1000)}k" for d in HIDDEN_DIMS]
    
    for idx, exp in enumerate(EXPERIMENTS):
        ax = axes[idx]
        dataset = exp['dataset']
        tasks = exp['tasks']
        step_size = int(exp['total_cls'] / tasks)
        
        print(f"正在处理图表 {idx+1}: {exp['title']} (Step={step_size})...")
        
        y_values = []
        x_values = [] # 记录有数据的 x 轴索引
        
        for i, hidden_dim in enumerate(HIDDEN_DIMS):
            log_file = find_log_file(dataset, hidden_dim, step_size)
            
            avg_acc = None
            if log_file:
                avg_acc = parse_log_avg_acc(log_file)
            
            if avg_acc is not None:
                y_values.append(avg_acc)
                x_values.append(i) # 记录索引位置
                print(f"  - Hidden {hidden_dim}: Acc {avg_acc:.2f}%")
            else:
                y_values.append(None) # 占位，防止错位，或者绘图时跳过
                print(f"  - Hidden {hidden_dim}: 未找到数据")

        # 过滤掉 None 值进行绘图
        valid_x = [HIDDEN_DIMS[i] for i, val in enumerate(y_values) if val is not None]
        valid_y = [val for val in y_values if val is not None]
        
        # 绘制主曲线
        if valid_y:
            # 为了让 x 轴显示等间距的 category，我们使用 range(len(HIDDEN_DIMS)) 作为 x 坐标
            # 但只绘制有效点
            plot_x = [HIDDEN_DIMS.index(vx) for vx in valid_x]
            
            ax.plot(plot_x, valid_y, 
                    color='#5b9bd5',  # 类似图中的淡蓝色
                    marker='o', 
                    linewidth=2, 
                    markersize=8,
                    label='MoAL')
            
        # 绘制 Baseline 虚线
        baseline = exp['baseline']
        ax.axhline(y=baseline, color='black', linestyle='--', linewidth=1.5, label='SOTA')
        
        # 样式设置
        ax.set_title(exp['title'], fontsize=14)
        ax.set_xlabel("Buffer Layer Dim", fontsize=12)
        if idx == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=12)
            
        # 设置 X 轴刻度为 3k, 5k ...
        ax.set_xticks(range(len(HIDDEN_DIMS)))
        ax.set_xticklabels(x_ticks_labels)
        
        ax.grid(axis='y', linestyle='-', alpha=0.3)
        
        # 设置 Y 轴范围，让图形居中好看一点 (根据数据动态调整或固定)
        if valid_y:
            min_y = min(min(valid_y), baseline)
            max_y = max(max(valid_y), baseline)
            margin = (max_y - min_y) * 0.2
            if margin == 0: margin = 1.0
            ax.set_ylim(min_y - margin, max_y + margin)

    plt.tight_layout()
    output_filename = 'moal_buffer_dim_impact.png'
    plt.savefig(output_filename, dpi=300)
    print(f"\n绘图完成，已保存为: {output_filename}")

if __name__ == "__main__":
    main()