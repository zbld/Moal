import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ================= 配置区域 =================
BASE_DIR = "/home/shihan/Moal/Moal/logs/adapt_ac_com_sdc_ema_auto"
DATASET = "imagenetr"
TOTAL_CLS = 200

# 任务设置 (列)
TASKS_CONFIG = {
    "T=10": {"tasks": 10, "step": 20},
    "T=5":  {"tasks": 5,  "step": 40}
}

# 实验映射 (行)
# key: 实验描述
# hidden_id: 对应的文件夹名
# ticks: (KD, Interp, Rumination) 三列的勾选状态
EXPERIMENTS = [
    {
        "name": "ACIL (Baseline)", # 表格第一行，通常引用论文数据
        "hidden_id": None,         # 如果您跑了 Baseline 请填入 ID，否则使用硬编码数值
        "ticks": ["", "", ""],
        "hardcoded_values": {"T=10": 81.21, "T=5": 83.36} # 论文原值
    },
    {
        "name": "Only KD",
        "hidden_id": 20001,
        "ticks": ["✔", "", ""],
        "hardcoded_values": None
    },
    {
        "name": "Only Weight Interp",
        "hidden_id": 20002,
        "ticks": ["", "✔", ""],
        "hardcoded_values": None
    },
    {
        "name": "KD + Rumination (No Interp)",
        "hidden_id": 20003,
        "ticks": ["✔", "", "✔"],
        "hardcoded_values": None
    },
    {
        "name": "MoAL (Ours)",
        "hidden_id": 20004, # 或者 20000
        "ticks": ["", "✔", "✔"], # 对应表格最后一行的勾选状态
        "hardcoded_values": None
    }
]
# ===========================================

def get_avg_accuracy(hidden_id, step_size):
    """读取指定 hidden_id 和 step_size 的日志并计算平均准确率"""
    if hidden_id is None:
        return None
        
    # 路径构建: dataset / hidden / 0 / step
    log_dir = os.path.join(BASE_DIR, DATASET, str(hidden_id), "0", str(step_size))
    
    # 尝试找到日志文件
    target_file = None
    if os.path.exists(log_dir):
        # 优先找名为 step 的文件
        f_path = os.path.join(log_dir, str(step_size))
        if os.path.isfile(f_path):
            target_file = f_path
        else:
            # 找第一个非代码文件
            for f in os.listdir(log_dir):
                if not f.endswith('.py') and not f.endswith('.json'):
                    target_file = os.path.join(log_dir, f)
                    break
    
    if target_file and os.path.exists(target_file):
        try:
            with open(target_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                for line in reversed(lines):
                    if "CNN top1 curve:" in line:
                        match = re.search(r"CNN top1 curve:\s*(\[.*?\])", line)
                        if match:
                            acc_list = eval(match.group(1))
                            return np.mean(acc_list)
        except:
            pass
            
    return None

def main():
    # 1. 准备数据表格
    # 列: KD, Interp, Rumination, ImgR(T=10), ImgR(T=5)
    table_data = []
    
    print(f"开始提取 ImageNet-R ({DATASET}) 的消融实验数据...")
    
    for exp in EXPERIMENTS:
        row = list(exp['ticks']) # Start with ticks
        
        # 获取 T=10 的数据
        val_t10 = None
        if exp['hardcoded_values']:
            val_t10 = exp['hardcoded_values'].get("T=10", "-")
        else:
            val = get_avg_accuracy(exp['hidden_id'], TASKS_CONFIG["T=10"]["step"])
            val_t10 = f"{val:.2f}" if val is not None else "N/A"
            print(f"  - {exp['name']} (T=10): {val_t10}")
            
        # 获取 T=5 的数据
        val_t5 = None
        if exp['hardcoded_values']:
            val_t5 = exp['hardcoded_values'].get("T=5", "-")
        else:
            val = get_avg_accuracy(exp['hidden_id'], TASKS_CONFIG["T=5"]["step"])
            val_t5 = f"{val:.2f}" if val is not None else "N/A"
            print(f"  - {exp['name']} (T=5): {val_t5}")
            
        row.append(val_t10)
        row.append(val_t5)
        table_data.append(row)

    # 2. 绘制表格
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    ax.axis('tight')
    
    # 列名
    col_labels = [
        "Knowledge\nDistillation", 
        "Weight\nInterpolation", 
        "Knowledge\nRumination", 
        "ImageNet-R\n(T=10)", 
        "ImageNet-R\n(T=5)"
    ]
    
    # 创建表格
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )
    
    # 3. 样式美化
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0) # 调整行高
    
    # 加粗最后一行的数字 (Ours)
    for col_idx in range(len(col_labels)):
        cell = table[(len(table_data), col_idx)] # (行索引从1开始算header, 所以最后一行是 len)
        cell.set_text_props(weight='bold')

    # 设置标题
    plt.title("Table 2. Ablation study on ImageNet-R (Average Incremental Accuracy %)", y=1.1, fontsize=12)
    
    output_filename = "ablation_study_imagenetr.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n表格已生成: {output_filename}")

if __name__ == "__main__":
    main()