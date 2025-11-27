import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ================= 配置区域 =================
BASE_DIR = "/home/shihan/Moal/Moal/logs/adapt_ac_com_sdc_ema_auto"
DATASET = "imagenetr"

# 实验 ID 定义与表格行的对应关系
# 格式: [ID, KD勾选?, Interp勾选?, Rumination勾选?]
# 假设 20004 是 Baseline (第一行)
# 假设 20000 是 Full Method (最后一行)
EXPERIMENTS = [
    {"id": 20004, "kd": False, "interp": False, "rum": False, "desc": "Baseline"},
    {"id": 20001, "kd": True,  "interp": False, "rum": False, "desc": "Only KD"},
    {"id": 20002, "kd": False, "interp": True,  "rum": False, "desc": "Only Interp"},
    {"id": 20003, "kd": True,  "interp": False, "rum": True,  "desc": "KD + Rumination"},
    {"id": 20000, "kd": False, "interp": True,  "rum": True,  "desc": "MoAL (Ours)"}
]

# 任务设置 (ImageNet-R)
# T=10 -> Step=20
# T=5  -> Step=40
TASKS = [
    {"t": 10, "step": 20},
    {"t": 5,  "step": 40}
]
# ===========================================

def get_avg_acc(hidden_id, step_size):
    """读取日志并计算平均准确率"""
    # 路径: logs/.../imagenetr/{ID}/0/{step}
    target_dir = os.path.join(BASE_DIR, DATASET, str(hidden_id), "0", str(step_size))
    
    # 寻找日志文件
    log_file = None
    if os.path.exists(target_dir):
        # 1. 尝试找文件名为 step 的文件
        f1 = os.path.join(target_dir, str(step_size))
        if os.path.isfile(f1):
            log_file = f1
        else:
            # 2. 找目录下第一个非 py/json 文件
            for f in os.listdir(target_dir):
                fp = os.path.join(target_dir, f)
                if os.path.isfile(fp) and not f.endswith(('.py', '.json', '.pyc')):
                    log_file = fp
                    break
    
    if not log_file:
        return "-" # 文件未找到

    # 解析
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            for line in reversed(lines):
                if "CNN top1 curve:" in line:
                    match = re.search(r"CNN top1 curve:\s*(\[.*?\])", line)
                    if match:
                        accs = eval(match.group(1))
                        return f"{np.mean(accs):.2f}"
    except:
        pass
    return "-" # 解析失败

def main():
    # 1. 收集数据
    table_data = []
    
    print(f"正在读取 {DATASET} 的消融实验数据...")
    
    for exp in EXPERIMENTS:
        row = []
        
        # 添加勾选列
        row.append("✔" if exp["kd"] else "")
        row.append("✔" if exp["interp"] else "")
        row.append("✔" if exp["rum"] else "")
        
        # 添加 ImageNet-A (空 placeholder)
        row.append("-")
        row.append("-")
        
        # 添加 ImageNet-R 数据 (读取日志)
        print(f"  - 处理实验 ID {exp['id']} ({exp['desc']})...")
        val_t10 = get_avg_acc(exp['id'], 20) # T=10
        val_t5 = get_avg_acc(exp['id'], 40)  # T=5
        row.append(val_t10)
        row.append(val_t5)
        
        # 添加 OmniBenchmark (空 placeholder)
        row.append("-")
        row.append("-")
        
        table_data.append(row)

    # 2. 绘制表格
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    # 表头
    columns = [
        "Knowledge\nDistillation", 
        "Weight\nInterpolation", 
        "Knowledge\nRumination",
        "ImageNet-A\nT=10", "ImageNet-A\nT=5",
        "ImageNet-R\nT=10", "ImageNet-R\nT=5",
        "OmniBenchmark\nT=15", "OmniBenchmark\nT=10"
    ]

    # 创建表格
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        loc='center',
        cellLoc='center'
    )

    # 样式调整
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2) # 调整行高

    # 加粗最后一行 (Ours)
    for col_idx in range(len(columns)):
        cell = table[(len(table_data), col_idx)]
        cell.set_text_props(weight='bold')

    # 调整表头颜色和字体
    for col_idx in range(len(columns)):
        cell = table[(0, col_idx)]
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#f0f0f0')

    plt.title(f"Ablation Study (Reproduced) - ImageNet-R Data Loaded from IDs", y=1.1, fontsize=14)
    
    output_filename = "moal_ablation_table_reproduced.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n表格已生成: {output_filename}")

if __name__ == "__main__":
    main()