import os
import matplotlib.pyplot as plt
import pandas as pd

# Get the base directory relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_LOG_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "logs", "adapt_ac_com_sdc_ema_auto"))

# === 1. 定义文件路径 ===
# Task=10 (Increment=20) 的文件列表，保持您提供的顺序
files_t10_raw = [
    os.path.join(BASE_LOG_DIR, "imagenetr/20000/0/20/reproduce_1993_pretrained_vit_b16_224_adapter_fkd0_alpha0_cw0_rg0_test.log"),
    os.path.join(BASE_LOG_DIR, "imagenetr/20000/0/20/reproduce_1993_pretrained_vit_b16_224_adapter_fkd0_alpha0.999_cw0_rg0_test.log"),
    os.path.join(BASE_LOG_DIR, "imagenetr/20000/0/20/reproduce_1993_pretrained_vit_b16_224_adapter_fkd0_alpha0.999_cw1_rg0.1_test.log"),
    os.path.join(BASE_LOG_DIR, "imagenetr/20000/0/20/reproduce_1993_pretrained_vit_b16_224_adapter_fkd1_alpha0_cw0_rg0_test.log"),
    os.path.join(BASE_LOG_DIR, "imagenetr/20000/0/20/reproduce_1993_pretrained_vit_b16_224_adapter_fkd1_alpha0_cw1_rg0.1_test.log")
]

# Task=5 (Increment=40) 的文件列表，保持您提供的顺序
files_t5_raw = [
    os.path.join(BASE_LOG_DIR, "imagenetr/20000/0/40/reproduce_1993_pretrained_vit_b16_224_adapter_fkd0_alpha0_cw0_rg0_test.log"),
    os.path.join(BASE_LOG_DIR, "imagenetr/20000/0/40/reproduce_1993_pretrained_vit_b16_224_adapter_fkd0_alpha0.999_cw0_rg0_test.log"),
    os.path.join(BASE_LOG_DIR, "imagenetr/20000/0/40/reproduce_1993_pretrained_vit_b16_224_adapter_fkd0_alpha0.999_cw1.0_rg0.1_test.log"),
    os.path.join(BASE_LOG_DIR, "imagenetr/20000/0/40/reproduce_1993_pretrained_vit_b16_224_adapter_fkd1_alpha0_cw0_rg0_test.log"),
    os.path.join(BASE_LOG_DIR, "imagenetr/20000/0/40/reproduce_1993_pretrained_vit_b16_224_adapter_fkd1_alpha0_cw1.0_rg0.1_test.log")
]

# === 2. 数据提取函数 ===
def extract_accuracy(filepath):
    acc = "N/A"
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return acc
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # 查找最后出现的 "Average Accuracy (CNN):"
            for line in lines:
                if "Average Accuracy (CNN):" in line:
                    # 格式如: ... Average Accuracy (CNN): 63.10400...
                    parts = line.split("Average Accuracy (CNN):")
                    if len(parts) > 1:
                        try:
                            val = float(parts[1].strip())
                            acc = f"{val:.2f}" # 保留两位小数
                        except ValueError:
                            pass
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return acc

# === 3. 数据处理与排序 ===
# 根据您的文件列表顺序：
# Index 0: fkd0_alpha0_cw0 (Baseline)
# Index 1: fkd0_alpha0.999_cw0 (Weight Interpolation)
# Index 2: fkd0_alpha0.999_cw1 (Weight Interpolation + Rumination)
# Index 3: fkd1_alpha0_cw0 (Knowledge Distillation)
# Index 4: fkd1_alpha0_cw1 (Knowledge Distillation + Rumination)

# 目标表格的行顺序通常为：
# Row 1: Baseline (No checks) -> 对应 Index 0
# Row 2: KD (check 1) -> 对应 Index 3
# Row 3: WI (check 2) -> 对应 Index 1
# Row 4: KD + KR (check 1, 3) -> 对应 Index 4
# Row 5: WI + KR (check 2, 3) -> 对应 Index 2

# 定义从文件列表到表格行的映射索引
map_indices = [0, 3, 1, 4, 2]

acc_t10_list = [extract_accuracy(f) for f in files_t10_raw]
acc_t5_list = [extract_accuracy(f) for f in files_t5_raw]

# 按表格顺序重新排列数据
data_t10 = [acc_t10_list[i] for i in map_indices]
data_t5 = [acc_t5_list[i] for i in map_indices]

# === 4. 构建表格数据 ===
# 表格各列的勾选状态 (✓)
check_marks = [
    ["", "", ""],           # Row 1: Baseline
    ["✓", "", ""],          # Row 2: KD
    ["", "✓", ""],          # Row 3: WI
    ["✓", "", "✓"],         # Row 4: KD + KR
    ["", "✓", "✓"]          # Row 5: WI + KR
]

# 准备绘图数据
table_data = []
for i in range(5):
    row = check_marks[i] + [data_t10[i], data_t5[i]]
    table_data.append(row)

columns = ["Knowledge\nDistillation", "Weight\nInterpolation", "Knowledge\nRumination", "T = 10", "T = 5"]

# === 5. 绘制并保存图片 ===
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')

# 创建表格
table = ax.table(
    cellText=table_data,
    colLabels=columns,
    cellLoc='center',
    loc='center',
    colColours=["#f2f2f2"] * 5 
)

# 调整表格样式
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)  # 调整行高

# 添加标题 (包含 Imagenet-R)
plt.title("Ablation study on Imagenet-R (Reproduced)", pad=20, fontsize=14, fontweight='bold')

# 添加额外的列头分组标注 (手动添加文本)
# 坐标需要根据实际图形调整，这里是大致位置
plt.text(0.73, 0.82, "Imagenet-R", transform=ax.transAxes, ha='center', fontsize=12, fontweight='bold')

output_file = "ablation_study_imagenetr.png"
plt.savefig(output_file, bbox_inches='tight', dpi=300)
print(f"表格图片已保存为: {output_file}")
print("-" * 30)
print("提取的数据预览:")
print("Row | KD | WI | KR | T=10 | T=5")
for i, row in enumerate(table_data):
    print(f" {i+1}  | {row[0]:<2} | {row[1]:<2} | {row[2]:<2} | {row[3]} | {row[4]}")