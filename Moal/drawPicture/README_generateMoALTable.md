# MoAL Table Generation Script

## Overview

`generateMoALTable.py` is a Python script that generates a performance table showing MoAL's metrics across different datasets and task configurations.

## Features

- Reads log files using the same approach as `drawBasePicture.py`
- Extracts accuracy curves from experiment logs
- Computes two key metrics:
  - **Ā (Average Incremental Accuracy)**: Mean of all accuracy values across tasks
  - **AT (Last-Task Accuracy)**: Accuracy on the final task
- Generates a formatted table as an image file
- Outputs results in both image and text format

## Usage

```bash
python Moal/drawPicture/generateMoALTable.py
```

## Configuration

The script reads from the following log directory:
```
BASE_DIR = "/home/runner/work/Moal/Moal/logs/adapt_ac_com_sdc_ema_auto"
```

It expects log files in the structure:
```
BASE_DIR/
  ├── <dataset_name>/
      └── <hidden_size>/
          └── 0/
              └── <step_size>/
                  └── *.log
```

### Supported Datasets

The script is configured to process the following datasets:

| Dataset | Total Classes | Task Configurations (T) | Hidden Size |
|---------|---------------|------------------------|-------------|
| CIFAR-100 | 100 | 20, 10, 5 | 5000 |
| ImageNet-A | 200 | 20, 10, 5 | 15000 |
| ImageNet-R | 200 | 20, 10, 5 | 20000 |
| OmniBenchmark | 300 | 30, 15, 10 | 10000 |

## Output

The script generates:
1. **moal_table_results.png**: A formatted table image showing MoAL's performance metrics
2. **Console output**: Text-based table and processing status

## Example Output

```
开始收集MoAL数据...

正在处理数据集: CIFAR-100 (Hidden=5000)...
  - T=20: Ā=93.27%, AT=88.36%
  - T=10: Ā=94.22%, AT=90.49%
  - T=5: Ā=94.03%, AT=90.85%
...

表格已保存为: moal_table_results.png
```

## How It Works

1. **parse_log_file()**: Extracts the accuracy curve from log files by searching for "CNN top1 curve:" pattern
2. **find_log_path()**: Locates the appropriate log file based on dataset, hidden size, and step size
3. **compute_metrics()**: Calculates Ā (mean) and AT (last value) from the accuracy curve
4. **collect_moal_data()**: Gathers all metrics across all datasets and configurations
5. **generate_table_image()**: Creates a professional-looking table using matplotlib

## Dependencies

```bash
pip install matplotlib numpy
```

## Notes

- The script uses the same file reading logic as `drawBasePicture.py` for consistency
- Only processes log files containing "CNN top1 curve:" data
- Missing data is marked as "-" in the table
- Average values are computed for each dataset across all task configurations
