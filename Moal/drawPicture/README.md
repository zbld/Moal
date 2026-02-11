# Plotting Scripts for MoAL Results

This directory contains Python scripts for visualizing MoAL (Momentum-based Analytical Learning) experimental results.

## Scripts

### 1. drawBasePicture.py
Generates a 2x2 grid of accuracy curves for four datasets:
- CIFAR-100
- ImageNet-A
- ImageNet-R
- OmniBenchmark

**Output**: `moal_reproduce_results_fixed.png`

**Usage**:
```bash
cd Moal/drawPicture
python drawBasePicture.py
```

### 2. bufferLayerImpact.py
Analyzes the impact of buffer layer dimensions on model performance across different experimental settings.

**Output**: `moal_buffer_dim_impact.png`

**Usage**:
```bash
cd Moal/drawPicture
python bufferLayerImpact.py
```

### 3. ablation1.py
Generates ablation study results table for ImageNet-R, showing the effects of:
- Knowledge Distillation (KD)
- Weight Interpolation (WI)
- Knowledge Rumination (KR)

**Output**: `ablation_study_imagenetr.png`

**Usage**:
```bash
cd Moal/drawPicture
python ablation1.py
```

## Requirements

All required dependencies are specified in `../environment.yml`:
- matplotlib==3.7.5
- pandas==2.0.3
- numpy==1.24.4

## Data Sources

The scripts read log files from `../../logs/adapt_ac_com_sdc_ema_auto/` directory structure:
```
logs/
└── adapt_ac_com_sdc_ema_auto/
    ├── cifar224/
    ├── imageneta/
    ├── imagenetr/
    └── omnibenchmark/
```

Each dataset directory contains subdirectories organized by:
- Hidden layer size (e.g., 5000, 15000, 20000)
- Experiment number (e.g., 0)
- Step size (e.g., 10, 20, 40)

## Notes

- The scripts use relative paths, so they work regardless of where the repository is cloned
- If log files are not found for specific configurations, the plots will show "No Data Found" or "N/A"
- All plots are saved in the current directory with high resolution (300 DPI)
