# MoAL (Momentum-based Analytical Learning) - Experimental Reproduction Guide

## 📚 Paper Information

**Title**: Knowledge Memorization and Rumination for Pre-trained Model-based Class-Incremental Learning

**Conference**: CVPR 2025

**Core Method**: MoAL - Momentum-based Analytical Learning

**Research Goal**: In Class-Incremental Learning (CIL) scenarios, combine Pre-Trained Models (PTMs) with knowledge memorization and rumination mechanisms to achieve good adaptivity to new classes (plasticity) and stable retention of old class knowledge (stability).

---

## 🎯 Key Innovations

1. **Analytical Classification Head**: Recursive knowledge acquisition using least squares solution
2. **Momentum-based Adapter Weight Interpolation**: Adaptive to new classes through EMA mechanism while forgetting outdated knowledge
3. **Knowledge Rumination Mechanism**: Revisit and reinforce old knowledge using refined adaptivity
4. **Dual-Branch Network Architecture**: Combines Cosine classifier and Analytical Learning classifier

---

## 📋 Environment Setup

### 1. System Requirements
- Linux OS (Ubuntu 18.04+ recommended)
- CUDA 11.x (for PyTorch 2.0.1)
- Python 3.8
- Recommended GPU memory: 12GB+

### 2. Installation

#### Step 1: Create Conda Environment
```bash
# Create environment using the provided file
conda env create -f environment.yml

# Activate environment
conda activate AL
```

#### Step 2: Key Dependencies
The environment file includes the following core dependencies:
- **PyTorch**: 2.0.1 (with CUDA 11)
- **torchvision**: 0.15.2
- **timm**: 0.6.12 (for Vision Transformer)
- **numpy**: 1.24.4
- **scipy**: 1.10.1
- **tqdm**: Progress bar display

### 3. Dataset Preparation

#### Supported Datasets
| Dataset | Download | Description |
|---------|----------|-------------|
| **CIFAR-100** | Auto-download | Code will download automatically |
| **ImageNet-R** | [Google Drive](https://drive.google.com/file/d/1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR/view?usp=sharing) / [OneDrive](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EU4jyLL29CtBsZkB6y-JSbgBzWF5YHhBAUz1Qw8qM2954A?e=hlWpNW) | 200-class ImageNet variant |
| **ImageNet-A** | [Google Drive](https://drive.google.com/file/d/19l52ua_vvTtttgVRziCZJjal0TPE9f2p/view?usp=sharing) / [OneDrive](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/ERYi36eg9b1KkfEplgFTW3gBg1otwWwkQPSml0igWBC46A?e=NiTUkL) | Adversarial examples |
| **OmniBenchmark** | [Google Drive](https://drive.google.com/file/d/1AbCP3zBMtv_TDXJypOCnOgX8hJmvJm3u/view?usp=sharing) / [OneDrive](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EcoUATKl24JFo3jBMnTV2WcBwkuyBH0TmCAy6Lml1gOHJA?e=eCNcoA) | Comprehensive benchmark |
| **Car196** | [GitHub](https://github.com/jhpohovey/StanfordCars-Dataset) | Stanford Cars dataset |
| **CUB-200** | [Caltech](https://www.vision.caltech.edu/datasets/cub_200_2011/) | Fine-grained bird classification |

#### Dataset Path Configuration
After downloading datasets, configure paths in `utils/data.py`:

```python
def download_data(self):
    # Modify to your dataset path
    train_dir = '[YOUR-DATA-PATH]/train/'
    test_dir = '[YOUR-DATA-PATH]/val/'
```

**Important**: You must modify this path, otherwise the code will raise an error.

### 4. Pre-trained Weights Preparation

#### Download Pre-trained Models
MoAL supports three pre-trained ViT-B/16 models:

| Model | Download Link | Description |
|-------|---------------|-------------|
| **Sup-21K ViT** | [Google Storage](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz) | ImageNet-21K supervised pre-training |
| **iBOT** | [ByteDance](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16/checkpoint_teacher.pth) | Self-supervised pre-training |
| **DINO** | [Facebook](https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth) | Self-supervised pre-training |

#### Place Pre-trained Weights
Put downloaded weights in: `Moal/checkpoints/` directory

#### Configure Weight Paths
In `backbone/vision_transformer_adapter.py` or corresponding model files:

```python
def vit_base_patch16_224_adapter_dino(pretrained=False, **kwargs):
    model = VisionTransformer(...)
    # Modify to your weight path
    ckpt = torch.load('Moal/checkpoints/dino_vitbase16_pretrain.pth', map_location='cpu')
    ...

def vit_base_patch16_224_adapter_ibot(pretrained=False, **kwargs):
    model = vit_base_patch16_224_adapter(False, **kwargs)
    # Modify to your weight path
    ckpt = torch.load('Moal/checkpoints/checkpoint_teacher.pth', map_location='cpu')['state_dict']
    ...
```

---

## 🚀 Model Training Workflow

### Training Command Format
```bash
cd Moal
python main.py --config=exps/MoAL_[dataset_name].json
```

### Supported Configuration Files
```bash
exps/Moal_cifar224.json      # CIFAR-100 experiment
exps/Moal_imagenetr.json     # ImageNet-R experiment
exps/Moal_imageneta.json     # ImageNet-A experiment
exps/Moal_cars.json          # Car196 experiment
exps/Moal_cub.json           # CUB-200 experiment
exps/Moal_omnibenchmark.json # OmniBenchmark experiment
```

### Configuration Parameters Explained

Using `Moal_cifar224.json` as example:

```json
{
    "prefix": "reproduce",              // Experiment prefix identifier
    "dataset": "cifar224",              // Dataset name
    "memory_size": 0,                   // Memory buffer size (MoAL doesn't use memory)
    "memory_per_class": 0,
    "fixed_memory": false,
    "shuffle": true,                    // Whether to shuffle class order
    "init_cls": 10,                     // Number of classes in first task
    "increment": 10,                    // Number of classes per incremental task
    "model_name": "adapt_ac_com_sdc_ema_auto",  // Model name
    "backbone_type": "pretrained_vit_b16_224_adapter",  // Backbone network type
    "device": ["0"],                    // GPU device ID
    "seed": [1993],                     // Random seed
    "Hidden": 15000,                    // Hidden dimension for Analytical Learning
    "rg": 0.1,                          // Regularization coefficient
    "lambda_fkd": 1,                    // Knowledge distillation loss weight
    "cali_weight": 1.0,                 // Knowledge rumination weight
    "alpha": 0.999,                     // EMA momentum coefficient
    "tuned_epoch": 80,                  // Training epochs for first task
    "progreesive_epoch": 80,            // Training epochs for incremental tasks
    "init_lr": 0.01,                    // Initial learning rate
    "progressive_lr": 0.01,             // Incremental learning rate
    "batch_size": 48,                   // Batch size
    "weight_decay": 0.0005,             // Weight decay
    "min_lr": 0,                        // Minimum learning rate
    "ffn_num": 64,                      // Adapter FFN dimension
    "optimizer": "sgd",                 // Optimizer type
    "vpt_type": "shallow",              // Visual Prompt Tuning type
    "prompt_token_num": 5               // Number of prompt tokens
}
```

### Key Hyperparameters

| Parameter | Function | Recommended Value | Ablation Impact |
|-----------|----------|-------------------|-----------------|
| **alpha** | EMA momentum coefficient, controls weight interpolation speed | 0.999 | Larger = more conservative, smaller = faster adaptation |
| **lambda_fkd** | Knowledge distillation loss weight | 0-1 | Controls retention of old knowledge |
| **cali_weight** | Knowledge rumination weight | 0-1.0 | Controls strength of rumination mechanism |
| **rg** | Parameter regularization coefficient | 0-0.1 | Prevents excessive adapter parameter drift |
| **Hidden** | AL hidden dimension | 15000-20000 | Affects representation capacity |

---

## 🔬 Training Process Details

### Overall Workflow

```
Start
  ↓
Task 0 (Initial Task)
  ├─ Load pre-trained ViT + Adapter
  ├─ Fine-tune adapter (tuned_epoch=80)
  ├─ Construct dual-branch network (Cosine + AC)
  ├─ Initialize Analytical Learning classification head
  ├─ Compute class prototype means
  └─ cls_align: Solve initial weights using least squares
  ↓
Task 1-N (Incremental Tasks)
  ├─ Extend classification head to new class count
  ├─ Freeze AC model
  ├─ Train adapter (progreesive_epoch=80)
  │   ├─ Cross-Entropy Loss
  │   ├─ Knowledge Distillation Loss (if lambda_fkd > 0)
  │   ├─ Parameter Regularization (if rg > 0)
  │   └─ EMA Update (if alpha > 0)
  ├─ Compute new class prototype means
  ├─ IL_align: Incrementally update AL weights
  ├─ (Optional) Knowledge Rumination:
  │   ├─ cali_prototye_model: Calibrate prototype model
  │   ├─ _compute_relations: Compute old-new class relationships
  │   ├─ _build_feature_set: Build rumination feature set
  │   └─ cali_weight: Calibrate weights based on misclassifications
  └─ Evaluate current task performance
  ↓
End: Output average accuracy across all tasks
```

### First Task (Task 0) Detailed Flow

**Goal**: Fine-tune pre-trained model on initial classes, establish foundational knowledge

1. **Network Initialization**
   ```python
   # Load pre-trained ViT-B/16 + Adapter
   self._network = SimpleVitNet_AL(args, True)
   # Initialize Cosine classifier
   self._network.fc = CosineLinear(feature_dim=768, num_classes=init_cls)
   ```

2. **Adapter Fine-tuning** (`_init_train`)
   - Training epochs: `tuned_epoch` (default 80)
   - Loss function: Cross-Entropy
   - Optimizer: SGD (momentum=0.9)
   - Learning rate scheduler: CosineAnnealing
   - Only train Adapter parameters, freeze ViT backbone

3. **Construct Dual-Branch Network** (`construct_dual_branch_network`)
   ```python
   # Create two branches:
   # Branch 1: Cosine Classifier (for training)
   # Branch 2: Analytical Classifier (for inference)
   network = MultiBranchCosineIncrementalNet_adapt_AC(args, True)
   ```

4. **Initialize Analytical Learning Head** (`cls_align`)
   - Extract training set features
   - Compute auto-correlation matrix: `auto_cor = Φ^T Φ`
   - Compute cross-correlation matrix: `crs_cor = Φ^T Y`
   - Ridge regression optimization: Select best regularization parameter λ
   - Solve least squares: `W = (Φ^T Φ + λI)^(-1) Φ^T Y`

5. **Compute Class Prototypes** (`_compute_means`)
   - Compute feature mean for each class
   - Compute covariance matrix and standard deviation (for later rumination)

### Incremental Tasks (Task 1-N) Detailed Flow

**Goal**: Learn new classes while maintaining memory of old classes

1. **Network Extension**
   ```python
   # Extend classification head to include new classes
   self._network.update_fc(total_classes, Hidden, cosine_fc=True)  # Cosine branch
   self._network.update_fc(total_classes, Hidden)  # AC branch
   # Freeze AC model parameters
   for param in self._network.ac_model.parameters():
       param.requires_grad = False
   ```

2. **Adapter Incremental Training** (`_progreessive_train`)
   - Training epochs: `progreesive_epoch` (default 80)
   - Create EMA model for momentum updates
   
   **Loss Function Components**:
   ```python
   # Basic classification loss
   loss_ce = CrossEntropy(logits, targets)
   
   # (Optional) Knowledge distillation loss (lambda_fkd > 0)
   loss_kd = KL_Divergence(
       current_logits[:, :old_classes] / T,
       old_logits[:, :old_classes] / T
   ) * T^2
   
   # (Optional) Parameter regularization (rg > 0)
   loss_reg = ||θ_new - θ_old||^2
   
   # Total loss
   loss = loss_ce + λ_fkd * loss_kd + rg * loss_reg
   ```

3. **EMA Momentum Update** (after each epoch)
   ```python
   # If alpha > 0
   θ_ema = alpha * θ_ema + (1 - alpha) * θ_current
   # After training, copy EMA parameters back to main model
   ```

4. **Compute New Class Prototypes** (`_compute_means`)
   - Compute feature mean and statistics for each new class

5. **Incremental AL Weight Update** (`IL_align`)
   - Use Recursive Least Squares (RLS) algorithm
   - For each new sample batch:
     ```python
     # Sherman-Morrison-Woodbury formula
     R_new = R_old - R_old @ Φ^T @ (I + Φ @ R_old @ Φ^T)^(-1) @ Φ @ R_old
     W_new = W_old + R_new @ Φ^T @ (Y - Φ @ W_old)
     ```

6. **Knowledge Rumination Mechanism** (if `cali_weight > 0`)
   
   a. **Calibrate Prototype Model** (`cali_prototye_model`)
      - Create calibration model for old classes
   
   b. **Compute Old-New Class Relationships** (`_compute_relations`)
      ```python
      # Find most similar new class for each old class
      similarity = normalize(old_means) @ normalize(new_means)^T
      relations[i] = argmax(similarity[i]) + known_classes
      ```
   
   c. **Build Rumination Feature Set** (`_build_feature_set`)
      - Collect real features from new classes
      - Generate pseudo-features for old classes:
        ```python
        pseudo_feature[old_class] = 
            feature[related_new_class] - mean[related_new_class] + mean[old_class]
        ```
   
   d. **Calibrate Weights Based on Errors** (`cali_weight`)
      - Only update weights for misclassified samples
      - Apply RLS algorithm again

7. **Post-Task Processing** (`after_task`)
   - Save current network as old network `_old_network`
   - Used for knowledge distillation in next task

---

## 📊 Experiment Running Examples

### CIFAR-100 Experiment

```bash
# Setting: 10 tasks, 10 classes per task
python main.py --config=exps/Moal_cifar224.json
```

**Incremental Setting**: 10 + 10×9 = 100 classes

**Expected Output**:
```
Task 0: Learn classes 0-9
  - Train 80 epochs
  - Initialize AL classification head
  - Test Accuracy: ~90%+

Task 1: Learn classes 10-19 (known classes 0-9)
  - Train 80 epochs
  - Incrementally update AL weights
  - Apply knowledge rumination
  - Test Accuracy: ~85%+

...

Task 9: Learn classes 90-99 (known classes 0-89)
  - Final average accuracy: ~78%+
```

### ImageNet-R Experiment

```bash
# Setting: 10 tasks, 20 classes per task
python main.py --config=exps/Moal_imagenetr.json
```

**Incremental Setting**: 20 + 20×9 = 200 classes

**Note**: In ImageNet-R experiments, some ablation parameters are set to 0:
- `rg = 0`: No parameter regularization
- `lambda_fkd = 0`: No knowledge distillation
- `cali_weight = 0`: No knowledge rumination

This is to test the core method's performance on large-scale datasets.

---

## 🔍 Model Usage Workflow

### Inference with Trained Model

While the code includes infrastructure for model saving, it's primarily for experimental evaluation. For deployment:

1. **Save Checkpoint**
   ```python
   # Already implemented in base.py
   model.save_checkpoint(filename='model_checkpoint')
   # Saves as: model_checkpoint_{task_id}.pkl
   ```

2. **Load Checkpoint**
   ```python
   checkpoint = torch.load('model_checkpoint_9.pkl')
   model._network.load_state_dict(checkpoint['model_state_dict'])
   model._cur_task = checkpoint['tasks']
   ```

3. **Inference**
   ```python
   model._network.eval()
   with torch.no_grad():
       features = model._network(images)["features"]
       # Use AC classification head
       activation = model._network.ac_model.fc[:2](features)
       logits = model._network.ac_model.fc[-1](activation)
       predictions = torch.argmax(logits, dim=1)
   ```

### Feature Extraction

```python
# Extract prototype feature for a specific class
def extract_class_prototype(model, class_idx, dataloader):
    model._network.eval()
    vectors = []
    with torch.no_grad():
        for _, inputs, targets in dataloader:
            if targets == class_idx:
                features = model._network(inputs.to(device))["features"]
                vectors.append(features.cpu().numpy())
    return np.mean(np.concatenate(vectors), axis=0)
```

---

## 📈 Logs and Results Analysis

### Log File Location
```
logs/
└── {model_name}/
    └── {dataset}/
        └── {Hidden}/
            └── {init_cls}/
                └── {increment}/
                    └── {prefix}_{seed}_{backbone}_fkd{λ}_alpha{α}_cw{w}_rg{r}_test.log
```

### Key Output Metrics

1. **After Each Task**:
   - CNN top1/top5 accuracy (Cosine classifier)
   - NME top1/top5 accuracy (Analytical classifier)
   - Grouped accuracy: {total, old, new, task0, task1, ...}

2. **Final Metrics**:
   - Average Accuracy (CNN): Average accuracy across all tasks
   - Average Accuracy (NME): AL classifier average accuracy
   - All History Accuracy: Accuracy matrix for each task on all seen classes

### Results Visualization

Code includes plotting functionality (in `drawPicture/` directory) to generate:
- Accuracy curve plots
- Ablation study comparison plots
- Per-task performance breakdown plots

---

## ⚙️ Ablation Study Configurations

MoAL's key components can be ablated through configuration files:

### 1. Without EMA (alpha = 0)
```json
{
    "alpha": 0,
    // Keep other parameters unchanged
}
```
**Impact**: Adapter parameters won't undergo momentum interpolation, may lead to decreased adaptivity

### 2. Without Knowledge Distillation (lambda_fkd = 0)
```json
{
    "lambda_fkd": 0,
}
```
**Impact**: Reduced old knowledge retention, increased catastrophic forgetting

### 3. Without Knowledge Rumination (cali_weight = 0)
```json
{
    "cali_weight": 0,
}
```
**Impact**: Cannot reinforce old class knowledge through pseudo-features, decreased old class performance

### 4. Without Parameter Regularization (rg = 0)
```json
{
    "rg": 0,
}
```
**Impact**: Adapter parameters may drift excessively, affecting feature extraction stability

### 5. Hidden Dimension Impact
```json
{
    "Hidden": 10000,  // vs 15000, 20000, 25000
}
```
**Impact**: Trade-off between representation capacity and computational cost

---

## ❓ Questions and Suggestions

### Key Questions

1. **Hard-coded Pre-trained Weight Paths**
   - **Issue**: Weight paths in `backbone/vision_transformer_adapter.py` require manual modification
   - **Suggestion**: Make weight path a config parameter in JSON files
   ```json
   {
       "pretrained_weight_path": "checkpoints/dino_vitbase16_pretrain.pth"
   }
   ```

2. **Inflexible Dataset Path Configuration**
   - **Issue**: Data paths hard-coded in `utils/data.py`
   - **Suggestion**: Support environment variables or config file specification
   ```python
   data_path = os.environ.get('MOAL_DATA_PATH', args.get('data_path', '/default/path'))
   ```

3. **Memory Mechanism Implemented but Unused**
   - **Observation**: Config has `memory_size=0`, but code includes exemplar building logic
   - **Question**: Does MoAL completely not need sample storage? Or is it an optional feature?
   - **Suggestion**: Document whether memory-based variants are supported

4. **Multi-GPU Training Stability**
   - **Issue**: Uses `nn.DataParallel`, may be inefficient for large-scale experiments
   - **Suggestion**: Consider upgrading to `DistributedDataParallel`

5. **Evaluation Metrics**
   - **Observation**: Outputs both CNN and NME accuracy
   - **Question**: Which metric should be used in final reports?
   - **Suggestion**: Documentation should clarify NME (Analytical Learning) is the main metric

6. **Knowledge Rumination Computational Cost**
   - **Observation**: `_build_feature_set` needs to iterate through all new class samples
   - **Question**: Performance bottleneck on large-scale datasets?
   - **Suggestion**: Consider sampling strategies or batch processing optimization

7. **Ridge Regression Hyperparameter Selection**
   - **Observation**: `optimise_ridge_parameter` uses fixed candidate set
   - **Suggestion**: Dynamically adjust search range based on dataset scale

### Usage Recommendations

1. **First Run Recommendations**
   - First validate environment setup on CIFAR-100
   - Use smaller Hidden dimension (e.g., 10000) for quick testing
   - Confirm log output and accuracy meet expectations

2. **Hyperparameter Tuning Recommendations**
   - `alpha`: Large datasets recommend 0.999, small datasets can lower to 0.99
   - `Hidden`: Adjust based on GPU memory (15000 requires ~10-12GB VRAM)
   - `lambda_fkd`: Increase for more tasks (0.5-1.0)

3. **Hardware Configuration Recommendations**
   - CIFAR-100: Single 1080Ti/2080Ti (11GB) sufficient
   - ImageNet-R/A: Recommend 3090/V100 (24GB)
   - OmniBenchmark: Recommend A100 (40GB) or multi-GPU

4. **Experiment Time Estimates**
   - CIFAR-100 (10 tasks): ~8-10 hours (single 3090)
   - ImageNet-R (10 tasks): ~20-24 hours (single 3090)
   - Main time consumption in adapter fine-tuning phase

---

## 🔧 Common Issues Troubleshooting

### 1. CUDA Out of Memory
**Solutions**:
- Reduce `batch_size` (48 → 24)
- Reduce `Hidden` dimension (15000 → 10000)
- Use gradient accumulation

### 2. Dataset Loading Failure
**Check**:
- Is path configuration in `utils/data.py` correct?
- Does dataset directory structure meet requirements (train/, val/)?
- CIFAR-100 needs network connection for auto-download

### 3. Pre-trained Weight Loading Failure
**Check**:
- Are weight files completely downloaded?
- Does `checkpoints/` directory exist?
- Is path configuration in `vision_transformer_adapter.py` correct?

### 4. Abnormally Low Accuracy
**Possible Causes**:
- Pre-trained weights not loaded correctly
- Inappropriate learning rate settings
- Data augmentation too strong or weak
- Ridge regression parameter selection failure

### 5. Training Process Hangs
**Possible Causes**:
- `num_workers` set too high, try reducing to 4 or 0
- Data loading bottleneck, check disk I/O

---

## 📚 Code Structure Summary

```
Moal/
├── main.py                    # Main entry point
├── trainer.py                 # Training flow control
├── environment.yml            # Conda environment config
├── exps/                      # Experiment config files
│   ├── Moal_cifar224.json
│   ├── Moal_imagenetr.json
│   └── ...
├── models/                    # Model implementations
│   ├── base.py               # Base class
│   └── adapt_ac_com_sdc_EMA_auto.py  # MoAL main implementation
├── backbone/                  # Backbone networks
│   ├── vision_transformer_adapter.py  # ViT + Adapter
│   └── linears.py            # Classifier layers
├── utils/                     # Utility functions
│   ├── data.py               # Dataset definitions
│   ├── data_manager.py       # Data management
│   ├── factory.py            # Model factory
│   ├── inc_net.py            # Incremental network
│   ├── AC_net.py             # Analytical Learning network
│   └── toolkit.py            # Helper functions
└── logs/                      # Log output directory
```

---

## 📖 Related Resources

- **Paper**: CVPR 2025 - Knowledge Memorization and Rumination for Pre-trained Model-based Class-Incremental Learning
- **PyCIL Framework**: https://github.com/G-U-N/PyCIL
- **LAMDA-PILOT**: https://github.com/sun-hailong/LAMDA-PILOT
- **ViT Adapters**: https://github.com/jxhe/unify-parameter-efficient-tuning

---

## ✅ Quick Start Checklist

- [ ] Install Conda environment (`conda env create -f environment.yml`)
- [ ] Download and configure dataset paths (`utils/data.py`)
- [ ] Download pre-trained weights to `checkpoints/`
- [ ] Configure pre-trained weight paths (`backbone/vision_transformer_adapter.py`)
- [ ] Select experiment config file (`exps/Moal_*.json`)
- [ ] Modify GPU device ID in config
- [ ] Run training: `python main.py --config=exps/Moal_cifar224.json`
- [ ] Check log output (`logs/`)
- [ ] Analyze results and accuracy curves

---

## 🎓 Summary

MoAL achieves **plasticity-stability** balance in class-incremental learning through these core techniques:

1. **Analytical Learning**: Recursive least squares solution for efficient new knowledge acquisition
2. **Momentum-based Adaptation**: EMA mechanism for smooth weight transitions
3. **Knowledge Rumination**: Reinforce old knowledge through pseudo-feature rumination

Key to successful reproduction:
- ✅ Correctly configure pre-trained weights and datasets
- ✅ Understand the role and impact of each hyperparameter
- ✅ Follow incremental learning evaluation protocols
- ✅ Analyze Analytical Learning classifier performance

For additional questions or further experimental guidance, feel free to ask!
