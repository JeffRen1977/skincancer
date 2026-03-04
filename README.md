# Skin Cancer Classification - Transfer Learning Project

PyTorch-based transfer learning project for skin cancer classification using the HAM10000 dataset.

## Project Requirements Verification

This project meets all requirements from the Transfer Learning Project assignment (AICV, Fall 2025).

### Required Items ✅

1. **✅ Different Dataset from defungi**
   - Using HAM10000 skin cancer dataset (7 classes: actinic keratoses, basal cell carcinoma, benign keratosis-like lesions, dermatofibroma, melanocytic nevi, melanoma, vascular lesions)
   - No references to defungi dataset in the codebase
   - Dataset downloaded from Kaggle/Dataset Ninja

2. **✅ PyTorch Framework (instead of TensorFlow)**
   - All code uses PyTorch (`torch`, `torchvision`)
   - PyTorch version: 2.8.0+
   - No TensorFlow dependencies

3. **✅ Different Model from ResNet**
   - Using **EfficientNet-B0** (not ResNet)
   - Pre-trained on ImageNet
   - Code: `efficientnet_b0, EfficientNet_B0_Weights`

### Customizations (Minimum 3 Required - We Have 5) ✅

4. **✅ Model Saving with Checkpoints**
   - Saves final models: `skincancer_model_{optimizer}.pth`
   - **Checkpoints saved automatically every epoch**: `checkpoint_epoch_{N}.pth`
   - Best model saved: `best_model.pth` (based on validation accuracy)
   - Checkpoint includes: model state, optimizer state, epoch, loss, accuracy

5. **✅ Separate Inference Program**
   - `inference.py` - Standalone script to load models and classify images
   - Can classify images from **outside the training dataset** (as required)
   - Supports command-line arguments for model path, image path, device selection
   - Displays prediction with confidence scores for all classes

6. **✅ Matplotlib Visualization**
   - Plots **4 metrics** as required:
     - Training Loss
     - Validation Loss
     - Training Accuracy
     - Validation Accuracy
   - Two subplots showing all 4 line graphs
   - Saved as PNG files: `training_history_{optimizer}.png`
   - High-resolution output (300 DPI)

7. **✅ Class-Based Model Architecture**
   - `SkinCancerModel(nn.Module)` - Class-based architecture
   - Not using functional/sequential API
   - Custom `forward()` method implementation
   - Proper PyTorch module structure

8. **✅ Optimizer Comparison (3 Different Optimizers)**
   - **Adam**: `torch.optim.Adam` (adaptive learning rate)
   - **SGD**: `torch.optim.SGD` (with momentum=0.9)
   - **AdamW**: `torch.optim.AdamW` (with weight_decay=0.01)
   - All three optimizers trained with same hyperparameters for fair comparison
   - Comparison summary printed at end of training

### Summary

- **All Required Items**: ✅ Met
- **Customizations**: ✅ 5 implemented (minimum 3 required)
- **Code Structure**: ✅ Follows PyTorch best practices
- **Project Requirements**: ✅ Fully satisfied  

## Dataset

The project uses the HAM10000 skin cancer dataset with 7 classes:
1. Actinic keratoses
2. Basal cell carcinoma
3. Benign keratosis-like lesions
4. Dermatofibroma
5. Melanocytic nevi
6. Melanoma
7. Vascular lesions

### Downloading the HAM10000 dataset

The HAM10000 dataset can be downloaded from:
- **Kaggle**: https://www.kaggle.com/datasets/kmader/skin-cancer-ham10000
- **Dataset Ninja**: https://datasetninja.com/skin-cancer-ham10000

The dataset should have the following structure:
```
skincancer/
├── HAM10000_images_part_1/
│   └── [image files: ISIC_*.jpg]
├── HAM10000_images_part_2/
│   └── [image files: ISIC_*.jpg]
└── HAM10000_metadata.csv
```

The metadata CSV contains the following columns:
- `lesion_id`: Unique lesion identifier
- `image_id`: Image filename (without extension)
- `dx`: Diagnosis code (akiec, bcc, bkl, df, mel, nv, vasc)
- `dx_type`: Diagnosis type (histo, follow-up, consensus, etc.)
- `age`, `sex`, `localization`: Additional metadata

**Note**: After downloading, place the dataset in the `skincancer/` folder and run `organize_data.py` to organize images by class.

## Installation

Install required dependencies:
```bash
pip3 install -r requirements.txt
```

For Mac with Apple Silicon (M1/M2/M3), PyTorch will automatically use MPS (Metal Performance Shaders) for GPU acceleration.

## Setup

1. **Download and place the dataset**:
   - Download the HAM10000 dataset from Kaggle or Dataset Ninja
   - Extract it so you have:
     - `skincancer/HAM10000_images_part_1/` (contains image files)
     - `skincancer/HAM10000_images_part_2/` (contains image files)
     - `skincancer/HAM10000_metadata.csv` (contains labels and metadata)

2. **Organize the dataset** (run this first):
```bash
python3 organize_data.py
```

This script will:
- Read the `HAM10000_metadata.csv` file
- Map diagnosis codes (dx) to class names:
  - `akiec` → `actinic_keratoses`
  - `bcc` → `basal_cell_carcinoma`
  - `bkl` → `benign_keratosis-like_lesions`
  - `df` → `dermatofibroma`
  - `mel` → `melanoma`
  - `nv` → `melanocytic_nevi`
  - `vasc` → `vascular_lesions`
- Search for images in both `part_1` and `part_2` folders
- Copy images to `skincancer/organized/` organized by class folders
- Print a summary of images organized per class

3. **Train models** — see [Trainers](#trainers) below for all options.

4. **Train the EfficientNet model** (transfer learning):
```bash
python3 train_skincancer.py
```

This will:
- Train models with 3 different optimizers (Adam, SGD, AdamW)
- Save checkpoints after each epoch
- Save the best model for each optimizer
- Generate training history plots
- Save final models

5. **Run inference** on a new image:
```bash
python3 inference.py --model skincancer_model_adam.pth --image path/to/image.jpg
```

6. **Analyze class-wise accuracy** — see [Class Accuracy Analysis](#class-accuracy-analysis) below.

---

## Trainers

The project includes five trainers for skin cancer classification. All use the HAM10000 dataset from `skincancer/organized/` (run `organize_data.py` first).

**Prerequisites:** Create `saves/` for model outputs, and install dependencies including OpenCV:
```bash
mkdir -p saves
pip3 install -r requirements.txt
pip3 install opencv-python
```

### Trainer Comparison

| Trainer | Script | Input Size | Architecture | Key Features | Output |
|---------|--------|------------|--------------|--------------|--------|
| **First CNN** | `first_cnn_torch.py` | 256×256 | Conv 13/7/3, ReLU, dropout | Class weights, focal loss, WeightedRandomSampler | `saves/first_cnn_model.pth` |
| **Second CNN** | `second_cnn_torch.py` | 256×256 | Conv 13/7/3, LeakyReLU*, dropout | Data augmentation, WeightedRandomSampler | `saves/second_cnn_model.pth` |
| **Image CNN** | `image_cnn_torch.py` | 256×256 | Conv 13/7/3, ReLU, no dropout | Original architecture | `saves/image_cnn_model.pth` |
| **Image2 CNN** | `image2_cnn_torch.py` | 160×160 | Conv 9/5/3, ReLU, no dropout | Smaller input, smaller kernels | `saves/image2_cnn_model.pth` |
| **EfficientNet** | `train_skincancer.py` | 224×224 | EfficientNet-B0 transfer learning | 3 optimizers (Adam, SGD, AdamW) | `skincancer_model_*.pth` |

*Second CNN: configurable activation (relu, leaky_relu, gelu, silu, elu); default is LeakyReLU.

### How to Run Each Trainer

**Individual custom CNNs:**
```bash
python3 first_cnn_torch.py      # First CNN (ReLU, dropout, focal loss)
python3 second_cnn_torch.py     # Second CNN (LeakyReLU, augmentation)
python3 image_cnn_torch.py      # Image CNN (256×256, no dropout)
python3 image2_cnn_torch.py     # Image2 CNN (160×160, no dropout)
```

**EfficientNet (transfer learning):**
```bash
python3 train_skincancer.py      # Trains 3 models: Adam, SGD, AdamW
```

**Train all models and run analysis:**
```bash
python3 train_and_analyze_all.py
```

Options for `train_and_analyze_all.py`:
- `--skip-training` — Only run analysis on existing models
- `--skip-analysis` — Only train models
- `--skip-efficientnet` — Skip EfficientNet (saves ~30 min)
- `--max-images N` — Max images per class for analysis (default: 50)

### Trainer Differences (Custom CNNs)

- **First CNN** vs **Image CNN**: Same conv layout (256×256, kernels 13/7/3). First CNN adds dropout, class weights, and focal loss for imbalanced data.
- **Second CNN**: Same conv layout as First CNN, but uses configurable activation (LeakyReLU by default) and training-time data augmentation (flips, rotation, color jitter, affine).
- **Image CNN** vs **Image2 CNN**: Image CNN uses 256×256 input and kernels 13/7/3; Image2 CNN uses 160×160 input and smaller kernels 9/5/3. Neither uses dropout.

### Architecture Comparison

| Aspect | First CNN | Second CNN | Image CNN | Image2 CNN | EfficientNet |
|--------|-----------|------------|-----------|------------|--------------|
| **Input size** | 256×256 | 256×256 | 256×256 | 160×160 | 224×224 |
| **Approach** | From scratch | From scratch | From scratch | From scratch | Transfer learning |
| **Conv layers** | 3 blocks (24→48→96 ch) | 3 blocks (24→48→96 ch) | 3 blocks (24→48→96 ch) | 3 blocks (24→48→96 ch) | EfficientNet-B0 backbone |
| **Conv kernels** | 13, 7, 3 | 13, 7, 3 | 13, 7, 3 | 9, 5, 3 | Compound scaling |
| **Strides** | 4, 2, 1 | 4, 2, 1 | 4, 2, 1 | 3, 2, 1 | — |
| **Activation** | ReLU | LeakyReLU* | ReLU | ReLU | ReLU (in classifier) |
| **Dropout** | 0.5 (linear) | 0.5 (linear) | None | None | 0.2 (classifier) |
| **Classifier** | 864→256→64→7 | 864→256→64→7 | 864→256→64→7 | 864→256→64→7 | 1280→512→7 |
| **Params** | ~0.5M | ~0.5M | ~0.5M | ~0.5M | ~5M (backbone frozen) |
| **Epochs** | 50 | 50 | 50 | 50 | 10 |
| **Batch size** | 32 | 32 | 32 | 32 | 32 |
| **Optimizer** | Adam (1e-3) | Adam (1e-3) | Adam (1e-3) | Adam (1e-3) | Adam / SGD / AdamW (1e-4) |
| **LR scheduler** | ReduceLROnPlateau | ReduceLROnPlateau | ReduceLROnPlateau | ReduceLROnPlateau | None |
| **Loss** | Focal (γ=2) + class weights | Focal (γ=2) + class weights | Focal (γ=2) + class weights | Focal (γ=2) + class weights | CrossEntropy |
| **Sampling** | WeightedRandomSampler (4×) | WeightedRandomSampler (4×) | WeightedRandomSampler (4×) | WeightedRandomSampler (4×) | Standard |
| **Augmentation** | Horizontal flip (2×) | Flips, rotation, color jitter, affine (on-the-fly) | Horizontal flip (2×) | Horizontal flip (2×) | Standard ImageNet transforms |
| **Pre-trained** | No | No | No | No | Yes (ImageNet) |

*Second CNN: configurable (relu, leaky_relu, gelu, silu, elu).

**Summary:**
- **Custom CNNs** share a similar 3-conv + 3-linear layout but differ in: input size (Image2: 160 vs 256), kernels (Image2: 9/5/3 vs 13/7/3), dropout (First/Second: 0.5; Image/Image2: none), and activation (Second: LeakyReLU vs ReLU). All use focal loss, class weights, and WeightedRandomSampler for imbalance. Second CNN has the richest augmentation (flips, rotation, color jitter, affine); others use horizontal flip only.
- **EfficientNet** uses a pre-trained backbone, freezes it, and trains only a small classifier head. Typically achieves higher accuracy with fewer epochs but requires more memory.

---

## Class Accuracy Analysis

Evaluate per-class accuracy for trained models.

**First CNN** (built-in in `first_cnn_torch.py`):
```bash
python3 first_cnn_torch.py --analyze [--model saves/first_cnn_model.pth] [--data_dir skincancer/organized] [--max_images 50] [--output saves/class_accuracy_first_cnn.png]
```

**Other models** (use `analyze_class_accuracy.py`):
```bash
python3 analyze_class_accuracy.py --model <path_to_model.pth> --model_type <type> [options]
```

**Examples:**
```bash
# First CNN: built-in analysis (no separate script)
python3 first_cnn_torch.py --analyze --model saves/first_cnn_model.pth --output saves/class_accuracy_first_cnn.png

# Other custom CNNs (must specify --model_type)
python3 analyze_class_accuracy.py --model saves/second_cnn_model.pth --model_type second_cnn --output saves/class_accuracy_second_cnn.png
python3 analyze_class_accuracy.py --model saves/image_cnn_model.pth --model_type image_cnn --output saves/class_accuracy_image_cnn.png
python3 analyze_class_accuracy.py --model saves/image2_cnn_model.pth --model_type image2_cnn --output saves/class_accuracy_image2_cnn.png

# EfficientNet (default model_type)
python3 analyze_class_accuracy.py --model skincancer_model_adam.pth --data_dir skincancer/organized --output saves/class_accuracy_efficientnet_adam.png
```

**Options:**
- `--model` — Path to trained model file (.pth)
- `--model_type` — Architecture: `first_cnn`, `second_cnn`, `image_cnn`, `image2_cnn`, or `efficientnet` (default)
- `--data_dir` — Path to organized data (default: `skincancer/organized`)
- `--max_images N` — Max images per class (default: 50)
- `--output` — Path for the accuracy plot (default: `class_accuracy_analysis.png`)
- `--device` — Device: `cpu`, `cuda`, `mps`, or `auto`

**What it does:**
- Tests images from each of the 7 classes
- Computes per-class accuracy
- Produces a bar chart (green ≥80%, orange 60–80%, red &lt;60%)
- Saves the plot to the specified output path

## Output Files

After training and analysis, you'll have:

**Custom CNN models (in `saves/`):**
- `first_cnn_model.pth`, `second_cnn_model.pth`, `image_cnn_model.pth`, `image2_cnn_model.pth`
- `training_history_first_cnn.png`, `training_history_second_cnn.png`, etc.

**EfficientNet (in project root):**
- `checkpoints_adam/`, `checkpoints_sgd/`, `checkpoints_adamw/` - Checkpoint directories
- `skincancer_model_adam.pth`, `skincancer_model_sgd.pth`, `skincancer_model_adamw.pth` - Final models
- `training_history_adam.png`, `training_history_sgd.png`, `training_history_adamw.png` - Training plots

**Class accuracy analysis (in `saves/`):**
- `class_accuracy_first_cnn.png`, `class_accuracy_second_cnn.png`, etc.

## Model Architecture

- **Base Model**: EfficientNet-B0 (pre-trained on ImageNet)
- **Custom Classifier**: 
  - Dropout (0.2)
  - Linear(1280 → 512)
  - ReLU
  - Dropout (0.2)
  - Linear(512 → 7)
  - Softmax

## Training Parameters

- Batch size: 32
- Learning rate: 0.0001
- Epochs: 10 (configurable)
- Train/Validation split: 80/20
- Random seed: 37 (for reproducibility)

## Optimizers Compared

1. **Adam**: Adaptive learning rate optimizer
2. **SGD**: Stochastic Gradient Descent with momentum (0.9)
3. **AdamW**: Adam with weight decay (0.01)

