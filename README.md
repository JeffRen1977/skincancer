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

3. **Train the model**:
```bash
python3 train_skincancer.py
```

This will:
- Train models with 3 different optimizers (Adam, SGD, AdamW)
- Save checkpoints after each epoch
- Save the best model for each optimizer
- Generate training history plots
- Save final models

4. **Run inference** on a new image:
```bash
python3 inference.py --model skincancer_model_adam.pth --image path/to/image.jpg
```

## Output Files

After training, you'll have:
- `checkpoints_adam/`, `checkpoints_sgd/`, `checkpoints_adamw/` - Checkpoint directories
- `skincancer_model_adam.pth`, `skincancer_model_sgd.pth`, `skincancer_model_adamw.pth` - Final models
- `training_history_adam.png`, `training_history_sgd.png`, `training_history_adamw.png` - Training plots

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

