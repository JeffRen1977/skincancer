# Skin Cancer Classification - Transfer Learning Project

PyTorch-based transfer learning project for skin cancer classification using the HAM10000 dataset.

## Project Requirements Met

✅ **Framework**: PyTorch (instead of TensorFlow)  
✅ **Model**: EfficientNet-B0 (different from ResNet)  
✅ **Model Architecture**: Class-based model (instead of functional/sequential)  
✅ **Model Saving**: Saves final model and checkpoints during training  
✅ **Inference**: Separate inference script to classify images  
✅ **Visualization**: Matplotlib plots for 4 metrics (train loss, val loss, train acc, val acc)  
✅ **Comparison Study**: Compares 3 optimizers (Adam, SGD, AdamW)  

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

1. Visit the Dataset Ninja page for HAM10000: https://datasetninja.com/skin-cancer-ham10000  
2. Click **Download** to grab the dataset in Supervisely format (≈2.6 GB) or use the `dataset-tools` helper:
   ```bash
   pip install --upgrade dataset-tools
   python - <<'PY'
   import dataset_tools as dtools
   dtools.download(dataset='Skin Cancer: HAM10000', dst_dir='~/dataset-ninja/')
   PY
   ```
3. Extract the archive so that the raw images and annotations are available locally, then run `python3 organize_data.py` (next section) to arrange them into the structure expected by this project.

## Installation

Install required dependencies:
```bash
pip3 install -r requirements.txt
```

For Mac with Apple Silicon (M1/M2/M3), PyTorch will automatically use MPS (Metal Performance Shaders) for GPU acceleration.

## Setup

1. **Organize the dataset** (run this first):
```bash
python3 organize_data.py
```

This will create `skincancer/organized/` with images organized by class folders.

2. **Train the model**:
```bash
python3 train_skincancer.py
```

This will:
- Train models with 3 different optimizers (Adam, SGD, AdamW)
- Save checkpoints after each epoch
- Save the best model for each optimizer
- Generate training history plots
- Save final models

3. **Run inference** on a new image:
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


