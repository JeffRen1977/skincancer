# Run Instructions

## All Models (train + analyze)

```bash
python train_and_analyze_all.py
```

Options: `--skip-training`, `--skip-analysis`, `--skip-efficientnet`

---

## Individual Models

### First CNN

```bash
python first_cnn_torch.py              # train
python first_cnn_torch.py --analyze   # analyze
```

### Second CNN

```bash
python second_cnn_torch.py              # train
python second_cnn_torch.py --analyze   # analyze
```

### Image CNN

```bash
python image_cnn_torch.py              # train
python image_cnn_torch.py --analyze   # analyze
```

### Image2 CNN

```bash
python image2_cnn_torch.py              # train
python image2_cnn_torch.py --analyze   # analyze
```

### EfficientNet (Adam, SGD, AdamW)

```bash
python train_skincancer.py   # train (3 optimizers)
python analyze_class_accuracy.py --model saves/skincancer_model_adam.pth --model_type efficientnet --output saves/class_accuracy_efficientnet_adam.png
python analyze_class_accuracy.py --model saves/skincancer_model_sgd.pth --model_type efficientnet --output saves/class_accuracy_efficientnet_sgd.png
python analyze_class_accuracy.py --model saves/skincancer_model_adamw.pth --model_type efficientnet --output saves/class_accuracy_efficientnet_adamw.png
```
