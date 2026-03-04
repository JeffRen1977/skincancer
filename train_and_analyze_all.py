#!/usr/bin/env python3
"""
Train all skin cancer models and run class accuracy analysis on each.

This script:
1. Trains the four custom CNNs (first_cnn, second_cnn, image_cnn, image2_cnn)
2. Trains EfficientNet via train_skincancer.py (Adam, SGD, AdamW)
3. Runs analyze_class_accuracy.py on each trained model
4. Saves per-model accuracy plots to saves/

Usage:
    python train_and_analyze_all.py [--skip-training] [--skip-analysis] [--skip-efficientnet]

Options:
    --skip-training    Skip training; only run analysis on existing models
    --skip-analysis    Skip analysis; only train models
    --skip-efficientnet  Skip train_skincancer.py (saves ~30 min; trains 3 optimizers)
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd, description):
    """Run a command and exit on failure."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"  $ {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parent)
    if result.returncode != 0:
        print(f"\nError: Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Train all models and run class accuracy analysis")
    parser.add_argument("--skip-training", action="store_true", help="Skip training; only run analysis")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip analysis; only train models")
    parser.add_argument("--skip-efficientnet", action="store_true",
                        help="Skip train_skincancer.py (trains 3 optimizers, takes longer)")
    parser.add_argument("--max-images", type=int, default=50,
                        help="Max images per class for analysis (default: 50)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    root.mkdir(exist_ok=True)
    saves = root / "saves"
    saves.mkdir(exist_ok=True)

    # -------------------------------------------------------------------------
    # Training: four custom CNNs + EfficientNet (train_skincancer)
    # -------------------------------------------------------------------------
    if not args.skip_training:
        run([sys.executable, "first_cnn_torch.py"], "Training First CNN (first_cnn_torch.py)")
        run([sys.executable, "second_cnn_torch.py"], "Training Second CNN (second_cnn_torch.py)")
        run([sys.executable, "image_cnn_torch.py"], "Training Image CNN (image_cnn_torch.py)")
        run([sys.executable, "image2_cnn_torch.py"], "Training Image2 CNN (image2_cnn_torch.py)")

        if not args.skip_efficientnet:
            run([sys.executable, "train_skincancer.py"], "Training EfficientNet (train_skincancer.py)")

    # -------------------------------------------------------------------------
    # Analysis: first_cnn uses built-in --analyze; others use analyze_class_accuracy.py
    # -------------------------------------------------------------------------
    if not args.skip_analysis:
        # First CNN: built-in analyze in first_cnn_torch.py
        first_cnn_path = root / "saves/first_cnn_model.pth"
        if first_cnn_path.exists():
            run(
                [sys.executable, "first_cnn_torch.py", "--analyze",
                 "--model", "saves/first_cnn_model.pth",
                 "--data_dir", "skincancer/organized",
                 "--max_images", str(args.max_images),
                 "--output", "saves/class_accuracy_first_cnn.png"],
                "Analyzing first_cnn (first_cnn_torch.py --analyze)"
            )
        else:
            print("\nSkipping analysis for first_cnn: saves/first_cnn_model.pth not found")

        # Second CNN: built-in analyze in second_cnn_torch.py
        second_cnn_path = root / "saves/second_cnn_model.pth"
        if second_cnn_path.exists():
            run(
                [sys.executable, "second_cnn_torch.py", "--analyze",
                 "--model", "saves/second_cnn_model.pth",
                 "--data_dir", "skincancer/organized",
                 "--max_images", str(args.max_images),
                 "--output", "saves/class_accuracy_second_cnn.png"],
                "Analyzing second_cnn (second_cnn_torch.py --analyze)"
            )
        else:
            print("\nSkipping analysis for second_cnn: saves/second_cnn_model.pth not found")

        # Other custom CNNs + EfficientNet: use analyze_class_accuracy.py
        analyze_cmd = [
            sys.executable, "analyze_class_accuracy.py",
            "--data_dir", "skincancer/organized",
            "--max_images", str(args.max_images),
        ]
        models = [
            ("saves/image_cnn_model.pth", "image_cnn", "saves/class_accuracy_image_cnn.png"),
            ("saves/image2_cnn_model.pth", "image2_cnn", "saves/class_accuracy_image2_cnn.png"),
        ]
        for model_path, model_type, output_path in models:
            path = root / model_path
            if not path.exists():
                print(f"\nSkipping analysis for {model_type}: {model_path} not found")
                continue
            run(
                analyze_cmd + ["--model", model_path, "--model_type", model_type, "--output", output_path],
                f"Analyzing {model_type} ({model_path})"
            )

        # EfficientNet models (from train_skincancer)
        efficientnet_models = [
            ("skincancer_model_adam.pth", "saves/class_accuracy_efficientnet_adam.png"),
            ("skincancer_model_sgd.pth", "saves/class_accuracy_efficientnet_sgd.png"),
            ("skincancer_model_adamw.pth", "saves/class_accuracy_efficientnet_adamw.png"),
        ]
        for model_path, output_path in efficientnet_models:
            path = root / model_path
            if not path.exists():
                print(f"\nSkipping analysis for {model_path}: not found")
                continue
            run(
                analyze_cmd + ["--model", model_path, "--model_type", "efficientnet", "--output", output_path],
                f"Analyzing EfficientNet ({model_path})"
            )

    print("\n" + "="*60)
    print("  DONE")
    print("="*60)
    print("\nTraining outputs: saves/*.pth, saves/training_history_*.png")
    print("Analysis outputs: saves/class_accuracy_*.png")


if __name__ == "__main__":
    main()
