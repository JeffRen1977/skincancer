"""
Custom CNN for skin cancer classification (PyTorch).
From scratch, no pre-trained models. Data: one subfolder per class.

Run training:  python first_cnn_torch.py
Run analysis:  python first_cnn_torch.py --analyze
"""

import torch
import torch.nn.functional as F
import pathlib
import cv2
import time
import atexit
import random
import argparse
import numpy as np
from collections import defaultdict
import torchvision.transforms.v2
import matplotlib.pyplot as plt

# Config
INPUT_SHAPE = (3, 256, 256)
IMAGES_PATH = pathlib.Path("skincancer/organized")
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
TIME_STAMP = time.strftime("%Y_%m_%de_%H_%M")


def clean_up() -> None:
    torch.save(model, "saves/model_" + str(epoch) + "_" + TIME_STAMP)


class Dataset(torch.utils.data.Dataset):
    """Load images from class folders into memory."""

    def __init__(self, image_path: pathlib.Path):
        print("Loading images...")
        self.class_names = sorted(
            p.name for p in image_path.iterdir() if p.is_dir()
        )
        self.images = []
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        for path in sorted(image_path.iterdir(), key=lambda p: p.name):
            if path.is_dir():
                for f in path.iterdir():
                    img = cv2.imread(str(f))
                    if img is None:
                        continue
                    label = torch.tensor(
                        self.class_names.index(path.name),
                        dtype=torch.float32, device=dev
                    )
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, INPUT_SHAPE[1:]) / 255.0
                    img = torch.tensor(
                        img.transpose([2, 0, 1]),
                        dtype=torch.float32, device=dev
                    )
                    self.images.append((img, label))
        print("Images loaded.")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        return self.images[idx]


class Model(torch.nn.Module):
    """CNN: conv blocks -> flatten -> linear + dropout -> 7 logits."""

    def __init__(self, input_shape):
        super().__init__()
        print(f"{'Input Shape:':>30}", input_shape)
        self.zp1 = torch.nn.ZeroPad2d((1, 1, 1, 1))
        self.conv1 = torch.nn.Conv2d(3, 24, kernel_size=13, stride=4)
        self.relu = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = torch.nn.Conv2d(24, 48, kernel_size=7, stride=2)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = torch.nn.Conv2d(48, 96, kernel_size=3, stride=1)
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(864, 256)
        self.linear2 = torch.nn.Linear(256, 64)
        self.linear3 = torch.nn.Linear(64, 7)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        y = self.zp1(x)
        y = self.relu(self.conv1(y))
        y = self.maxpool1(y)
        y = self.relu(self.conv2(y))
        y = self.maxpool2(y)
        y = self.relu(self.conv3(y))
        y = self.flatten(y)
        y = self.dropout(self.linear1(y))
        y = self.dropout(self.linear2(y))
        return self.linear3(y)


def get_dataloaders(dataset, train_prop, batch_size, n_classes):
    """Split train/val, augment, WeightedRandomSampler 4x."""
    gen = torch.Generator().manual_seed(37)
    train_set, val_set = torch.utils.data.random_split(
        dataset, lengths=[train_prop, 1 - train_prop], generator=gen
    )
    aug = torchvision.transforms.v2.RandomHorizontalFlip(1.0)(train_set)
    train_set = torch.utils.data.ConcatDataset([train_set, aug])
    print(f"Train: {len(train_set)}, Val: {len(val_set)}")

    counts = [0] * n_classes
    for i in range(len(train_set)):
        _, lb = train_set[i]
        idx = int(lb.item()) if lb.dim() == 0 else int(lb[0].item())
        counts[idx] += 1
    weights = []
    for i in range(len(train_set)):
        _, lb = train_set[i]
        idx = int(lb.item()) if lb.dim() == 0 else int(lb[0].item())
        weights.append(1.0 / max(1, counts[idx]))
    sampler = torch.utils.data.WeightedRandomSampler(
        weights, num_samples=4 * len(weights), replacement=True
    )
    train_ld = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, sampler=sampler
    )
    val_ld = torch.utils.data.DataLoader(val_set, batch_size=batch_size)
    return train_ld, val_ld


def _get_class_mapping():
    dirs = ['actinic_keratoses', 'basal_cell_carcinoma',
            'benign_keratosis-like_lesions', 'dermatofibroma',
            'melanocytic_nevi', 'melanoma', 'vascular_lesions']
    disp = ['actinic keratoses', 'basal cell carcinoma',
            'benign keratosis-like lesions', 'dermatofibroma',
            'melanocytic nevi', 'melanoma', 'vascular lesions']
    return dirs, disp


def _preprocess(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, INPUT_SHAPE[1:]) / 255.0
    t = torch.tensor(img.transpose([2, 0, 1]), dtype=torch.float32)
    return t.unsqueeze(0)


def analyze_class_accuracy(
    model_path="saves/first_cnn_model.pth",
    data_dir=None,
    max_images_per_class=50,
    output_path="saves/class_accuracy_first_cnn.png",
    device=None,
):
    """Analyze per-class accuracy, save bar chart."""
    data_dir = data_dir or IMAGES_PATH
    data_path = pathlib.Path(data_dir)
    if device is None:
        device = ("mps" if torch.backends.mps.is_available() else
                  "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device) if isinstance(device, str) else device

    dir_names, display_names = _get_class_mapping()
    model = Model(INPUT_SHAPE).to(device)
    try:
        sd = torch.load(model_path, map_location=device)
        model.load_state_dict(sd.get('model_state_dict', sd))
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    model.eval()

    stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'confidences': []})
    all_images = {}
    for i, d in enumerate(dir_names):
        p = data_path / d
        if p.exists():
            imgs = list(p.glob('*.jpg')) + list(p.glob('*.png'))
            random.shuffle(imgs)
            all_images[i] = imgs[:max_images_per_class]
            print(f"Found {len(imgs)} in {d}, testing {len(all_images[i])}")
        else:
            all_images[i] = []

    print("\n" + "=" * 80)
    print("CLASS-WISE ACCURACY (First CNN)")
    print("=" * 80)

    for cidx in range(len(display_names)):
        imgs = all_images.get(cidx, [])
        if not imgs:
            continue
        for path in imgs:
            t = _preprocess(path)
            if t is None:
                continue
            with torch.no_grad():
                logits = model(t.to(device))
                probs = F.softmax(logits[0], dim=0).cpu().numpy()
                pred = logits[0].argmax().item()
            stats[cidx]['total'] += 1
            if pred == cidx:
                stats[cidx]['correct'] += 1
            stats[cidx]['confidences'].append(probs[pred])
        acc = 100.0 * stats[cidx]['correct'] / stats[cidx]['total']
        c, t = stats[cidx]['correct'], stats[cidx]['total']
        print(f"{display_names[cidx]}: {acc:.2f}% ({c}/{t})")

    print("\n" + "=" * 80)
    def _key(x):
        return (x[1]['correct'] / x[1]['total']) if x[1]['total'] > 0 else 0
    sorted_c = sorted(stats.items(), key=_key, reverse=True)
    for rank, (cidx, s) in enumerate(sorted_c, 1):
        if s['total'] > 0:
            acc = 100.0 * s['correct'] / s['total']
            avg = np.mean(s['confidences']) * 100
            n = display_names[cidx]
            print(f"{rank}. {n:35s} {acc:6.2f}%  Avg conf: {avg:.2f}%")

    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    accs = [
        100.0 * s['correct'] / s['total']
        for _, s in sorted_c if s['total'] > 0
    ]
    labels = [display_names[c] for c, s in sorted_c if s['total'] > 0]
    colors = [
        'green' if a >= 80 else 'orange' if a >= 60 else 'red'
        for a in accs
    ]
    bars = ax.barh(range(len(accs)), accs, color=colors)
    ax.set_yticks(range(len(accs)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Accuracy (%)')
    ax.set_title('Per-Class Accuracy (First CNN)')
    ax.set_xlim([0, 100])
    ax.grid(axis='x', alpha=0.3)
    for i, (bar, a) in enumerate(zip(bars, accs)):
        ax.text(a + 1, i, f'{a:.1f}%', va='center',
                fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {output_path}")
    return dict(stats)


def main():
    parser = argparse.ArgumentParser(description="First CNN: train or analyze")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--model", default="saves/first_cnn_model.pth")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--max_images", type=int, default=50)
    parser.add_argument(
        "--output", default="saves/class_accuracy_first_cnn.png"
    )
    args = parser.parse_args()

    if args.analyze:
        analyze_class_accuracy(
            model_path=args.model,
            data_dir=args.data_dir or str(IMAGES_PATH),
            max_images_per_class=args.max_images,
            output_path=args.output,
        )
        return

    global model, epoch
    epoch = 0
    atexit.register(clean_up)
    pathlib.Path("saves").mkdir(exist_ok=True)

    dataset = Dataset(IMAGES_PATH)
    print(f"Found {len(dataset)} images.")
    train_ld, val_ld = get_dataloaders(
        dataset, 0.8, BATCH_SIZE, len(dataset.class_names)
    )

    device = ("mps" if torch.backends.mps.is_available() else
              "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = Model(INPUT_SHAPE).to(device)

    counts = [0] * len(dataset.class_names)
    for _, lb in dataset.images:
        idx = int(lb.item()) if lb.dim() == 0 else int(lb[0].item())
        counts[idx] += 1
    total = sum(counts)
    n = len(counts)
    weights = torch.tensor(
        [total / (n * c) if c > 0 else 1.0 for c in counts],
        dtype=torch.float32, device=device
    )

    class FocalLoss(torch.nn.Module):
        def __init__(self, weight=None, gamma=2.0):
            super().__init__()
            self.weight, self.gamma = weight, gamma

        def forward(self, logits, targets):
            ce = F.cross_entropy(
                logits, targets, weight=self.weight, reduction='none'
            )
            return ((1 - torch.exp(-ce)) ** self.gamma * ce).mean()

    criterion = FocalLoss(weight=weights, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    out = open("saves/printout_" + TIME_STAMP + ".txt", "a")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        run_loss, corr, tot = 0.0, 0, 0
        for imgs, lbls in train_ld:
            imgs, lbls = imgs.to(device), lbls.long().to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            _, pred = logits.max(1)
            tot += lbls.size(0)
            corr += (pred == lbls).sum().item()
        train_loss = run_loss / len(train_ld)
        train_acc = 100.0 * corr / tot if tot else 0.0

        model.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_ld:
                imgs, lbls = imgs.to(device), lbls.long().to(device)
                logits = model(imgs)
                v_loss += criterion(logits, lbls).item()
                _, pred = logits.max(1)
                v_tot += lbls.size(0)
                v_corr += (pred == lbls).sum().item()
        val_loss = v_loss / max(1, len(val_ld))
        val_acc = 100.0 * v_corr / v_tot if v_tot else 0.0
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        line = (f"Epoch {epoch}/{NUM_EPOCHS}  train_loss={train_loss:.4f}  "
                f"train_acc={train_acc:.2f}%  val_loss={val_loss:.4f}  "
                f"val_acc={val_acc:.2f}%\n")
        print(line.strip())
        out.write(line)
        out.flush()

    out.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ep = range(1, len(history["train_loss"]) + 1)
    ax1.plot(ep, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    ax1.plot(ep, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)
    ax2.plot(ep, history["train_acc"], "b-", label="Train Acc", linewidth=2)
    ax2.plot(ep, history["val_acc"], "r-", label="Val Acc", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(
        "saves/training_history_first_cnn.png",
        dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("Training history saved: saves/training_history_first_cnn.png")

    torch.save(model.state_dict(), "saves/first_cnn_model.pth")
    print("Model saved: saves/first_cnn_model.pth")
    print("Training finished.")


if __name__ == "__main__":
    main()
