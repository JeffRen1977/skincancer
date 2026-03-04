"""
Custom CNN for skin cancer classification (PyTorch).

This script defines a convolutional neural network from scratch (no pre-trained
models), loads images from class folders, and supports training with dropout.
Data is expected in a root folder with one subfolder per class (e.g. skincancer/organized/).
"""

import torch
import pathlib
import cv2
import time
import atexit
import torchvision.transforms.v2
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Global configuration (adjust paths/device as needed)
# -----------------------------------------------------------------------------
INPUT_SHAPE = (3, 256, 256)   # (channels, height, width) — PyTorch uses channels-first
IMAGES_PATH = pathlib.Path("skincancer/organized")  # One subfolder per class (from organize_data.py)
BATCH_SIZE = 32               # Number of samples per training step
NUM_EPOCHS = 30               # More epochs needed for imbalanced data
LEARNING_RATE = 1e-3           # Step size for optimizer
TIME_STAMP = time.strftime("%Y_%m_%de_%H_%M")  # Used for unique save filenames


@atexit.register
def clean_up() -> None:
    """Run when the script exits (normal or error). Saves the current model and epoch."""
    torch.save(model, "saves/model_" + str(epoch) + "_" + TIME_STAMP)

# -----------------------------------------------------------------------------
# Dataset: load images from disk and serve (image, label) pairs
# -----------------------------------------------------------------------------
class Dataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset that loads all images from class folders into memory.
    Expects: image_path has one subfolder per class, each containing image files.
    """
    def __init__(self, image_path: pathlib.Path):
        """
        Scan image_path for class folders and load every image.
        Train/validation are not split here; that is done later by get_dataloaders.
        """
        print("Loading images...")
        # Use sorted order so class indices match analyze_class_accuracy.get_class_mapping()
        self.class_names = sorted(
            p.name for p in image_path.iterdir() if p.is_dir()
        )
        self.images = []        # list of (tensor image, tensor label)
        for path in sorted(image_path.iterdir(), key=lambda p: p.name):
            if path.is_dir():
                for image_file in path.iterdir():
                    img = cv2.imread(image_file)
                    if img is not None:
                        # Label = index of this class (integer 0..num_classes-1)
                        label = torch.tensor(
                            self.class_names.index(path.name),
                            dtype=torch.float32,
                            device='mps',
                        )
                        # OpenCV loads BGR; convert to RGB for consistency with typical pretrained models
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        # Resize to model input size (height, width)
                        img = cv2.resize(img, INPUT_SHAPE[1:])
                        # Normalize pixel values to [0, 1]
                        img = img / 255.0
                        # PyTorch expects (C, H, W); OpenCV gives (H, W, C) — move channels first
                        img = img.transpose([2, 0, 1])
                        img = torch.tensor(
                            img,
                            dtype=torch.float32,
                            device='mps',
                        )
                        self.images.append((img, label))
        print("Images loaded.")

    def __len__(self) -> int:
        """Total number of samples in the dataset."""
        return len(self.images)

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        """Return the idx-th (image, label) pair. Required by DataLoader."""
        return self.images[idx]
                                             
# -----------------------------------------------------------------------------
# Model: custom CNN (conv layers + fully connected head with dropout)
# -----------------------------------------------------------------------------
class Model(torch.nn.Module):
    """
    Convolutional neural network for image classification.
    Architecture: conv blocks (feature extraction) -> flatten -> linear layers + dropout -> logits.
    """
    def __init__(self, input_shape: (int, int, int)):
        """
        Build all layers. input_shape is (C, H, W), channels-first.
        Layer sizes are chosen so that after conv/maxpool the feature map flattens to 864.
        """
        super().__init__()
        print(f"{'Input Shape:':>30}", input_shape)

        # Zero-padding: add 1 pixel on left, right, top, bottom (keeps spatial size or helps with odd sizes)
        self.zp1 = torch.nn.ZeroPad2d((1, 1, 1, 1))

        # First conv block: 3 -> 24 channels, large kernel, stride 4 to reduce size quickly
        self.conv1 = torch.nn.Conv2d(
            in_channels=3,
            out_channels=24,
            kernel_size=13,
            stride=4,
        )

        # ReLU used after each conv (non-linearity; we reuse one module)
        self.relu = torch.nn.ReLU()

        # Max pooling: take max in 3x3 windows, stride 2 — reduces spatial dimensions
        self.maxpool1 = torch.nn.MaxPool2d(
            kernel_size=3,
            stride=2,
        )

        # Second conv block: 24 -> 48 channels
        self.conv2 = torch.nn.Conv2d(
            in_channels=24,
            out_channels=48,
            kernel_size=7,
            stride=2,
        )

        self.maxpool2 = torch.nn.MaxPool2d(
            kernel_size=3,
            stride=2,
        )

        # Third conv block: 48 -> 96 channels, smaller kernel
        self.conv3 = torch.nn.Conv2d(
            in_channels=48,
            out_channels=96,
            kernel_size=3,
            stride=1,
        )

        # Flatten: (batch, channels, H, W) -> (batch, channels*H*W) for the linear layers
        self.flatten = torch.nn.Flatten()

        # Fully connected (dense) head — also called MLP (Multi-Layer Perceptron)
        # in_features=864 must match the flattened size after all conv/maxpool layers
        self.linear1 = torch.nn.Linear(
            in_features=864,
            out_features=256,
        )
        self.linear2 = torch.nn.Linear(
            in_features=256,
            out_features=64,
        )
        # Output: 7 units for 7 skin cancer classes (logits; no softmax here — use CrossEntropyLoss)
        self.linear3 = torch.nn.Linear(
            in_features=64,
            out_features=7,
        )

        # Dropout: during training, randomly zero some outputs (p=0.5). Reduces overfitting.
        # Automatically disabled in eval mode (model.eval()).
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        One forward pass: input batch of images -> output logits (one per class).
        Data flow: pad -> conv1 -> relu -> pool -> conv2 -> relu -> pool -> conv3 -> relu
                  -> flatten -> linear1 -> dropout -> linear2 -> dropout -> linear3 -> logits.
        """
        y = self.zp1(x)
        y = self.relu(self.conv1(y))
        y = self.maxpool1(y)
        y = self.relu(self.conv2(y))
        y = self.maxpool2(y)
        y = self.relu(self.conv3(y))
        y = self.flatten(y)
        y = self.linear1(y)
        y = self.dropout(y)
        y = self.linear2(y)
        y = self.dropout(y)
        y = self.linear3(y)
        return y

# -----------------------------------------------------------------------------
# Data loaders: split data into train/validation and optionally augment
# -----------------------------------------------------------------------------
def get_dataloaders(
    dataset: Dataset,
    train_prop: float,
    batch_size: int,
    n_classes: int,
) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
    """
    Split dataset into train and validation, add augmented copies to train set,
    and wrap in DataLoaders. Uses WeightedRandomSampler to oversample minority classes.
    Returns (train_loader, validation_loader).
    """
    generator = torch.Generator().manual_seed(37)
    train_set, validation_set = torch.utils.data.random_split(
        dataset,
        lengths=[train_prop, 1 - train_prop],
        generator=generator,
    )
    print(f"Number of images in training set before augmentation: ", end="")
    print(f"{len(train_set)}")
    aug_set = torchvision.transforms.v2.RandomHorizontalFlip(1.0)(train_set)
    train_set = torch.utils.data.ConcatDataset([train_set, aug_set])
    print(f"Number of images in training set after augmentation: ", end="")
    print(f"{len(train_set)}")

    # WeightedRandomSampler: oversample minority classes so model sees them more often
    train_class_counts = [0] * n_classes
    for i in range(len(train_set)):
        _, label = train_set[i]
        idx = int(label.item()) if label.dim() == 0 else int(label[0].item())
        train_class_counts[idx] += 1
    sample_weights = []
    for i in range(len(train_set)):
        _, label = train_set[i]
        idx = int(label.item()) if label.dim() == 0 else int(label[0].item())
        sample_weights.append(1.0 / max(1, train_class_counts[idx]))
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )
    print("Using WeightedRandomSampler to oversample minority classes")
    train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=sampler)
    validation = torch.utils.data.DataLoader(validation_set, batch_size=batch_size)
    return train, validation


def main():
    """Load data, build model, and run training."""
    global model, epoch
    epoch = 0

    pathlib.Path("saves").mkdir(exist_ok=True)

    # Load all images from class folders into a single Dataset
    dataset = Dataset(IMAGES_PATH)
    print(f"Found {len(dataset)} images.")

    # Split into train/validation and create DataLoaders (with augmentation on train)
    train_loader, validation_loader = get_dataloaders(dataset, 0.8, BATCH_SIZE, len(dataset.class_names))

    # Device: 'mps' (Apple Silicon), 'cuda' (NVIDIA), or 'cpu'
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Build model and move to device
    model = Model(INPUT_SHAPE).to(device)

    # Class weights to handle imbalance (melanocytic_nevi ~71% of data)
    class_counts = [0] * len(dataset.class_names)
    for _, label in dataset.images:
        idx = int(label.item()) if label.dim() == 0 else int(label[0].item())
        class_counts[idx] += 1
    total = sum(class_counts)
    n_classes = len(class_counts)
    class_weights = torch.tensor(
        [total / (n_classes * c) if c > 0 else 1.0 for c in class_counts],
        dtype=torch.float32, device=device
    )
    print(f"Class weights (inverse frequency): {[f'{w:.2f}' for w in class_weights.tolist()]}")

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # Open log file for writing metrics
    output_file = open("saves/printout_" + TIME_STAMP + ".txt", "a")

    # History for plotting: train_loss, train_acc, val_loss, val_acc per epoch
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # ----- Training loop -----
    for epoch in range(1, NUM_EPOCHS + 1):
        # --- Train one epoch ---
        model.train()  # Enables dropout and train-time behavior
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # Labels are float32 from Dataset; CrossEntropyLoss needs long (integer) indices
            labels = labels.long()
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total if train_total else 0.0

        # --- Validation (no gradient needed) ---
        model.eval()  # Disables dropout for consistent evaluation
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                labels = labels.long()
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss /= max(1, len(validation_loader))
        val_acc = 100.0 * correct / total if total else 0.0

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        line = f"Epoch {epoch}/{NUM_EPOCHS}  train_loss={train_loss:.4f}  train_acc={train_acc:.2f}%  val_loss={val_loss:.4f}  val_acc={val_acc:.2f}%\n"
        print(line.strip())
        output_file.write(line)
        output_file.flush()

    output_file.close()

    # Plot and save training history (same style as training_history_adam.png)
    plot_path = "saves/training_history_first_cnn.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], "b-", label="Training Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "r-", label="Validation Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True)
    ax2.plot(epochs, history["train_acc"], "b-", label="Training Accuracy", linewidth=2)
    ax2.plot(epochs, history["val_acc"], "r-", label="Validation Accuracy", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Training history plot saved: {plot_path}")

    # Save final model (weights only, for inference) — similar to train_skincancer.py
    final_model_path = "saves/first_cnn_model.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model saved: {final_model_path}")

    print("Training finished.")


if __name__ == "__main__":
    main()
