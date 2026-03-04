"""
Custom CNN for skin cancer classification (PyTorch) — focus on activation functions.

Same architecture as first_cnn_torch.py, but the activation function after each
conv layer is configurable (ReLU, LeakyReLU, GELU, SiLU, ELU). Change ACTIVATION
at the top to compare different activations. Also uses data augmentation on
training data. Data is expected in a root folder with one subfolder per class
(e.g. skincancer/organized/).
"""

import torch
import pathlib
import cv2
import time
import atexit
import torchvision.transforms.v2 as T
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Global configuration (adjust paths/device as needed)
# -----------------------------------------------------------------------------
INPUT_SHAPE = (3, 256, 256)   # (channels, height, width) — PyTorch uses channels-first
IMAGES_PATH = pathlib.Path("skincancer/organized")  # One subfolder per class (from organize_data.py)
BATCH_SIZE = 32               # Number of samples per training step
NUM_EPOCHS = 30               # More epochs needed for imbalanced data
LEARNING_RATE = 1e-3          # Step size for optimizer
TIME_STAMP = time.strftime("%Y_%m_%de_%H_%M")  # Used for unique save filenames

# Activation function after each conv layer (main difference from first_cnn_torch.py).
# Options: "relu", "leaky_relu", "gelu", "silu", "elu"
ACTIVATION = "leaky_relu"


@atexit.register
def clean_up() -> None:
    """Run when the script exits (normal or error). Saves the current model and epoch."""
    torch.save(model, "saves/model_second_" + str(epoch) + "_" + TIME_STAMP)


# -----------------------------------------------------------------------------
# Activation function (focus of this script: try different activations)
# -----------------------------------------------------------------------------
def get_activation(name: str) -> torch.nn.Module:
    """Return the activation module for the given name. Used after each conv layer."""
    name = name.lower()
    if name == "relu":
        return torch.nn.ReLU()
    if name == "leaky_relu":
        return torch.nn.LeakyReLU(negative_slope=0.01)
    if name == "gelu":
        return torch.nn.GELU()
    if name == "silu":
        return torch.nn.SiLU()
    if name == "elu":
        return torch.nn.ELU(alpha=1.0)
    raise ValueError(f"Unknown activation: {name}. Use one of: relu, leaky_relu, gelu, silu, elu")


# -----------------------------------------------------------------------------
# Data augmentation: training-time transforms (applied on the fly per batch)
# -----------------------------------------------------------------------------
def get_train_transform():
    """
    Build a composition of random transforms for training only.
    Each batch will see different random augmentations (reduces overfitting).
    """
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=30),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
    ])


# -----------------------------------------------------------------------------
# Wrapper: apply an optional transform in __getitem__ (for train vs val)
# -----------------------------------------------------------------------------
class TransformWrapper(torch.utils.data.Dataset):
    """Wraps a dataset and applies an optional transform to the image in __getitem__."""

    def __init__(self, dataset: torch.utils.data.Dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, label = self.dataset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# -----------------------------------------------------------------------------
# Dataset: load images from disk and serve (image, label) pairs
# -----------------------------------------------------------------------------
class Dataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset that loads all images from class folders into memory.
    Expects: image_path has one subfolder per class, each containing image files.
    Loads to CPU so that augmentation (on CPU) can be applied before moving to device.
    """
    def __init__(self, image_path: pathlib.Path, device: str = "cpu"):
        """
        Scan image_path for class folders and load every image.
        device: where to put tensors ('cpu' recommended when using augmentation).
        """
        print("Loading images...")
        # Use sorted order so class indices match analyze_class_accuracy.get_class_mapping()
        self.class_names = sorted(
            p.name for p in image_path.iterdir() if p.is_dir()
        )
        self.images = []
        for path in sorted(image_path.iterdir(), key=lambda p: p.name):
            if path.is_dir():
                for image_file in path.iterdir():
                    img = cv2.imread(image_file)
                    if img is not None:
                        label = torch.tensor(
                            self.class_names.index(path.name),
                            dtype=torch.float32,
                            device=device,
                        )
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, INPUT_SHAPE[1:])
                        img = img / 255.0
                        img = img.transpose([2, 0, 1])
                        img = torch.tensor(
                            img,
                            dtype=torch.float32,
                            device=device,
                        )
                        self.images.append((img, label))
        print("Images loaded.")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        return self.images[idx]


# -----------------------------------------------------------------------------
# Model: custom CNN with configurable activation (same layout as first_cnn_torch.py)
# -----------------------------------------------------------------------------
class Model(torch.nn.Module):
    """
    Convolutional neural network for image classification.
    Same architecture as first_cnn_torch.py; the activation after each conv layer
    is set by ACTIVATION (relu, leaky_relu, gelu, silu, elu).
    """
    def __init__(self, input_shape: (int, int, int), activation_name: str = ACTIVATION):
        super().__init__()
        print(f"{'Input Shape:':>30}", input_shape)
        print(f"{'Activation:':>30}", activation_name)
        self.zp1 = torch.nn.ZeroPad2d((1, 1, 1, 1))
        self.conv1 = torch.nn.Conv2d(
            in_channels=3,
            out_channels=24,
            kernel_size=13,
            stride=4,
        )
        self.act = get_activation(activation_name)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = torch.nn.Conv2d(
            in_channels=24,
            out_channels=48,
            kernel_size=7,
            stride=2,
        )
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = torch.nn.Conv2d(
            in_channels=48,
            out_channels=96,
            kernel_size=3,
            stride=1,
        )
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(864, 256)
        self.linear2 = torch.nn.Linear(256, 64)
        self.linear3 = torch.nn.Linear(64, 7)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.zp1(x)
        y = self.act(self.conv1(y))
        y = self.maxpool1(y)
        y = self.act(self.conv2(y))
        y = self.maxpool2(y)
        y = self.act(self.conv3(y))
        y = self.flatten(y)
        y = self.linear1(y)
        y = self.dropout(y)
        y = self.linear2(y)
        y = self.dropout(y)
        y = self.linear3(y)
        return y


# -----------------------------------------------------------------------------
# Data loaders: split data, wrap train with augmentation
# -----------------------------------------------------------------------------
def get_dataloaders(
    dataset: Dataset,
    train_prop: float,
    batch_size: int,
    n_classes: int,
    train_transform=None,
) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
    """
    Split dataset into train and validation.
    Uses WeightedRandomSampler to oversample minority classes.
    Returns (train_loader, validation_loader).
    """
    generator = torch.Generator().manual_seed(37)
    train_set, validation_set = torch.utils.data.random_split(
        dataset,
        lengths=[train_prop, 1 - train_prop],
        generator=generator,
    )
    print(f"Number of images in training set: {len(train_set)} (augmentation applied on the fly)")
    print(f"Number of images in validation set: {len(validation_set)} (no augmentation)")

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

    train_wrapped = TransformWrapper(train_set, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_wrapped, batch_size=batch_size, sampler=sampler)
    validation_loader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=batch_size,
    )
    return train_loader, validation_loader


def main():
    """Load data, build model, and run training with data augmentation."""
    global model, epoch
    epoch = 0

    pathlib.Path("saves").mkdir(exist_ok=True)

    dataset = Dataset(IMAGES_PATH, device="cpu")
    print(f"Found {len(dataset)} images.")

    train_transform = get_train_transform()
    train_loader, validation_loader = get_dataloaders(
        dataset, 0.8, BATCH_SIZE, len(dataset.class_names), train_transform=train_transform
    )

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

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

    output_file = open("saves/printout_second_" + TIME_STAMP + ".txt", "a")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
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

        model.eval()
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

    plot_path = "saves/training_history_second_cnn.png"
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

    # Save final model (weights only, for inference)
    final_model_path = "saves/second_cnn_model.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model saved: {final_model_path}")

    print("Training finished.")


if __name__ == "__main__":
    main()
