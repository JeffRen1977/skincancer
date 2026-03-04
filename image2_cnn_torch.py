import torch
import torch.nn.functional as F
import pathlib
import cv2
import time
import atexit
import torchvision.transforms.v2
import matplotlib.pyplot as plt

INPUT_SHAPE = (3, 160, 160)
IMAGES_PATH = pathlib.Path("skincancer/organized")  # One subfolder per class (from organize_data.py)
BATCH_SIZE = 32
NUM_EPOCHS = 50               # More epochs for imbalanced 7-class data
LEARNING_RATE = 0.001
TIME_STAMP = time.strftime("%Y_%m_%de_%H_%M")

@atexit.register
def clean_up() -> None:
    torch.save(model, "saves/model_" + str(epoch) + "_" + TIME_STAMP)

class Dataset(torch.utils.data.Dataset):
    """Represent the dataset as an object. Loads to CPU for compatibility."""
    def __init__(self, image_path: pathlib.Path, device: str = "cpu"):
        """Provide a path where all the images are, 1 folder for each class."""
        print("Loading images...")
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
                        img = torch.tensor(img, dtype=torch.float32, device=device)
                        self.images.append((img, label))
        print("Images loaded.")
    def __len__(self) -> int:
        return len(self.images)
        
    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        return self.images[idx]
                                        
class Model(torch.nn.Module):
    """Represent my CNN as an object."""
    def __init__(self, input_shape: (int, int, int)):
        """input_shape is expected to be channels first."""
        super().__init__()
        print(f"{'Input Shape:':>30}", input_shape)

        #zero-padding
        self.zp1 = torch.nn.ZeroPad2d((1, 1, 1, 1))

        #first convolution
        self.conv1 = torch.nn.Conv2d(
            in_channels = 3,
            out_channels = 24,
            kernel_size = 9,
            stride = 3,
        )

        #relu activation
        self.relu = torch.nn.ReLU()

        #first MaxPool
        self.maxpool1 = torch.nn.MaxPool2d(
            kernel_size = 3,
            stride = 2,
        )

        #second convolution
        self.conv2 = torch.nn.Conv2d(
            in_channels = 24,
            out_channels = 48,
            kernel_size = 5,
            stride = 2,
        )

        #second maxpool
        self.maxpool2 = torch.nn.MaxPool2d(
            kernel_size = 3,
            stride = 2,
        )

        #third convolution
        self.conv3 = torch.nn.Conv2d(
            in_channels = 48,
            out_channels = 96,
            kernel_size = 3,
            stride = 1,
        )

        #flatten layer
        self.flatten = torch.nn.Flatten()

        #linear layer also called fully-connected or dense layer
        #this section is also called MLP = Multi-Layer Perception
        #first linear
        self.linear1 = torch.nn.Linear(
            in_features = 864,
            out_features = 256,
        )

        #second linear
        self.linear2 = torch.nn.Linear(
            in_features = 256,
            out_features = 64,
        )

        #third linear
        self.linear3 = torch.nn.Linear(
            in_features = 64,
            out_features = 7,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass of the CNN."""
        y = self.zp1(x)
        y = self.relu(self.conv1(y))
        y = self.maxpool1(y)
        y = self.relu(self.conv2(y))
        y = self.maxpool2(y)
        y = self.relu(self.conv3(y))
        y = self.flatten(y)
        y = self.linear1(y)
        y = self.linear2(y)
        y = self.linear3(y)
        return y

def get_dataloaders(dataset: Dataset, train_prop: float, batch_size: int,
        n_classes: int) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
    """Split the dataset and prepare for training. Uses WeightedRandomSampler for imbalance."""
    generator = torch.Generator().manual_seed(37)
    train_set, validation_set = torch.utils.data.random_split(
            dataset,
            lengths = [train_prop, 1-train_prop],
            generator = generator,
    )
    print(f"Number of images in training set before augmentation: ", end = "")
    print(f"{len(train_set)}")
    aug_set = torchvision.transforms.v2.RandomHorizontalFlip(1.0)(train_set)
    train_set = torch.utils.data.ConcatDataset([train_set, aug_set])
    print(f"Number of images in training set after augmentation: ", end = "")
    print(f"{len(train_set)}")

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
        weights=sample_weights,
        num_samples=4 * len(sample_weights),  # 4x oversampling
        replacement=True
    )
    print("Using WeightedRandomSampler to oversample minority classes (4x)")
    train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=sampler)
    validation = torch.utils.data.DataLoader(validation_set, batch_size=batch_size)
    return train, validation

def save_code() -> None:
    with open(__file__, 'r') as f:
        this_code = f.read()
    with open("saves/code_" + TIME_STAMP + ".py", "w") as f:
        print(this_code, file = f)

def main():
    """
    Entry point: load dataset, build CNN on MPS, train for NUM_EPOCHS, and save model.
    Model and epoch are globals so atexit clean_up() can save the model when the script exits.
    """
    global model, epoch

    pathlib.Path("saves").mkdir(exist_ok=True)
    save_code()
    dataset = Dataset(IMAGES_PATH, device="cpu")
    print(f"Found {len(dataset)} images.")

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train, validation = get_dataloaders(dataset, 0.8, BATCH_SIZE, len(dataset.class_names))
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

    class FocalLoss(torch.nn.Module):
        def __init__(self, weight=None, gamma=2.0):
            super().__init__()
            self.weight = weight
            self.gamma = gamma
        def forward(self, logits, targets):
            ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
            pt = torch.exp(-ce)
            return ((1 - pt) ** self.gamma * ce).mean()
    criterion = FocalLoss(weight=class_weights, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    output_file = open("saves/printout_image2_" + TIME_STAMP + ".txt", "a")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for e in range(NUM_EPOCHS):
        epoch = e
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train:
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
        train_loss = running_loss / len(train)
        train_acc = 100.0 * train_correct / train_total if train_total else 0.0

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in validation:
                images, labels = images.to(device), labels.to(device)
                labels = labels.long()
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss /= max(1, len(validation))
        val_acc = 100.0 * correct / total if total else 0.0

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        line = f"Epoch {e + 1}/{NUM_EPOCHS}  train_loss={train_loss:.4f}  train_acc={train_acc:.2f}%  val_loss={val_loss:.4f}  val_acc={val_acc:.2f}%\n"
        print(line.strip())
        output_file.write(line)
        output_file.flush()

    output_file.close()

    # Plot training and validation loss and accuracy (same style as second_cnn_torch.py).
    plot_path = "saves/training_history_image2_cnn.png"
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
    final_model_path = "saves/image2_cnn_model.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model saved: {final_model_path}")

    epoch = NUM_EPOCHS - 1  # last completed epoch for atexit save filename

if __name__ == "__main__":
    main()
