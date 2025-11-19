"""
PyTorch Transfer Learning Training Script for Skin Cancer Classification
Meets project requirements:
- PyTorch framework
- Different model (EfficientNet instead of ResNet)
- Class-based model architecture
- Model saving and checkpoints
- Visualization of loss and accuracy
- Comparison study with 3 optimizers
"""
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

# Set device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Set random seed for reproducibility
torch.manual_seed(37)
if device.type == "cuda":
    torch.cuda.manual_seed(37)


class SkinCancerModel(nn.Module):
    """
    Class-based model for skin cancer classification.
    Uses EfficientNet-B0 as the base model (different from ResNet).
    """
    def __init__(self, num_classes=7, freeze_backbone=True):
        super(SkinCancerModel, self).__init__()
        
        # Load pre-trained EfficientNet-B0
        weights = EfficientNet_B0_Weights.DEFAULT
        self.backbone = efficientnet_b0(weights=weights)
        
        # Freeze backbone parameters if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace classifier with our custom classifier
        # EfficientNet-B0 has 1280 features in the last layer
        self.backbone.classifier = nn.Identity()
        
        # Custom classifier (no softmax - CrossEntropyLoss includes it)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


def get_dataloaders(data_dir, batch_size=32, train_proportion=0.8):
    """Set up dataloaders for train and validation sets."""
    
    # Use EfficientNet's default transforms
    weights = EfficientNet_B0_Weights.DEFAULT
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    full_dataset = torchvision.datasets.ImageFolder(
        root=data_dir,
        transform=transform
    )
    
    # Split into train and validation
    generator = torch.Generator().manual_seed(37)
    train_size = int(train_proportion * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_set, val_set = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for Mac MPS compatibility
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, full_dataset.classes


def train_epoch(model, train_loader, optimizer, loss_fn, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)  # outputs are logits
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, loss_fn, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)  # outputs are logits
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def save_checkpoint(model, optimizer, epoch, loss, acc, filepath):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': acc
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def plot_training_history(history, save_path="training_history.png"):
    """Plot training and validation loss and accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved: {save_path}")
    plt.close()


def train_with_optimizer(optimizer_name, optimizer_fn, data_dir, num_epochs=10, batch_size=32):
    """Train model with a specific optimizer."""
    print(f"\n{'='*60}")
    print(f"Training with {optimizer_name}")
    print(f"{'='*60}")
    
    # Get dataloaders
    train_loader, val_loader, classes = get_dataloaders(data_dir, batch_size=batch_size)
    print(f"Number of classes: {len(classes)}")
    print(f"Classes: {classes}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    model = SkinCancerModel(num_classes=len(classes)).to(device)
    
    # Setup optimizer and loss
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optimizer_fn(model.parameters())
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Create checkpoints directory
    checkpoint_dir = Path(f"checkpoints_{optimizer_name.lower().replace(' ', '_')}")
    checkpoint_dir.mkdir(exist_ok=True)
    
    best_val_acc = 0.0
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, loss_fn, device)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save checkpoint every epoch
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
        save_checkpoint(model, optimizer, epoch, val_loss, val_acc, checkpoint_path)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = checkpoint_dir / "best_model.pth"
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, best_model_path)
    
    # Save final model
    final_model_path = f"skincancer_model_{optimizer_name.lower().replace(' ', '_')}.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model saved: {final_model_path}")
    
    # Plot training history
    plot_path = f"training_history_{optimizer_name.lower().replace(' ', '_')}.png"
    plot_training_history(history, plot_path)
    
    return history, model


def main():
    """Main training function with optimizer comparison."""
    
    # Data directory (organized images)
    data_dir = "skincancer/organized"
    
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} not found. Please run organize_data.py first.")
        return
    
    # Training parameters
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.0001
    
    # Define 3 different optimizers for comparison
    optimizers = {
        'Adam': lambda params: torch.optim.Adam(params, lr=learning_rate),
        'SGD': lambda params: torch.optim.SGD(params, lr=learning_rate, momentum=0.9),
        'AdamW': lambda params: torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.01)
    }
    
    # Train with each optimizer
    all_results = {}
    for opt_name, opt_fn in optimizers.items():
        history, model = train_with_optimizer(
            opt_name, opt_fn, data_dir, num_epochs, batch_size
        )
        all_results[opt_name] = history
    
    # Print comparison summary
    print("\n" + "="*60)
    print("OPTIMIZER COMPARISON SUMMARY")
    print("="*60)
    for opt_name, history in all_results.items():
        final_train_acc = history['train_acc'][-1]
        final_val_acc = history['val_acc'][-1]
        final_train_loss = history['train_loss'][-1]
        final_val_loss = history['val_loss'][-1]
        print(f"\n{opt_name}:")
        print(f"  Final Train Acc: {final_train_acc:.2f}%, Val Acc: {final_val_acc:.2f}%")
        print(f"  Final Train Loss: {final_train_loss:.4f}, Val Loss: {final_val_loss:.4f}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

