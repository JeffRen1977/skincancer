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
# Core PyTorch libraries for deep learning
import torch
import torch.nn as nn  # Neural network modules (layers, loss functions, etc.)

# Torchvision for datasets, transforms, and pre-trained models
import torchvision
from torchvision import transforms  # Image preprocessing transformations
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights  # EfficientNet-B0 model

# Standard library imports
import os  # File system operations
import json  # JSON file handling (for saving class names, etc.)
from pathlib import Path  # Path manipulation (cross-platform)
import matplotlib.pyplot as plt  # Plotting and visualization
from collections import defaultdict  # Dictionary with default values

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
# Automatically detect and use the best available device for computation
# Priority: MPS (Apple Silicon GPU) > CUDA (NVIDIA GPU) > CPU
if torch.backends.mps.is_available():
    # MPS (Metal Performance Shaders) for Apple Silicon Macs (M1/M2/M3)
    device = torch.device("mps")
elif torch.cuda.is_available():
    # CUDA for NVIDIA GPUs
    device = torch.device("cuda")
else:
    # Fallback to CPU if no GPU is available
    device = torch.device("cpu")

print(f"Using device: {device}")

# ============================================================================
# REPRODUCIBILITY SETUP
# ============================================================================
# Set random seed to ensure reproducible results across runs
# This affects random number generation for data shuffling, weight initialization, etc.
torch.manual_seed(37)
if device.type == "cuda":
    # Also set seed for CUDA operations if using GPU
    torch.cuda.manual_seed(37)


class SkinCancerModel(nn.Module):
    """
    Class-based model for skin cancer classification.
    Uses EfficientNet-B0 as the base model (different from ResNet).
    
    This is a transfer learning approach:
    1. Uses pre-trained EfficientNet-B0 (trained on ImageNet) as feature extractor
    2. Replaces the final classifier with a custom one for our 7 skin cancer classes
    3. Optionally freezes the backbone to only train the new classifier (faster training)
    
    Args:
        num_classes (int): Number of output classes (default: 7 for skin cancer types)
        freeze_backbone (bool): If True, freeze pre-trained weights (transfer learning)
    """
    def __init__(self, num_classes=7, freeze_backbone=True):
        super(SkinCancerModel, self).__init__()
        
        # ========================================================================
        # LOAD PRE-TRAINED EFFICIENTNET-B0 BACKBONE
        # ========================================================================
        # EfficientNet-B0 is a lightweight, efficient CNN architecture
        # Pre-trained on ImageNet (1.4M images, 1000 classes)
        # This provides good feature extraction capabilities
        weights = EfficientNet_B0_Weights.DEFAULT  # Use default ImageNet pre-trained weights
        self.backbone = efficientnet_b0(weights=weights)
        
        # ========================================================================
        # FREEZE BACKBONE PARAMETERS (TRANSFER LEARNING)
        # ========================================================================
        # Freezing means we don't update these weights during training
        # This is faster and requires less memory, and prevents overfitting
        # The pre-trained features are already good for general image recognition
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False  # Disable gradient computation for these parameters
        
        # ========================================================================
        # REPLACE ORIGINAL CLASSIFIER WITH IDENTITY
        # ========================================================================
        # EfficientNet-B0's original classifier outputs 1000 classes (ImageNet)
        # We replace it with Identity() to extract features only
        # EfficientNet-B0 outputs 1280-dimensional feature vectors
        self.backbone.classifier = nn.Identity()
        
        # ========================================================================
        # CUSTOM CLASSIFIER FOR SKIN CANCER CLASSIFICATION
        # ========================================================================
        # Build a new classifier head for our specific task (7 classes)
        # Architecture: 1280 -> 512 -> 7 (with dropout for regularization)
        # Note: No softmax here - CrossEntropyLoss includes it internally
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),  # Dropout for regularization (prevents overfitting)
            nn.Linear(1280, 512),  # First fully connected layer: 1280 features -> 512 hidden units
            nn.ReLU(),  # ReLU activation function (introduces non-linearity)
            nn.Dropout(0.2),  # Another dropout layer
            nn.Linear(512, num_classes)  # Output layer: 512 -> num_classes (7 for skin cancer)
        )
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224) - RGB images
            
        Returns:
            output: Logits tensor of shape (batch_size, num_classes)
                   (raw scores before softmax)
        """
        # Extract features using pre-trained EfficientNet-B0 backbone
        features = self.backbone(x)  # Output: (batch_size, 1280)
        
        # Pass through custom classifier to get class predictions
        output = self.classifier(features)  # Output: (batch_size, num_classes)
        
        return output


def get_dataloaders(data_dir, batch_size=32, train_proportion=0.8):
    """
    Set up dataloaders for train and validation sets.
    
    This function:
    1. Loads images from organized class folders
    2. Applies preprocessing transformations
    3. Splits data into train/validation sets
    4. Creates DataLoaders for efficient batch processing
    
    Args:
        data_dir (str): Path to organized data directory (should contain class subfolders)
        batch_size (int): Number of samples per batch (default: 32)
        train_proportion (float): Proportion of data for training (default: 0.8 = 80%)
    
    Returns:
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        classes: List of class names (automatically inferred from folder names)
    """
    
    # ========================================================================
    # IMAGE PREPROCESSING TRANSFORMATIONS
    # ========================================================================
    # These transforms prepare images for EfficientNet-B0
    # Must match the normalization used during ImageNet pre-training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize all images to 224x224 (EfficientNet input size)
        transforms.ToTensor(),  # Convert PIL Image to PyTorch tensor (0-1 range)
        # Normalize using ImageNet statistics (required for pre-trained models)
        # These values are the mean and std of ImageNet dataset
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # ========================================================================
    # LOAD DATASET USING IMAGEFOLDER
    # ========================================================================
    # ImageFolder automatically:
    # - Finds all images in subdirectories
    # - Assigns class labels based on folder names
    # - Applies transforms to each image
    # Expected structure:
    #   data_dir/
    #     class1/
    #       image1.jpg
    #       image2.jpg
    #     class2/
    #       image3.jpg
    #     ...
    full_dataset = torchvision.datasets.ImageFolder(
        root=data_dir,
        transform=transform
    )
    
    # ========================================================================
    # SPLIT DATASET INTO TRAIN AND VALIDATION SETS
    # ========================================================================
    # Use fixed random seed for reproducible splits
    generator = torch.Generator().manual_seed(37)
    
    # Calculate split sizes
    train_size = int(train_proportion * len(full_dataset))  # 80% for training
    val_size = len(full_dataset) - train_size  # 20% for validation
    
    # Randomly split dataset while maintaining class distribution
    train_set, val_set = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator  # Fixed seed ensures same split every time
    )
    
    # ========================================================================
    # CREATE DATALOADERS FOR BATCH PROCESSING
    # ========================================================================
    # DataLoaders handle:
    # - Batching (combining multiple samples)
    # - Shuffling (for training)
    # - Parallel loading (if num_workers > 0)
    
    # Training DataLoader: shuffle=True for better learning
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,  # Process 32 images at a time
        shuffle=True,  # Randomize order each epoch
        num_workers=0  # Set to 0 for Mac MPS compatibility (MPS doesn't support multiprocessing)
    )
    
    # Validation DataLoader: shuffle=False (deterministic order)
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation data
        num_workers=0
    )
    
    return train_loader, val_loader, full_dataset.classes


def train_epoch(model, train_loader, optimizer, loss_fn, device):
    """
    Train the model for one complete epoch (one pass through all training data).
    
    This function:
    1. Sets model to training mode (enables dropout, batch norm updates)
    2. Iterates through all batches in training data
    3. Performs forward pass, computes loss, backward pass, and updates weights
    4. Tracks loss and accuracy metrics
    
    Args:
        model: The neural network model to train
        train_loader: DataLoader containing training batches
        optimizer: Optimizer for updating model weights (Adam, SGD, etc.)
        loss_fn: Loss function (CrossEntropyLoss)
        device: Device to run computation on (CPU, CUDA, or MPS)
    
    Returns:
        epoch_loss (float): Average loss over all batches in this epoch
        epoch_acc (float): Accuracy percentage for this epoch
    """
    # Set model to training mode
    # This enables dropout layers and updates batch normalization statistics
    model.train()
    
    # Initialize metrics for this epoch
    running_loss = 0.0  # Accumulate loss across batches
    correct = 0  # Count of correct predictions
    total = 0  # Total number of samples processed
    
    # ========================================================================
    # ITERATE THROUGH ALL TRAINING BATCHES
    # ========================================================================
    for images, labels in train_loader:
        # Move data to the appropriate device (GPU/CPU)
        # images: (batch_size, 3, 224, 224) tensor
        # labels: (batch_size,) tensor with class indices
        images, labels = images.to(device), labels.to(device)
        
        # ====================================================================
        # FORWARD PASS
        # ====================================================================
        # Zero out gradients from previous iteration
        # PyTorch accumulates gradients, so we need to reset them
        optimizer.zero_grad()
        
        # Pass images through model to get predictions (logits)
        outputs = model(images)  # Shape: (batch_size, num_classes)
        
        # Compute loss between predictions and true labels
        loss = loss_fn(outputs, labels)
        
        # ====================================================================
        # BACKWARD PASS (GRADIENT COMPUTATION)
        # ====================================================================
        # Compute gradients of loss with respect to model parameters
        loss.backward()
        
        # ====================================================================
        # UPDATE WEIGHTS
        # ====================================================================
        # Update model parameters using computed gradients
        optimizer.step()
        
        # ====================================================================
        # TRACK STATISTICS
        # ====================================================================
        # Accumulate loss for averaging later
        running_loss += loss.item()  # .item() converts tensor to Python float
        
        # Get predicted class indices (class with highest logit value)
        # outputs.data contains logits (raw scores), not probabilities
        _, predicted = torch.max(outputs.data, 1)  # Returns (values, indices), we use indices
        
        # Update accuracy counters
        total += labels.size(0)  # Number of samples in this batch
        correct += (predicted == labels).sum().item()  # Count correct predictions
    
    # ========================================================================
    # COMPUTE EPOCH METRICS
    # ========================================================================
    # Average loss: total loss divided by number of batches
    epoch_loss = running_loss / len(train_loader)
    
    # Accuracy: percentage of correct predictions
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, loss_fn, device):
    """
    Validate the model on validation set (evaluate performance without training).
    
    This function:
    1. Sets model to evaluation mode (disables dropout, freezes batch norm)
    2. Iterates through validation batches WITHOUT updating weights
    3. Computes loss and accuracy to monitor model performance
    4. Uses torch.no_grad() to save memory (no gradient computation needed)
    
    Args:
        model: The neural network model to validate
        val_loader: DataLoader containing validation batches
        loss_fn: Loss function (CrossEntropyLoss)
        device: Device to run computation on (CPU, CUDA, or MPS)
    
    Returns:
        epoch_loss (float): Average loss over all validation batches
        epoch_acc (float): Accuracy percentage on validation set
    """
    # Set model to evaluation mode
    # This disables dropout and freezes batch normalization statistics
    # Important: Different behavior from training mode!
    model.eval()
    
    # Initialize metrics
    running_loss = 0.0
    correct = 0
    total = 0
    
    # ========================================================================
    # DISABLE GRADIENT COMPUTATION FOR VALIDATION
    # ========================================================================
    # torch.no_grad() context manager:
    # - Speeds up computation (no gradient tracking)
    # - Reduces memory usage
    # - We don't need gradients for validation (no weight updates)
    with torch.no_grad():
        # Iterate through all validation batches
        for images, labels in val_loader:
            # Move data to device
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass only (no backward pass needed)
            outputs = model(images)  # Get predictions
            loss = loss_fn(outputs, labels)  # Compute loss
            
            # Accumulate statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)  # Get predicted class indices
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Compute average metrics
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def save_checkpoint(model, optimizer, epoch, loss, acc, filepath):
    """
    Save a training checkpoint to disk.
    
    A checkpoint contains everything needed to resume training:
    - Model weights (to restore model state)
    - Optimizer state (to restore optimizer momentum, etc.)
    - Training progress (epoch number, loss, accuracy)
    
    This allows:
    - Resuming training from a specific point
    - Loading the best model for inference
    - Comparing models at different training stages
    
    Args:
        model: The neural network model
        optimizer: The optimizer being used
        epoch (int): Current epoch number
        loss (float): Current loss value
        acc (float): Current accuracy value
        filepath (str or Path): Path where checkpoint will be saved
    """
    # Create dictionary with all checkpoint information
    checkpoint = {
        'epoch': epoch,  # Which epoch this checkpoint is from
        'model_state_dict': model.state_dict(),  # All model parameters (weights, biases)
        'optimizer_state_dict': optimizer.state_dict(),  # Optimizer state (momentum, etc.)
        'loss': loss,  # Loss at this checkpoint
        'accuracy': acc  # Accuracy at this checkpoint
    }
    
    # Save checkpoint to file
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def plot_training_history(history, save_path="training_history.png"):
    """
    Create and save visualization plots of training history.
    
    Generates two side-by-side plots:
    1. Loss over epochs (training vs validation)
    2. Accuracy over epochs (training vs validation)
    
    These plots help identify:
    - Overfitting (large gap between train/val metrics)
    - Convergence (metrics plateauing)
    - Training progress over time
    
    Args:
        history (dict): Dictionary containing training history with keys:
            - 'train_loss': List of training losses per epoch
            - 'val_loss': List of validation losses per epoch
            - 'train_acc': List of training accuracies per epoch
            - 'val_acc': List of validation accuracies per epoch
        save_path (str): Path where the plot image will be saved
    """
    # Create figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Generate epoch numbers (1, 2, 3, ..., num_epochs)
    epochs = range(1, len(history['train_loss']) + 1)
    
    # ========================================================================
    # PLOT 1: LOSS OVER EPOCHS
    # ========================================================================
    # Plot training loss (blue line)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    # Plot validation loss (red line)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    
    # Configure axes labels and title
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()  # Show legend with line labels
    ax1.grid(True)  # Add grid for easier reading
    
    # ========================================================================
    # PLOT 2: ACCURACY OVER EPOCHS
    # ========================================================================
    # Plot training accuracy (blue line)
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    # Plot validation accuracy (red line)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    
    # Configure axes labels and title
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout to prevent label overlap
    plt.tight_layout()
    
    # Save plot as high-resolution PNG
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 300 DPI for publication quality
    print(f"Training history plot saved: {save_path}")
    
    # Close figure to free memory
    plt.close()


def train_with_optimizer(optimizer_name, optimizer_fn, data_dir, num_epochs=10, batch_size=32):
    """
    Train the model using a specific optimizer.
    
    This function orchestrates the complete training process:
    1. Sets up data loaders
    2. Creates and initializes the model
    3. Trains for specified number of epochs
    4. Validates after each epoch
    5. Saves checkpoints and best model
    6. Generates training history plots
    
    Args:
        optimizer_name (str): Name of optimizer (for logging and file naming)
        optimizer_fn (callable): Function that creates optimizer (e.g., lambda params: Adam(...))
        data_dir (str): Path to organized data directory
        num_epochs (int): Number of training epochs (default: 10)
        batch_size (int): Batch size for training (default: 32)
    
    Returns:
        history (dict): Training history with loss and accuracy for each epoch
        model: Trained model
    """
    # Print header for this optimizer's training session
    print(f"\n{'='*60}")
    print(f"Training with {optimizer_name}")
    print(f"{'='*60}")
    
    # ========================================================================
    # SETUP DATA LOADERS
    # ========================================================================
    # Load and split dataset, create DataLoaders for batch processing
    train_loader, val_loader, classes = get_dataloaders(data_dir, batch_size=batch_size)
    print(f"Number of classes: {len(classes)}")
    print(f"Classes: {classes}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # ========================================================================
    # CREATE AND INITIALIZE MODEL
    # ========================================================================
    # Create model with number of classes matching dataset
    # Move model to appropriate device (GPU/CPU)
    model = SkinCancerModel(num_classes=len(classes)).to(device)
    
    # ========================================================================
    # SETUP LOSS FUNCTION AND OPTIMIZER
    # ========================================================================
    # CrossEntropyLoss: Standard loss for multi-class classification
    # Combines LogSoftmax and NLLLoss (includes softmax internally)
    loss_fn = nn.CrossEntropyLoss()
    
    # Create optimizer using the provided function
    # Pass model parameters that require gradients (only classifier if backbone is frozen)
    optimizer = optimizer_fn(model.parameters())
    
    # ========================================================================
    # INITIALIZE TRAINING HISTORY TRACKING
    # ========================================================================
    # Store metrics for each epoch to plot later
    history = {
        'train_loss': [],  # Training loss per epoch
        'train_acc': [],   # Training accuracy per epoch
        'val_loss': [],    # Validation loss per epoch
        'val_acc': []      # Validation accuracy per epoch
    }
    
    # ========================================================================
    # CREATE CHECKPOINT DIRECTORY
    # ========================================================================
    # Create directory for saving checkpoints (e.g., "checkpoints_adam")
    checkpoint_dir = Path(f"checkpoints_{optimizer_name.lower().replace(' ', '_')}")
    checkpoint_dir.mkdir(exist_ok=True)  # Create if doesn't exist, don't error if exists
    
    # Track best validation accuracy for saving best model
    best_val_acc = 0.0
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    # Train for specified number of epochs
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # ====================================================================
        # TRAIN FOR ONE EPOCH
        # ====================================================================
        # Process all training batches, update weights
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, device)
        
        # Store training metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # ====================================================================
        # VALIDATE AFTER TRAINING
        # ====================================================================
        # Evaluate on validation set (no weight updates)
        val_loss, val_acc = validate(model, val_loader, loss_fn, device)
        
        # Store validation metrics
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # ====================================================================
        # SAVE CHECKPOINT
        # ====================================================================
        # Save checkpoint after every epoch (allows resuming training)
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
        save_checkpoint(model, optimizer, epoch, val_loss, val_acc, checkpoint_path)
        
        # ====================================================================
        # SAVE BEST MODEL
        # ====================================================================
        # Save model with best validation accuracy (for inference)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = checkpoint_dir / "best_model.pth"
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, best_model_path)
            print(f"  → New best model! (Val Acc: {val_acc:.2f}%)")
    
    # ========================================================================
    # SAVE FINAL MODEL
    # ========================================================================
    # Save final model state dict (weights only, for inference)
    # This is lighter than full checkpoint (no optimizer state)
    final_model_path = f"skincancer_model_{optimizer_name.lower().replace(' ', '_')}.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model saved: {final_model_path}")
    
    # ========================================================================
    # GENERATE TRAINING HISTORY PLOTS
    # ========================================================================
    # Create visualization of training progress
    plot_path = f"training_history_{optimizer_name.lower().replace(' ', '_')}.png"
    plot_training_history(history, plot_path)
    
    return history, model


def main():
    """
    Main training function with optimizer comparison.
    
    This function:
    1. Sets up training parameters
    2. Defines 3 different optimizers for comparison
    3. Trains a model with each optimizer
    4. Compares final results
    
    The comparison helps identify which optimizer works best for this task.
    """
    
    # ========================================================================
    # DATA DIRECTORY SETUP
    # ========================================================================
    # Path to organized dataset (should contain class subfolders)
    data_dir = "skincancer/organized"
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} not found. Please run organize_data.py first.")
        return
    
    # ========================================================================
    # TRAINING HYPERPARAMETERS
    # ========================================================================
    num_epochs = 10  # Number of complete passes through training data
    batch_size = 32  # Number of samples processed together
    learning_rate = 0.0001  # Step size for weight updates (0.0001 is conservative for transfer learning)
    
    # ========================================================================
    # DEFINE OPTIMIZERS FOR COMPARISON
    # ========================================================================
    # Three different optimizers to compare:
    # 1. Adam: Adaptive learning rate, good default choice
    # 2. SGD: Stochastic Gradient Descent with momentum, classic optimizer
    # 3. AdamW: Adam with decoupled weight decay, often better generalization
    
    # Dictionary mapping optimizer names to their creation functions
    # Each function takes model parameters and returns an optimizer instance
    optimizers = {
        'Adam': lambda params: torch.optim.Adam(
            params, 
            lr=learning_rate  # Adaptive learning rate per parameter
        ),
        'SGD': lambda params: torch.optim.SGD(
            params, 
            lr=learning_rate, 
            momentum=0.9  # Momentum helps accelerate convergence
        ),
        'AdamW': lambda params: torch.optim.AdamW(
            params, 
            lr=learning_rate, 
            weight_decay=0.01  # L2 regularization (weight decay)
        )
    }
    
    # ========================================================================
    # TRAIN WITH EACH OPTIMIZER
    # ========================================================================
    # Store results from all optimizers for comparison
    all_results = {}
    
    # Train a separate model with each optimizer
    # This allows fair comparison (same data, same architecture, different optimizers)
    for opt_name, opt_fn in optimizers.items():
        # Train model with this optimizer
        history, model = train_with_optimizer(
            opt_name, opt_fn, data_dir, num_epochs, batch_size
        )
        # Store training history for later comparison
        all_results[opt_name] = history
    
    # ========================================================================
    # PRINT COMPARISON SUMMARY
    # ========================================================================
    # Display final metrics for each optimizer side-by-side
    print("\n" + "="*60)
    print("OPTIMIZER COMPARISON SUMMARY")
    print("="*60)
    
    # Extract and display final metrics for each optimizer
    for opt_name, history in all_results.items():
        # Get metrics from the last epoch
        final_train_acc = history['train_acc'][-1]
        final_val_acc = history['val_acc'][-1]
        final_train_loss = history['train_loss'][-1]
        final_val_loss = history['val_loss'][-1]
        
        # Print summary for this optimizer
        print(f"\n{opt_name}:")
        print(f"  Final Train Acc: {final_train_acc:.2f}%, Val Acc: {final_val_acc:.2f}%")
        print(f"  Final Train Loss: {final_train_loss:.4f}, Val Loss: {final_val_loss:.4f}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

