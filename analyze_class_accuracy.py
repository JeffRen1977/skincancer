"""
Class-wise accuracy analysis script.
Tests images from each class one at a time and displays results.
Shows which classes are more accurate or less accurate.

This script helps identify:
- Which classes the model predicts well (high accuracy)
- Which classes need improvement (low accuracy)
- Per-class performance metrics
- Visual comparison of class accuracies
"""
# Core PyTorch libraries
import torch
import torch.nn as nn  # Neural network modules

# Torchvision for image transforms and pre-trained models
from torchvision import transforms  # Image preprocessing
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights  # Model architecture

# Image processing
from PIL import Image  # Load and manipulate images

# Command-line argument parsing
import argparse  # Parse command-line arguments

# File system operations
from pathlib import Path  # Cross-platform path handling

# Visualization
import matplotlib.pyplot as plt  # Plotting and visualization

# Numerical operations
import numpy as np  # Array operations and statistics

# Data structures
from collections import defaultdict  # Dictionary with default values

# Random operations
import random  # Shuffle image lists for random sampling


class SkinCancerModel(nn.Module):
    """
    Class-based model for skin cancer classification.
    
    This model architecture matches the one used during training.
    It uses EfficientNet-B0 as a feature extractor with a custom classifier.
    
    Note: This must match the architecture used in train_skincancer.py
    to ensure proper model loading.
    """
    def __init__(self, num_classes=7, freeze_backbone=True):
        super(SkinCancerModel, self).__init__()
        
        # Load pre-trained EfficientNet-B0 backbone
        weights = EfficientNet_B0_Weights.DEFAULT
        self.backbone = efficientnet_b0(weights=weights)
        
        # Freeze backbone parameters (not needed for inference, but kept for consistency)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace original classifier with identity (extract features only)
        self.backbone.classifier = nn.Identity()
        
        # Custom classifier head (matches training architecture)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 512),  # 1280 features from EfficientNet -> 512 hidden units
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)  # 512 -> num_classes (7 for skin cancer)
        )
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            output: Logits tensor of shape (batch_size, num_classes)
        """
        # Extract features using pre-trained backbone
        features = self.backbone(x)
        # Pass through custom classifier
        output = self.classifier(features)
        return output


# -----------------------------------------------------------------------------
# Custom CNN models (first_cnn, second_cnn, image_cnn, image2_cnn)
# Architectures must match the training scripts exactly for state_dict loading.
# -----------------------------------------------------------------------------

def _get_activation(name):
    """Return activation module for second_cnn (LeakyReLU, GELU, etc.)."""
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "elu":
        return nn.ELU(alpha=1.0)
    raise ValueError(f"Unknown activation: {name}")


class FirstCNNModel(nn.Module):
    """Custom CNN from first_cnn_torch.py: ReLU + dropout, 256x256 input."""
    def __init__(self, input_shape=(3, 256, 256)):
        super().__init__()
        self.zp1 = nn.ZeroPad2d((1, 1, 1, 1))
        self.conv1 = nn.Conv2d(3, 24, kernel_size=13, stride=4)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(24, 48, kernel_size=7, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(48, 96, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(864, 256)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, 7)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
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


class SecondCNNModel(nn.Module):
    """Custom CNN from second_cnn_torch.py: LeakyReLU + dropout, 256x256 input."""
    def __init__(self, input_shape=(3, 256, 256), activation_name="leaky_relu"):
        super().__init__()
        self.zp1 = nn.ZeroPad2d((1, 1, 1, 1))
        self.conv1 = nn.Conv2d(3, 24, kernel_size=13, stride=4)
        self.act = _get_activation(activation_name)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(24, 48, kernel_size=7, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(48, 96, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(864, 256)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, 7)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
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


class ImageCNNModel(nn.Module):
    """Custom CNN from image_cnn_torch.py: ReLU, no dropout, 256x256 input."""
    def __init__(self, input_shape=(3, 256, 256)):
        super().__init__()
        self.zp1 = nn.ZeroPad2d((1, 1, 1, 1))
        self.conv1 = nn.Conv2d(3, 24, kernel_size=13, stride=4)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(24, 48, kernel_size=7, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(48, 96, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(864, 256)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, 7)

    def forward(self, x):
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


class Image2CNNModel(nn.Module):
    """Custom CNN from image2_cnn_torch.py: ReLU, no dropout, 160x160 input."""
    def __init__(self, input_shape=(3, 160, 160)):
        super().__init__()
        self.zp1 = nn.ZeroPad2d((1, 1, 1, 1))
        self.conv1 = nn.Conv2d(3, 24, kernel_size=9, stride=3)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(24, 48, kernel_size=5, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(48, 96, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(864, 256)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, 7)

    def forward(self, x):
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


def load_model(model_path, num_classes=None, device='cpu'):
    """
    Load a trained model from a checkpoint file.
    
    This function handles two checkpoint formats:
    1. Full checkpoint: Dictionary with 'model_state_dict' key (from training)
    2. State dict only: Just the model weights (saved separately)
    
    If num_classes is None, it is auto-detected from the checkpoint (classifier.4.weight shape).
    
    Args:
        model_path (str): Path to the model checkpoint file (.pth)
        num_classes (int or None): Number of output classes. If None, inferred from checkpoint.
        device (str or torch.device): Device to load model onto ('cpu', 'cuda', 'mps')
    
    Returns:
        model: Loaded model in evaluation mode, or None if loading failed
    """
    try:
        state_dict = torch.load(model_path, map_location=device)
        sd = state_dict['model_state_dict'] if isinstance(state_dict, dict) and 'model_state_dict' in state_dict else state_dict
        
        # Auto-detect num_classes from checkpoint if not provided
        if num_classes is None and 'classifier.4.weight' in sd:
            num_classes = int(sd['classifier.4.weight'].shape[0])
            print(f"Auto-detected num_classes={num_classes} from checkpoint")
        if num_classes is None:
            num_classes = 7
        if num_classes != 7:
            print(f"Warning: Model has {num_classes} classes but analysis expects 7. "
                  "Ensure data_dir/class mapping matches training.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

    model = SkinCancerModel(num_classes=num_classes).to(device)
    
    try:
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
            epoch = state_dict.get('epoch', 'unknown')
            print(f"Loaded model from checkpoint (epoch {epoch})")
        else:
            model.load_state_dict(state_dict)
            print("Loaded model state dict")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    model.eval()
    return model


def load_custom_cnn_model(model_path, model_type, device='cpu'):
    """
    Load a custom CNN model (first_cnn, second_cnn, image_cnn, image2_cnn).

    Args:
        model_path (str): Path to the .pth file (state_dict only)
        model_type (str): One of 'first_cnn', 'second_cnn', 'image_cnn', 'image2_cnn'
        device: Device to load model onto

    Returns:
        model: Loaded model in evaluation mode, or None if loading failed
    """
    input_shapes = {
        'first_cnn': (3, 256, 256),
        'second_cnn': (3, 256, 256),
        'image_cnn': (3, 256, 256),
        'image2_cnn': (3, 160, 160),
    }
    model_classes = {
        'first_cnn': FirstCNNModel,
        'second_cnn': SecondCNNModel,
        'image_cnn': ImageCNNModel,
        'image2_cnn': Image2CNNModel,
    }
    if model_type not in model_classes:
        print(f"Error: Unknown model_type '{model_type}'. Use: first_cnn, second_cnn, image_cnn, image2_cnn")
        return None

    shape = input_shapes[model_type]
    model = model_classes[model_type](input_shape=shape).to(device)

    try:
        state_dict = torch.load(model_path, map_location=device)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
        print(f"Loaded {model_type} model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    model.eval()
    return model


def preprocess_image(image_path, model_type='efficientnet'):
    """
    Load and preprocess an image for model inference.

    Supports multiple model types with different input sizes and normalization:
    - efficientnet: 224x224, ImageNet normalization
    - first_cnn, second_cnn, image_cnn: 256x256, [0,1] range only
    - image2_cnn: 160x160, [0,1] range only

    Args:
        image_path (str or Path): Path to the image file
        model_type (str): Model type for correct preprocessing

    Returns:
        image_tensor: Preprocessed tensor (1, 3, H, W)
        original_image: PIL Image for visualization
        (None, None) if loading fails
    """
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None

    if model_type == 'efficientnet':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif model_type in ('first_cnn', 'second_cnn', 'image_cnn'):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),  # [0, 1] range, no ImageNet norm
        ])
    elif model_type == 'image2_cnn':
        transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
        ])
    else:
        print(f"Warning: Unknown model_type '{model_type}', using efficientnet preprocessing")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, image


def predict_image(model, image_tensor, class_names, device='cpu'):
    """
    Run inference on a single image and get prediction results.
    
    This function:
    1. Moves image to the correct device (GPU/CPU)
    2. Runs model forward pass
    3. Converts logits to probabilities using softmax
    4. Extracts predicted class and confidence
    
    Args:
        model: Trained model in evaluation mode
        image_tensor: Preprocessed image tensor, shape (1, 3, 224, 224)
        class_names: List of class names (for reference, not used in computation)
        device: Device to run inference on
    
    Returns:
        predicted_class_idx (int): Index of predicted class (0 to num_classes-1)
        confidence (float): Probability of predicted class (0 to 1)
        probabilities (np.array): Probability distribution over all classes
                                  Shape: (num_classes,)
    """
    import torch.nn.functional as F  # For softmax function
    
    # Move image tensor to the same device as model
    image_tensor = image_tensor.to(device)
    
    # Disable gradient computation for inference (faster, less memory)
    with torch.no_grad():
        # Forward pass: get raw logits (unnormalized scores)
        logits = model(image_tensor)  # Shape: (1, num_classes)
        
        # Apply softmax to convert logits to probabilities
        # logits[0] removes batch dimension: (1, num_classes) -> (num_classes,)
        probabilities = F.softmax(logits[0], dim=0).cpu().numpy()
        
        # Get predicted class (index with highest probability)
        predicted_class_idx = logits[0].argmax().item()
        
        # Get confidence (probability of predicted class)
        confidence = probabilities[predicted_class_idx]
    
    return predicted_class_idx, confidence, probabilities


def get_class_mapping():
    """
    Get mapping between directory names and display names for classes.
    
    This function provides two representations:
    1. Directory names: As they appear in file system (with underscores, hyphens)
    2. Display names: Human-readable names for output (with spaces)
    
    The order of classes must match:
    - The order used during training
    - The order in the organized data directory
    - The model's output class indices
    
    Returns:
        dir_names (list): Directory folder names (for finding image files)
        display_names (list): Human-readable class names (for display)
    """
    # Directory names (as they appear in organized folder structure)
    # These match the folder names created by organize_data.py
    dir_names = [
        'actinic_keratoses',
        'basal_cell_carcinoma',
        'benign_keratosis-like_lesions',
        'dermatofibroma',
        'melanocytic_nevi',
        'melanoma',
        'vascular_lesions'
    ]
    
    # Display names (human-readable, for output and visualization)
    # These are used in print statements and plots
    display_names = [
        'actinic keratoses',
        'basal cell carcinoma',
        'benign keratosis-like lesions',
        'dermatofibroma',
        'melanocytic nevi',
        'melanoma',
        'vascular lesions'
    ]
    
    return dir_names, display_names


def analyze_class_accuracy(model, data_dir, class_names, device='cpu',
                          max_images_per_class=50, show_images=True, model_type='efficientnet',
                          output_path='class_accuracy_analysis.png'):
    """
    Analyze accuracy for each class, testing images one at a time.
    
    This is the main analysis function that:
    1. Finds all images for each class
    2. Tests each image individually
    3. Tracks per-class statistics (correct/total, confidences)
    4. Displays results for each image (optional)
    5. Generates summary statistics and visualization
    
    Args:
        model: Trained model in evaluation mode
        data_dir (str): Directory containing organized class folders
        class_names (list): List of class display names
        device (str or torch.device): Device to run inference on
        max_images_per_class (int): Maximum number of images to test per class
                                   (limits analysis time, uses random sampling)
        show_images (bool): If True, display each image with prediction results
                           (slower but more informative)
    
    Returns:
        class_stats (dict): Dictionary with statistics for each class
                          Format: {class_idx: {'correct': int, 'total': int, 'confidences': list}}
    """
    # Get directory and display name mappings
    dir_names, display_names = get_class_mapping()
    
    # ========================================================================
    # INITIALIZE STATISTICS TRACKING
    # ========================================================================
    # Use defaultdict to automatically create entries for each class
    # Each class tracks:
    #   - correct: Number of correct predictions
    #   - total: Total number of images tested
    #   - confidences: List of confidence scores for all predictions
    class_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'confidences': []})
    
    # ========================================================================
    # FIND ALL IMAGES FOR EACH CLASS
    # ========================================================================
    data_path = Path(data_dir)
    all_images = {}  # Dictionary: {class_idx: [list of image paths]}
    
    # Iterate through each class directory
    for i, dir_name in enumerate(dir_names):
        class_dir = data_path / dir_name
        
        if class_dir.exists():
            # Find all image files in this class directory
            # Support both .jpg and .png formats
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            
            # Shuffle for random sampling (if more images than max_images_per_class)
            random.shuffle(images)
            
            # Take up to max_images_per_class images
            all_images[i] = images[:max_images_per_class]
            
            print(f"Found {len(images)} images in {dir_name}, testing {len(all_images[i])}")
        else:
            # Class directory doesn't exist
            print(f"Warning: {dir_name} not found")
            all_images[i] = []
    
    # ========================================================================
    # START ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("STARTING CLASS-WISE ACCURACY ANALYSIS")
    print("="*80)
    print(f"Testing up to {max_images_per_class} images per class\n")
    
    # ========================================================================
    # TEST EACH CLASS
    # ========================================================================
    # Iterate through each class and test all its images
    for class_idx in range(len(class_names)):
        true_class_name = display_names[class_idx]
        images = all_images[class_idx]
        
        # Skip if no images found for this class
        if not images:
            print(f"\nSkipping {true_class_name}: No images found")
            continue
        
        # Print header for this class
        print(f"\n{'='*80}")
        print(f"ANALYZING CLASS: {true_class_name.upper()}")
        print(f"{'='*80}")
        print(f"Testing {len(images)} images...\n")
        
        # ====================================================================
        # TEST EACH IMAGE IN THIS CLASS
        # ====================================================================
        for img_idx, image_path in enumerate(images):
            # Load and preprocess image
            image_tensor, original_image = preprocess_image(image_path, model_type)
            if image_tensor is None:
                # Skip if image loading failed
                continue
            
            # Run model inference to get prediction
            predicted_idx, confidence, probabilities = predict_image(
                model, image_tensor, class_names, device
            )
            
            # ================================================================
            # UPDATE STATISTICS
            # ================================================================
            # Check if prediction is correct
            is_correct = (predicted_idx == class_idx)
            
            # Update counters
            class_stats[class_idx]['total'] += 1
            if is_correct:
                class_stats[class_idx]['correct'] += 1
            
            # Store confidence for this prediction
            class_stats[class_idx]['confidences'].append(confidence)
            
            # ================================================================
            # DISPLAY RESULT
            # ================================================================
            # Determine status symbol
            status = "✓ CORRECT" if is_correct else "✗ WRONG"
            
            # Print prediction result
            print(f"Image {img_idx+1}/{len(images)}: {status}")
            print(f"  True Class: {true_class_name}")
            print(f"  Predicted: {display_names[predicted_idx]}")
            print(f"  Confidence: {confidence*100:.2f}%")
            
            # ================================================================
            # VISUALIZE IMAGE AND PREDICTIONS (OPTIONAL)
            # ================================================================
            if show_images:
                # Create figure with two subplots
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                
                # Left plot: Show original image
                axes[0].imshow(original_image)
                axes[0].set_title(f'Image: {image_path.name}\nTrue Class: {true_class_name}', 
                                 fontsize=12, fontweight='bold')
                axes[0].axis('off')  # Remove axes for cleaner image display
                
                # Right plot: Show prediction probabilities for all classes
                # Color coding:
                #   - Green: Predicted class (what model thinks it is)
                #   - Red: True class (what it actually is)
                #   - Gray: Other classes
                colors = ['green' if i == predicted_idx else 'red' if i == class_idx else 'gray' 
                         for i in range(len(class_names))]
                
                # Create horizontal bar chart
                bars = axes[1].barh(range(len(class_names)), probabilities * 100, color=colors)
                axes[1].set_yticks(range(len(class_names)))
                axes[1].set_yticklabels(display_names)
                axes[1].set_xlabel('Probability (%)', fontsize=11)
                axes[1].set_title(f'Prediction Probabilities\n{status}', 
                                 fontsize=12, fontweight='bold')
                axes[1].set_xlim([0, 100])  # Probability range 0-100%
                axes[1].grid(axis='x', alpha=0.3)  # Add grid for readability
                
                # Add probability values on bars (only if > 1% to avoid clutter)
                for i, (bar, prob) in enumerate(zip(bars, probabilities)):
                    if prob > 0.01:  # Only show if > 1%
                        axes[1].text(prob * 100 + 1, i, f'{prob*100:.1f}%', 
                                    va='center', fontsize=9)
                
                # Display plot
                plt.tight_layout()  # Adjust layout to prevent overlap
                plt.show(block=False)  # Non-blocking display
                plt.pause(0.5)  # Brief pause to allow image to render
                plt.close()  # Close to free memory
            
            # ================================================================
            # PRINT RUNNING ACCURACY
            # ================================================================
            # Calculate current accuracy for this class so far
            current_acc = (class_stats[class_idx]['correct'] / 
                          class_stats[class_idx]['total']) * 100
            print(f"  Current Class Accuracy: {current_acc:.2f}%")
            print()
        
        # ====================================================================
        # PRINT CLASS SUMMARY
        # ====================================================================
        # Calculate final statistics for this class
        final_acc = (class_stats[class_idx]['correct'] / 
                    class_stats[class_idx]['total']) * 100
        avg_confidence = np.mean(class_stats[class_idx]['confidences']) * 100
        
        # Print summary
        print(f"\n{'─'*80}")
        print(f"CLASS SUMMARY: {true_class_name}")
        print(f"{'─'*80}")
        print(f"Total Images Tested: {class_stats[class_idx]['total']}")
        print(f"Correct Predictions: {class_stats[class_idx]['correct']}")
        print(f"Accuracy: {final_acc:.2f}%")
        print(f"Average Confidence: {avg_confidence:.2f}%")
        print()
    
    # ========================================================================
    # OVERALL SUMMARY AND RANKING
    # ========================================================================
    print("\n" + "="*80)
    print("OVERALL CLASS-WISE ACCURACY SUMMARY")
    print("="*80)
    
    # Sort classes by accuracy (highest to lowest)
    # This helps identify which classes perform well and which need improvement
    sorted_classes = sorted(class_stats.items(), 
                           key=lambda x: (x[1]['correct'] / x[1]['total']) if x[1]['total'] > 0 else 0,
                           reverse=True)
    
    # Print ranked list of classes
    print("\nClasses ranked by accuracy (highest to lowest):\n")
    for rank, (class_idx, stats) in enumerate(sorted_classes, 1):
        if stats['total'] > 0:
            # Calculate metrics
            accuracy = (stats['correct'] / stats['total']) * 100
            avg_conf = np.mean(stats['confidences']) * 100
            
            # Print formatted summary
            print(f"{rank}. {display_names[class_idx]:35s} "
                  f"Accuracy: {accuracy:6.2f}%  "
                  f"({stats['correct']}/{stats['total']})  "
                  f"Avg Confidence: {avg_conf:.2f}%")
    
    # ========================================================================
    # GENERATE VISUALIZATION
    # ========================================================================
    print("\nGenerating accuracy visualization...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Prepare data for bar chart
    class_accs = []  # List of accuracy values
    class_labels = []  # List of class names
    colors_list = []  # List of colors (based on accuracy thresholds)
    
    # Extract data from sorted classes
    for class_idx, stats in sorted_classes:
        if stats['total'] > 0:
            accuracy = (stats['correct'] / stats['total']) * 100
            class_accs.append(accuracy)
            class_labels.append(display_names[class_idx])
            
            # Color coding based on accuracy thresholds:
            #   - Green: High accuracy (≥80%) - model performs well
            #   - Orange: Medium accuracy (60-80%) - acceptable but could improve
            #   - Red: Low accuracy (<60%) - model struggles with this class
            if accuracy >= 80:
                colors_list.append('green')
            elif accuracy >= 60:
                colors_list.append('orange')
            else:
                colors_list.append('red')
    
    # Create horizontal bar chart
    bars = ax.barh(range(len(class_accs)), class_accs, color=colors_list)
    ax.set_yticks(range(len(class_accs)))
    ax.set_yticklabels(class_labels)
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 100])  # Accuracy range 0-100%
    ax.grid(axis='x', alpha=0.3)  # Add grid for easier reading
    
    # Add accuracy value labels on each bar
    for i, (bar, acc) in enumerate(zip(bars, class_accs)):
        ax.text(acc + 1, i, f'{acc:.1f}%', va='center', fontsize=10, fontweight='bold')
    
    # Save and display plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    if show_images:
        plt.show()  # Blocks until user closes window
    else:
        plt.close()  # Exit immediately when --no_display
    return class_stats


def main():
    """
    Main function to run class-wise accuracy analysis.
    
    This function:
    1. Parses command-line arguments
    2. Sets up device (CPU/GPU)
    3. Loads the trained model
    4. Runs the accuracy analysis
    """
    # ========================================================================
    # COMMAND-LINE ARGUMENT PARSING
    # ========================================================================
    parser = argparse.ArgumentParser(description='Class-wise Accuracy Analysis')
    
    # Required argument: path to trained model
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model file (.pth)')
    
    # Optional argument: data directory (defaults to organized folder)
    parser.add_argument('--data_dir', type=str, default='skincancer/organized',
                        help='Path to organized data directory')
    
    # Optional argument: limit number of images per class (for faster analysis)
    parser.add_argument('--max_images', type=int, default=50,
                        help='Maximum number of images to test per class')
    
    # Optional flag: skip image display (faster analysis)
    parser.add_argument('--no_display', action='store_true',
                        help='Do not display images during analysis (faster)')
    
    # Optional argument: specify device manually
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cpu, cuda, mps, or auto)')

    # Model type: which architecture to load (required for correct loading and preprocessing)
    parser.add_argument('--model_type', type=str, default='efficientnet',
                        choices=['efficientnet', 'first_cnn', 'second_cnn', 'image_cnn', 'image2_cnn'],
                        help='Model architecture: efficientnet (train_skincancer), first_cnn, second_cnn, image_cnn, image2_cnn')

    # Output path for the accuracy visualization
    parser.add_argument('--output', type=str, default='class_accuracy_analysis.png',
                        help='Path to save the class accuracy plot')

    # Parse arguments
    args = parser.parse_args()
    
    # ========================================================================
    # DEVICE SETUP
    # ========================================================================
    # Automatically detect best available device, or use specified one
    if args.device == 'auto':
        # Priority: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
        if torch.backends.mps.is_available():
            device = torch.device("mps")  # Apple Silicon GPU
        elif torch.cuda.is_available():
            device = torch.device("cuda")  # NVIDIA GPU
        else:
            device = torch.device("cpu")  # Fallback to CPU
    else:
        # Use user-specified device
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # ========================================================================
    # GET CLASS NAMES
    # ========================================================================
    # Get class mapping (we only need display names here)
    _, class_names = get_class_mapping()
    print(f"Number of classes: {len(class_names)}")
    
    # ========================================================================
    # LOAD MODEL
    # ========================================================================
    print(f"\nLoading model from: {args.model} (type: {args.model_type})")
    if args.model_type == 'efficientnet':
        model = load_model(args.model, num_classes=None, device=device)  # Auto-detect from checkpoint
    else:
        model = load_custom_cnn_model(args.model, args.model_type, device=device)

    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # ========================================================================
    # RUN ANALYSIS
    # ========================================================================
    # Run the main analysis function
    analyze_class_accuracy(
        model=model,
        data_dir=args.data_dir,
        class_names=class_names,
        device=device,
        max_images_per_class=args.max_images,
        show_images=not args.no_display,
        model_type=args.model_type,
        output_path=args.output
    )


if __name__ == "__main__":
    main()