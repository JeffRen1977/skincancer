"""
Inference script for skin cancer classification.
Loads a trained model and classifies images (preferably from outside the training dataset).

This script is designed for:
- Single image classification
- Testing on new images not seen during training
- Getting prediction probabilities for all classes
- Command-line usage for easy integration

Usage:
    python inference.py --model model.pth --image image.jpg
"""
# Core PyTorch libraries
import torch
import torch.nn as nn  # Neural network modules

# Torchvision for image transforms and pre-trained models
from torchvision import transforms  # Image preprocessing transformations
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights  # Model architecture

# Image processing
from PIL import Image  # Load and manipulate images

# Command-line argument parsing
import argparse  # Parse command-line arguments

# File system operations
from pathlib import Path  # Cross-platform path handling


class SkinCancerModel(nn.Module):
    """
    Class-based model for skin cancer classification.
    Uses EfficientNet-B0 as the base model.
    
    This model architecture must match the one used during training.
    It uses EfficientNet-B0 as a feature extractor with a custom classifier
    for 7 skin cancer classes.
    
    Architecture:
    - Backbone: Pre-trained EfficientNet-B0 (frozen during training)
    - Classifier: Custom 2-layer fully connected network
    """
    def __init__(self, num_classes=7, freeze_backbone=True):
        super(SkinCancerModel, self).__init__()
        
        # ========================================================================
        # LOAD PRE-TRAINED EFFICIENTNET-B0 BACKBONE
        # ========================================================================
        # EfficientNet-B0 is a lightweight, efficient CNN pre-trained on ImageNet
        # We use it as a feature extractor (transfer learning)
        weights = EfficientNet_B0_Weights.DEFAULT
        self.backbone = efficientnet_b0(weights=weights)
        
        # ========================================================================
        # FREEZE BACKBONE PARAMETERS
        # ========================================================================
        # During inference, we don't need gradients, but keeping this for consistency
        # with training architecture. Freezing prevents accidental weight updates.
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # ========================================================================
        # REPLACE ORIGINAL CLASSIFIER
        # ========================================================================
        # EfficientNet-B0's original classifier outputs 1000 classes (ImageNet)
        # Replace with Identity() to extract features only (1280-dimensional vectors)
        self.backbone.classifier = nn.Identity()
        
        # ========================================================================
        # CUSTOM CLASSIFIER FOR SKIN CANCER CLASSIFICATION
        # ========================================================================
        # Build a new classifier head for our specific task (7 classes)
        # Architecture: 1280 (features) -> 512 (hidden) -> 7 (output)
        # Note: No softmax here - we apply it during inference for interpretability
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),  # Dropout for regularization (disabled in eval mode)
            nn.Linear(1280, 512),  # First fully connected layer
            nn.ReLU(),  # ReLU activation function
            nn.Dropout(0.2),  # Another dropout layer
            nn.Linear(512, num_classes)  # Output layer: 512 -> num_classes (7)
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
        
        # Pass through custom classifier to get class predictions (logits)
        output = self.classifier(features)  # Output: (batch_size, num_classes)
        
        return output


def load_model(model_path, num_classes=7, device='cpu'):
    """
    Load a trained model from a checkpoint file.
    
    This function handles two checkpoint formats:
    1. Full checkpoint: Dictionary with 'model_state_dict' key (from training)
    2. State dict only: Just the model weights (saved separately)
    
    Args:
        model_path (str): Path to the model checkpoint file (.pth)
        num_classes (int): Number of output classes (must match training, default: 7)
        device (str or torch.device): Device to load model onto ('cpu', 'cuda', 'mps')
    
    Returns:
        model: Loaded model in evaluation mode, or None if loading failed
    """
    # Create model instance with matching architecture
    model = SkinCancerModel(num_classes=num_classes).to(device)
    
    # ========================================================================
    # LOAD CHECKPOINT FILE
    # ========================================================================
    try:
        # Load checkpoint file
        # map_location ensures model loads on correct device (CPU/GPU)
        state_dict = torch.load(model_path, map_location=device)
        
        # Check if it's a full checkpoint (from training) or just state dict
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            # Full checkpoint format: contains model, optimizer, epoch, etc.
            model.load_state_dict(state_dict['model_state_dict'])
            epoch = state_dict.get('epoch', 'unknown')
            print(f"Loaded model from checkpoint (epoch {epoch})")
        else:
            # State dict only format: just model weights
            model.load_state_dict(state_dict)
            print("Loaded model state dict")
    except Exception as e:
        # Handle loading errors gracefully
        print(f"Error loading model: {e}")
        return None
    
    # ========================================================================
    # SET MODEL TO EVALUATION MODE
    # ========================================================================
    # This is crucial for inference:
    # - Disables dropout layers (use deterministic outputs)
    # - Freezes batch normalization statistics
    # - Ensures consistent predictions
    model.eval()
    
    return model


def preprocess_image(image_path):
    """
    Load and preprocess an image for model inference.
    
    This function:
    1. Loads image from file path
    2. Converts to RGB format (handles grayscale, RGBA, etc.)
    3. Applies the same transforms used during training
    4. Returns tensor ready for model input
    
    The preprocessing must match training transforms exactly for accurate predictions.
    
    Args:
        image_path (str or Path): Path to the image file
    
    Returns:
        image_tensor: Preprocessed tensor ready for model input
                     Shape: (1, 3, 224, 224) - batch_size=1, RGB, 224x224
                     Returns None if loading fails
    """
    # ========================================================================
    # DEFINE PREPROCESSING PIPELINE
    # ========================================================================
    # These transforms must match exactly what was used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to model input size (EfficientNet-B0)
        transforms.ToTensor(),  # Convert PIL Image to tensor (0-1 range)
        # Normalize using ImageNet statistics (required for pre-trained models)
        # These values are the mean and std of the ImageNet dataset
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        # Load image and convert to RGB
        # This handles various formats: grayscale, RGBA, etc.
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms to get tensor
        image_tensor = transform(image)  # Shape: (3, 224, 224)
        
        # Add batch dimension: (3, 224, 224) -> (1, 3, 224, 224)
        # Models expect batch dimension even for single images
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    except Exception as e:
        # Handle errors gracefully (corrupted files, wrong format, etc.)
        print(f"Error loading image {image_path}: {e}")
        return None


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
    
    # ========================================================================
    # RUN INFERENCE
    # ========================================================================
    # Disable gradient computation for inference (faster, less memory)
    with torch.no_grad():
        # Forward pass: get raw logits (unnormalized scores)
        logits = model(image_tensor)  # Shape: (1, num_classes)
        
        # ====================================================================
        # CONVERT LOGITS TO PROBABILITIES
        # ====================================================================
        # Apply softmax to convert logits to probabilities
        # logits[0] removes batch dimension: (1, num_classes) -> (num_classes,)
        # dim=0 means apply softmax across the class dimension
        probabilities = F.softmax(logits[0], dim=0).cpu().numpy()
        
        # ====================================================================
        # EXTRACT PREDICTION
        # ====================================================================
        # Get predicted class (index with highest probability)
        predicted_class_idx = logits[0].argmax().item()
        
        # Get confidence (probability of predicted class)
        confidence = probabilities[predicted_class_idx]
    
    return predicted_class_idx, confidence, probabilities


def main():
    """
    Main function to run inference on a single image.
    
    This function:
    1. Parses command-line arguments
    2. Sets up device (CPU/GPU)
    3. Loads class names
    4. Loads the trained model
    5. Preprocesses the input image
    6. Runs prediction
    7. Displays results
    """
    # ========================================================================
    # COMMAND-LINE ARGUMENT PARSING
    # ========================================================================
    parser = argparse.ArgumentParser(description='Skin Cancer Classification Inference')
    
    # Required argument: path to trained model
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model file (.pth)')
    
    # Required argument: path to image to classify
    parser.add_argument('--image', type=str, required=True,
                        help='Path to image file to classify')
    
    # Optional argument: class names (defaults to standard 7 classes)
    parser.add_argument('--classes', type=str, default=None,
                        help='Path to classes file (JSON) or comma-separated class names')
    
    # Optional argument: specify device manually
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cpu, cuda, mps, or auto)')
    
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
    # LOAD CLASS NAMES
    # ========================================================================
    # Class names can be provided in three ways:
    # 1. JSON file with 'classes' key
    # 2. Comma-separated string
    # 3. Default list (if not provided)
    if args.classes:
        if args.classes.endswith('.json'):
            # Load from JSON file
            import json
            with open(args.classes, 'r') as f:
                class_data = json.load(f)
                class_names = class_data.get('classes', [])
        else:
            # Parse comma-separated string
            class_names = [name.strip() for name in args.classes.split(',')]
    else:
        # Default class names (based on the HAM10000 dataset)
        # Order must match the order used during training!
        class_names = [
            'actinic keratoses',
            'basal cell carcinoma',
            'benign keratosis-like lesions',
            'dermatofibroma',
            'melanocytic nevi',
            'melanoma',
            'vascular lesions'
        ]
    
    print(f"Number of classes: {len(class_names)}")
    
    # ========================================================================
    # LOAD MODEL
    # ========================================================================
    print(f"\nLoading model from: {args.model}")
    model = load_model(args.model, num_classes=len(class_names), device=device)
    
    # Check if model loaded successfully
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # ========================================================================
    # PREPROCESS IMAGE
    # ========================================================================
    print(f"\nLoading image: {args.image}")
    image_tensor = preprocess_image(args.image)
    
    # Check if image loaded successfully
    if image_tensor is None:
        print("Failed to load image. Exiting.")
        return
    
    # ========================================================================
    # RUN PREDICTION
    # ========================================================================
    print("\nClassifying image...")
    predicted_idx, confidence, probabilities = predict_image(
        model, image_tensor, class_names, device
    )
    
    # ========================================================================
    # DISPLAY RESULTS
    # ========================================================================
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    
    # Print predicted class and confidence
    print(f"Predicted Class: {class_names[predicted_idx]}")
    print(f"Confidence: {confidence*100:.2f}%")
    
    # Print probabilities for all classes
    print("\nAll Class Probabilities:")
    for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
        # Mark the predicted class
        marker = " <-- PREDICTED" if i == predicted_idx else ""
        print(f"  {class_name}: {prob*100:.2f}%{marker}")
    
    print("="*60)


if __name__ == "__main__":
    main()

