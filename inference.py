"""
Inference script for skin cancer classification.
Loads a trained model and classifies images (preferably from outside the training dataset).
"""
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import argparse
from pathlib import Path


class SkinCancerModel(nn.Module):
    """
    Class-based model for skin cancer classification.
    Uses EfficientNet-B0 as the base model.
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
        self.backbone.classifier = nn.Identity()
        
        # Custom classifier (no softmax - we'll apply it in inference)
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


def load_model(model_path, num_classes=7, device='cpu'):
    """Load trained model from checkpoint."""
    model = SkinCancerModel(num_classes=num_classes).to(device)
    
    # Try to load as state dict first
    try:
        state_dict = torch.load(model_path, map_location=device)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
            print(f"Loaded model from checkpoint (epoch {state_dict.get('epoch', 'unknown')})")
        else:
            model.load_state_dict(state_dict)
            print("Loaded model state dict")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    model.eval()
    return model


def preprocess_image(image_path):
    """Preprocess image for inference."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def predict_image(model, image_tensor, class_names, device='cpu'):
    """Predict class for a single image."""
    import torch.nn.functional as F
    
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        logits = model(image_tensor)
        # Apply softmax to get probabilities
        probabilities = F.softmax(logits[0], dim=0).cpu().numpy()
        predicted_class_idx = logits[0].argmax().item()
        confidence = probabilities[predicted_class_idx]
    
    return predicted_class_idx, confidence, probabilities


def main():
    parser = argparse.ArgumentParser(description='Skin Cancer Classification Inference')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model file (.pth)')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to image file to classify')
    parser.add_argument('--classes', type=str, default=None,
                        help='Path to classes file (JSON) or comma-separated class names')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cpu, cuda, mps, or auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load class names
    if args.classes:
        if args.classes.endswith('.json'):
            import json
            with open(args.classes, 'r') as f:
                class_data = json.load(f)
                class_names = class_data.get('classes', [])
        else:
            class_names = [name.strip() for name in args.classes.split(',')]
    else:
        # Default class names (based on the dataset)
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
    
    # Load model
    print(f"\nLoading model from: {args.model}")
    model = load_model(args.model, num_classes=len(class_names), device=device)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Preprocess image
    print(f"\nLoading image: {args.image}")
    image_tensor = preprocess_image(args.image)
    if image_tensor is None:
        print("Failed to load image. Exiting.")
        return
    
    # Make prediction
    print("\nClassifying image...")
    predicted_idx, confidence, probabilities = predict_image(
        model, image_tensor, class_names, device
    )
    
    # Print results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Predicted Class: {class_names[predicted_idx]}")
    print(f"Confidence: {confidence*100:.2f}%")
    print("\nAll Class Probabilities:")
    for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
        marker = " <-- PREDICTED" if i == predicted_idx else ""
        print(f"  {class_name}: {prob*100:.2f}%{marker}")
    print("="*60)


if __name__ == "__main__":
    main()

