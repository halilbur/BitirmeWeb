import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights

class ProtoNet(nn.Module):
    """
    Prototypical Network implementation using ResNet18 as feature extractor
    """
    def __init__(self, feature_dim=512):
        super(ProtoNet, self).__init__()
        # Load a ResNet18 model
        self.backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        # Remove the final classification layer and use identity
        self.backbone.fc = nn.Identity()
        self.feature_dim = feature_dim
        
    def forward(self, x):
        """Extract features from input images"""
        features = self.backbone(x)
        return features

def get_protonet_model(feature_extractor_path=None, device=None):
    """
    Create and load ProtoNet model for inference
    
    Args:
        feature_extractor_path: Path to the saved feature extractor weights
        device: Device to load the model on
    
    Returns:
        Loaded ProtoNet model
    """
    model = ProtoNet()
    
    if feature_extractor_path:
        # Load the state dictionary from your saved model
        state_dict = torch.load(feature_extractor_path, map_location=device or 'cpu')
        
        # Remove classifier (fc) keys from the state_dict if they exist
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if not key.startswith("fc."):
                cleaned_state_dict[key] = value
        
        # Load the cleaned state_dict into the backbone (non-strict loading)
        model.backbone.load_state_dict(cleaned_state_dict, strict=False)
    
    if device:
        model = model.to(device)
    
    model.eval()
    return model

def compute_prototypes(features, labels, num_classes):
    """
    Compute prototypes (class centroids) from features and labels
    
    Args:
        features: Feature vectors
        labels: Corresponding class labels
        num_classes: Total number of classes
    
    Returns:
        prototypes: Class prototypes tensor
    """
    prototypes = []
    for class_idx in range(num_classes):
        class_mask = (labels == class_idx)
        if class_mask.sum() > 0:
            class_features = features[class_mask]
            prototype = class_features.mean(dim=0)
            prototypes.append(prototype)
        else:
            # If no samples for this class, create zero prototype
            prototypes.append(torch.zeros_like(features[0]))
    
    return torch.stack(prototypes)

def classify_with_prototypes(query_features, prototypes, temperature=10.0):
    """
    Classify query features using prototypes
    
    Args:
        query_features: Features to classify
        prototypes: Class prototypes
        temperature: Temperature scaling factor for softmax (higher = more uncertain predictions)
    
    Returns:
        probabilities: Softmax probabilities over classes
    """
    # Compute squared Euclidean distances to prototypes
    dists = torch.cdist(query_features, prototypes, p=2) ** 2
    
    # Apply temperature scaling to make predictions less extreme
    # Higher temperature = more uncertain predictions
    logits = -dists / temperature
    
    # Apply softmax to get probabilities
    probs = F.softmax(logits, dim=1)
    return probs

if __name__ == '__main__':
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create ProtoNet model
    model = ProtoNet().to(device)
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Test forward pass
    with torch.no_grad():
        features = model(dummy_input)
        print("Feature shape:", features.shape)
        print("Feature dimension:", features.shape[1])
    
    print("ProtoNet model initialized successfully!")
