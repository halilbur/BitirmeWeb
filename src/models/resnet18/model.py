import torch
import torch.nn as nn
from torchvision import models

def get_resnet18_model(num_classes=12, weights_path=None, device=None):
    # Use weights=None to avoid downloading ImageNet weights since we load our own
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes),
    )
    if weights_path:
        state_dict = torch.load(weights_path, map_location=device or 'cpu')
        model.load_state_dict(state_dict)
    if device:
        model = model.to(device)
    model.eval()
    return model
