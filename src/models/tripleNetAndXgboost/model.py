import torch
import torch.nn as nn
import torchvision.models as models

class TripletNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super(TripletNet, self).__init__()
        # Use pretrained=False to avoid downloading ImageNet weights since we load our own
        resnet = models.resnet18(pretrained=False)
        # Remove the final classification layer (fc layer)
        modules = list(resnet.children())[:-1]  # up to avgpool
        self.backbone = nn.Sequential(*modules)
        self.embedding = nn.Linear(resnet.fc.in_features, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.embedding(x)
        x = nn.functional.normalize(x, p=2, dim=1)  # L2 normalize
        return x

def get_tripletnet_model(embedding_dim=128, weights_path=None, device=None):
    model = TripletNet(embedding_dim=embedding_dim)
    if weights_path:
        state_dict = torch.load(weights_path, map_location=device or 'cpu')
        model.load_state_dict(state_dict)
    if device:
        model = model.to(device)
    model.eval() # Set the model to evaluation mode
    return model

if __name__ == '__main__':
    # Example of how to initialize the model (matching your notebook)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create TripletNet exactly like in your notebook
    model = TripletNet(embedding_dim=128).to(device)
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Test forward pass (single image embedding extraction)
    with torch.no_grad():
        embedding = model(dummy_input)
        print("Embedding shape:", embedding.shape)
        print("Embedding dimension:", embedding.shape[1])
    
    print("TripletNet model initialized successfully!")
    print("This matches the model structure from your Triple50.ipynb notebook.")
