import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from pathlib import Path
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import random
import numpy as np
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
# ----------------------------
# Device and Directory Setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Update these paths to your directories
train_dir = r'C:\Users\halil\OneDrive\Masaüstü\pytorch\venv_bitirme2\BPVerilerFull\Train'
validation_dir = r'C:\Users\halil\OneDrive\Masaüstü\pytorch\venv_bitirme2\BPVerilerFull\Validation'
test_dir = r'C:\Users\halil\OneDrive\Masaüstü\pytorch\venv_bitirme2\BPVerilerFull\Test'

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ----------------------------
# Transforms and Custom Dataset
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted([d.name for d in self.image_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        for cls_name in self.classes:
            cls_path = self.image_dir / cls_name
            for img_path in cls_path.iterdir():
                if img_path.is_file():
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[cls_name])
                    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Create datasets for train (support), validation, and test
train_data = CustomImageDataset(train_dir, transform=transform)
val_data = CustomImageDataset(validation_dir, transform=transform)
test_data = CustomImageDataset(test_dir, transform=transform)

# DataLoaders (for prototype computation and evaluation)
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

print("Train classes:", train_data.class_to_idx)

# ----------------------------
# Load Pre-trained Model as Feature Extractor
# ----------------------------
# Load a ResNet18 model
model = models.resnet18(pretrained=True)
# Set the classifier to an identity function so that the network outputs features
model.fc = nn.Identity()
model = model.to(device)

# Load the state dictionary from your saved model
state_dict = torch.load("200_epoch_best_model.pth", map_location=device)

# Remove classifier (fc) keys from the state_dict if they exist
for key in list(state_dict.keys()):
    if key.startswith("fc."):
        del state_dict[key]

# Load the cleaned state_dict into the model (non-strict loading)
model.load_state_dict(state_dict, strict=False)
model.eval()

# ----------------------------
# Prototypical Network Functions
# ----------------------------
def compute_prototypes(loader, model):
    """
    Computes the class prototype (mean feature vector) for each class from the support set.
    """
    all_features = []
    all_labels = []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            feats = model(imgs)  # Extract features
            all_features.append(feats)
            all_labels.append(lbls)
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    unique_labels = torch.unique(all_labels)
    prototypes = []
    for label in unique_labels:
        idx = (all_labels == label)
        class_feats = all_features[idx]
        proto = class_feats.mean(dim=0)
        prototypes.append(proto)
    prototypes = torch.stack(prototypes)
    return prototypes, unique_labels

def classify_features(query_features, prototypes, temperature=10.0):
    """
    Classifies query features by computing squared Euclidean distances to the prototypes.
    Returns softmax probabilities over classes.
    
    Args:
        query_features: Feature vectors to classify
        prototypes: Class prototypes
        temperature: Temperature scaling factor for softmax (higher = more uncertain predictions)
    """
    # Compute pairwise Euclidean distances and then square them
    dists = torch.cdist(query_features, prototypes, p=2) ** 2
    
    # Apply temperature scaling to make predictions less extreme
    logits = -dists / temperature
    probs = F.softmax(logits, dim=1)
    return probs

# ----------------------------
# Compute Prototypes from the Training (Support) Set
# ----------------------------
prototypes, unique_labels = compute_prototypes(train_loader, model)
print("Computed prototypes shape:", prototypes.shape)

# Save the feature extractor and prototypes for later use (e.g., deployment)
torch.save(model.state_dict(), "protonet_feature_extractor.pth")
torch.save({
    "prototypes": prototypes.cpu(),
    "class_to_idx": train_data.class_to_idx,
    "classes": train_data.classes
}, "protonet_classifier_data.pth")

# ----------------------------
# Evaluate on the Test Set Using the Prototypical Classifier
# ----------------------------
all_preds = []
all_true = []
with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs = imgs.to(device)
        feats = model(imgs)  # Extract query features
        probs = classify_features(feats, prototypes, temperature=10.0)
        preds = torch.argmax(probs, dim=1)
        all_preds.append(preds.cpu())
        all_true.append(lbls)
all_preds = torch.cat(all_preds)
all_true = torch.cat(all_true)

accuracy = (all_preds == all_true).float().mean().item()
print(f"Test set accuracy using the prototypical network classifier: {accuracy*100:.2f}%")


# Precision, recall, f1-score hesapla
precision = precision_score(all_true, all_preds, average='weighted')
recall = recall_score(all_true, all_preds, average='weighted')
f1 = f1_score(all_true, all_preds, average='weighted')

print(f"📊 Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
print("\n📋 Classification Report:")
print(classification_report(all_true, all_preds, target_names=train_data.classes))



# --- Confusion Matrix Heatmap ---
# Compute confusion matrix
cm = confusion_matrix(all_true, all_preds)

# Plot heatmap
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=train_data.classes, yticklabels=train_data.classes)
plt.xlabel("Tahmin Edilen Etiket")
plt.ylabel("Gerçek Etiket")

title_text = "Prototype Network Confusion Matrix\n"
metrics_text = (
    f"Test Accuracy: {accuracy:.4f}    |    "
    f"Precision: {precision:.4f}    |    "
    f"Recall: {recall:.4f}    |    "
    f"F1 Score: {f1:.4f}"
)
plt.title(title_text + metrics_text, fontsize=12, pad=20)

plt.tight_layout()
plt.show()

# After your training loop, save the model state_dict for later use
# Example (uncomment and place after training):
# torch.save(model.state_dict(), "trained_protonet_model.pth")