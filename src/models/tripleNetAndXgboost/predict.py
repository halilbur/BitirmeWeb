import torch
import torchvision.transforms as transforms
from PIL import Image
import xgboost as xgb
import numpy as np
import os
from .model import TripletNet # Import the correct TripletNet class

# Define the same transformations used during training
IMG_SIZE = 224
data_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define class names - ensure this matches the order used for training XGBoost
# This should be the same as CLOTHING_CLASSES in main.py
CLOTHING_CLASSES = [
    'Ceket', 'Elbise', 'Etek', 'Gömlek', 'Hırka', 'Kazak',
    'Mont', 'Pantolon', 'Sweatshirt', 'Tshirt', 'Yelek', 'Şort'
]

def load_triplet_model(model_path, device='cpu'):
    # Create TripletNet instance with the same embedding dimension as in training (128)
    model = TripletNet(embedding_dim=128)
    
    # Load the state_dict directly into the TripletNet model
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    return model

def load_xgb_model(model_path):
    # Use XGBClassifier to match the notebook's usage
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(model_path)
    return xgb_model

def get_embedding(image_path, triplet_model, device='cpu'):
    image = Image.open(image_path).convert('RGB')
    image_tensor = data_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = triplet_model(image_tensor)  # Direct forward pass
    return embedding.cpu().numpy()

def predict_with_triplenet_xgboost(image_path, triplet_model_path, xgb_model_path, device='cpu'):
    # Load models
    triplet_model = load_triplet_model(triplet_model_path, device)
    xgb_classifier = load_xgb_model(xgb_model_path)

    # Get embedding
    embedding = get_embedding(image_path, triplet_model, device)

    # Predict with XGBoost - using numpy array directly like in the notebook
    # Notebook code: pred = xgb_model.predict(embedding_np); return pred[0]
    xgb_pred = xgb_classifier.predict(embedding)
    
    # Debug: Print shape and content of xgb_pred
    print(f"XGBoost prediction shape: {xgb_pred.shape}")
    print(f"XGBoost prediction content: {xgb_pred}")
    print(f"XGBoost prediction type: {type(xgb_pred)}")
    
    # The notebook returns pred[0], which suggests xgb_pred is an array with the predicted class index
    predicted_index = int(xgb_pred[0])  # Get the predicted class index
    predicted_class = CLOTHING_CLASSES[predicted_index]
    
    # For XGBClassifier.predict(), we get class predictions, not probabilities
    # To get probabilities, we need to use predict_proba()
    try:
        xgb_probs = xgb_classifier.predict_proba(embedding)
        print(f"XGBoost probabilities shape: {xgb_probs.shape}")
        print(f"XGBoost probabilities: {xgb_probs}")
        
        # Get probabilities for this sample
        if len(xgb_probs.shape) == 2 and xgb_probs.shape[0] == 1:
            probs = xgb_probs[0]  # Take first (and only) sample
        else:
            probs = xgb_probs
            
        # Get top 3 predictions
        top_indices = np.argsort(probs)[::-1][:3]
        top_predictions = []
        for idx in top_indices:
            top_predictions.append({
                'class': CLOTHING_CLASSES[idx],
                'confidence': round(float(probs[idx]) * 100, 2)
            })
        
        # The primary confidence for the top class
        primary_confidence = round(float(probs[predicted_index]) * 100, 2)
        
    except Exception as e:
        print(f"Warning: Could not get probabilities: {e}")
        # Fallback: create dummy probabilities
        probs = np.zeros(len(CLOTHING_CLASSES))
        probs[predicted_index] = 1.0  # 100% confidence for predicted class
        
        top_predictions = [
            {'class': predicted_class, 'confidence': 100.0},
            {'class': CLOTHING_CLASSES[1] if predicted_index != 1 else CLOTHING_CLASSES[0], 'confidence': 0.0},
            {'class': CLOTHING_CLASSES[2] if predicted_index != 2 else CLOTHING_CLASSES[0], 'confidence': 0.0}
        ]
        primary_confidence = 100.0

    return predicted_class, primary_confidence, top_predictions

if __name__ == '__main__':
    # Example usage:
    # Ensure you have a test image and the model paths are correct
    # These paths are relative to the BitirmeWeb/src directory if you run main.py from there
    # For direct execution of this script, adjust paths accordingly or use absolute paths.
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, '..', '..', '..') # Moves up to BitirmeWeb directory

    test_image_path = os.path.join(base_dir, 'static', 'uploads', 'dummy.jpg') # Create a dummy.jpg for testing
    # Create a dummy image if it doesn't exist for testing
    if not os.path.exists(test_image_path):
        try:
            from PIL import Image as PImage
            dummy_img = PImage.new('RGB', (100, 100), color = 'red')
            os.makedirs(os.path.dirname(test_image_path), exist_ok=True)
            dummy_img.save(test_image_path)
            print(f"Created dummy image at {test_image_path}")
        except Exception as e:
            print(f"Could not create dummy image: {e}")

    print("current_dir:", current_dir)
    triplet_model_weights_path = os.path.join(current_dir, 'Triplet50.pth')
    xgb_model_weights_path = os.path.join(current_dir, 'xgb_model.json')

    if not os.path.exists(triplet_model_weights_path):
        print(f"Triplet model weights not found at {triplet_model_weights_path}")
    if not os.path.exists(xgb_model_weights_path):
        print(f"XGBoost model not found at {xgb_model_weights_path}")
    if not os.path.exists(test_image_path):
        print(f"Test image not found at {test_image_path}")

    if os.path.exists(triplet_model_weights_path) and os.path.exists(xgb_model_weights_path) and os.path.exists(test_image_path):
        predicted_class, confidence, top_preds = predict_with_triplenet_xgboost(
            test_image_path,
            triplet_model_weights_path,
            xgb_model_weights_path
        )
        print(f"Predicted Class (Triplet+XGB): {predicted_class}")
        print(f"Confidence: {confidence}%")
        print(f"Top 3 Predictions: {top_preds}")
    else:
        print("Skipping example usage due to missing files.")
