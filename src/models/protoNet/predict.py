import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from .model import get_protonet_model, classify_with_prototypes
from .similarity_search import find_similar_images

# Flask projesindeki sınıf isimleriyle uyumlu olmalı
CLOTHING_CLASSES = [
    'Ceket', 'Elbise', 'Etek', 'Gömlek', 'Hırka', 'Kazak', 
    'Mont', 'Pantolon', 'Sweatshirt', 'Tshirt', 'Yelek', 'Şort'
]

# Inference için transform - same as training
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

def load_prototypes_and_model(feature_extractor_path, classifier_data_path, device='cpu'):
    """
    Load the ProtoNet feature extractor and pre-computed prototypes
    
    Args:
        feature_extractor_path: Path to saved feature extractor weights
        classifier_data_path: Path to saved prototypes and class information
        device: Device to load models on
    
    Returns:
        model: Loaded ProtoNet model
        prototypes: Pre-computed class prototypes
        class_info: Dictionary with class information
    """
    # Load the feature extractor model
    model = get_protonet_model(feature_extractor_path, device)
    
    # Load the prototypes and class information
    classifier_data = torch.load(classifier_data_path, map_location=device)
    prototypes = classifier_data['prototypes'].to(device)
    class_info = {
        'class_to_idx': classifier_data['class_to_idx'],
        'classes': classifier_data['classes']
    }
    
    return model, prototypes, class_info

def predict_with_protonet(image_path, feature_extractor_path, classifier_data_path, device='cpu'):
    """
    Predict clothing class using ProtoNet
    
    Args:
        image_path: Path to the image to classify
        feature_extractor_path: Path to saved feature extractor weights
        classifier_data_path: Path to saved prototypes and class information
        device: Device to run inference on
    
    Returns:
        predicted_class: Predicted clothing class name
        confidence: Confidence score for the prediction
        top_predictions: List of top 3 predictions with confidence scores
    """
    # Load model and prototypes
    model, prototypes, class_info = load_prototypes_and_model(
        feature_extractor_path, classifier_data_path, device
    )
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    input_tensor = inference_transform(image).unsqueeze(0).to(device)
      # Extract features
    with torch.no_grad():
        query_features = model(input_tensor)
        
        # Classify using prototypes with temperature scaling
        # Temperature=10.0 makes predictions less extreme (more realistic probabilities)
        probs = classify_with_prototypes(query_features, prototypes, temperature=10.0)
        probs = probs[0]  # Get probabilities for the single query
        
        # Get prediction
        confidence, pred_idx = torch.max(probs, dim=0)
        predicted_class = CLOTHING_CLASSES[pred_idx.item()]
        
        # Get top 3 predictions
        probs_np = probs.cpu().numpy()
        top_indices = np.argsort(probs_np)[::-1][:3]
        top_predictions = []
        for idx in top_indices:
            top_predictions.append({
                'class': CLOTHING_CLASSES[idx],
                'confidence': round(float(probs_np[idx]) * 100, 2)
            })
    
    return predicted_class, round(confidence.item() * 100, 2), top_predictions

def predict_similar_items_with_protonet(image_path, feature_extractor_path, similar_items_dir, device='cpu', top_k=3):
    """
    Find similar items using ProtoNet embeddings instead of classification
    
    Args:
        image_path: Path to the input image
        feature_extractor_path: Path to the feature extractor model
        similar_items_dir: Directory containing the similar items database
        device: Device to run on
        top_k: Number of similar items to return
    
    Returns:
        similar_items: List of similar items with their information
        avg_similarity: Average similarity score
        most_similar_class: The class of the most similar item
    """
    try:
        # Use the similarity search to find similar images
        similar_items = find_similar_images(
            image_path, 
            feature_extractor_path, 
            similar_items_dir, 
            device, 
            top_k
        )
        
        if not similar_items:
            return [], 0.0, "Unknown"
        
        # Calculate average similarity
        avg_similarity = sum(item['similarity'] for item in similar_items) / len(similar_items)
        
        # Get the most common class among similar items
        class_counts = {}
        for item in similar_items:
            class_name = item['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        most_similar_class = max(class_counts, key=class_counts.get) if class_counts else "Unknown"
        
        return similar_items, round(avg_similarity, 2), most_similar_class
        
    except Exception as e:
        print(f"Error in similarity-based prediction: {e}")
        return [], 0.0, "Error"

if __name__ == '__main__':
    # Example usage
    import os
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Paths to model files (adjust as needed)
    feature_extractor_path = os.path.join(current_dir, 'protonet_feature_extractor.pth')
    classifier_data_path = os.path.join(current_dir, 'protonet_classifier_data.pth')
    
    # Create a dummy test image for testing
    test_image_path = os.path.join(current_dir, '..', '..', '..', 'static', 'uploads', 'dummy_proto.jpg')
    
    if not os.path.exists(test_image_path):
        try:
            from PIL import Image as PImage
            dummy_img = PImage.new('RGB', (224, 224), color='blue')
            os.makedirs(os.path.dirname(test_image_path), exist_ok=True)
            dummy_img.save(test_image_path)
            print(f"Created dummy image at {test_image_path}")
        except Exception as e:
            print(f"Could not create dummy image: {e}")
    
    # Check if model files exist
    if not os.path.exists(feature_extractor_path):
        print(f"Feature extractor not found at {feature_extractor_path}")
    if not os.path.exists(classifier_data_path):
        print(f"Classifier data not found at {classifier_data_path}")
    
    if (os.path.exists(feature_extractor_path) and 
        os.path.exists(classifier_data_path) and 
        os.path.exists(test_image_path)):
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            predicted_class, confidence, top_preds = predict_with_protonet(
                test_image_path,
                feature_extractor_path,
                classifier_data_path,
                device
            )
            print(f"Predicted Class (ProtoNet): {predicted_class}")
            print(f"Confidence: {confidence}%")
            print(f"Top 3 Predictions: {top_preds}")
            
            # Similarity-based prediction
            similar_items, avg_similarity, most_similar_class = predict_similar_items_with_protonet(
                test_image_path,
                feature_extractor_path,
                os.path.join(current_dir, 'similar_items_db'),
                device
            )
            print(f"Most Similar Class: {most_similar_class}")
            print(f"Average Similarity: {avg_similarity}")
            print("Similar Items:")
            for item in similar_items:
                print(f"- {item['class']} (Similarity: {item['similarity']})")
                
        except Exception as e:
            print(f"Error during prediction: {e}")
    else:
        print("Skipping example usage due to missing files.")
        print("Make sure to run proto_net.py first to generate the required model files.")
