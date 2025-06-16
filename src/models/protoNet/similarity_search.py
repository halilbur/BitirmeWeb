import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import pickle
from .model import get_protonet_model

def create_embedding_database(feature_extractor_path, similar_items_dir, device='cpu'):
    """
    Create embeddings for all images in the similar_items directory and save to cache
    
    Args:
        feature_extractor_path: Path to the feature extractor model
        similar_items_dir: Directory containing similar_items with subdirectories for each class
        device: Device to run on
    
    Returns:
        embeddings_dict: Dictionary with image paths as keys and embeddings as values    """
    print("Creating embedding database...")
    
    # Load feature extractor
    model = get_protonet_model(feature_extractor_path, device)
    model.eval()
    
    # Define image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    embeddings_dict = {}
    image_paths = []
    embeddings_list = []
    
    # Walk through all subdirectories in similar_items
    for class_name in os.listdir(similar_items_dir):
        class_dir = os.path.join(similar_items_dir, class_name)
        if os.path.isdir(class_dir):
            print(f"Processing class: {class_name}")
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    img_path = os.path.join(class_dir, img_name)
                    try:
                        # Load and preprocess image
                        image = Image.open(img_path).convert('RGB')
                        input_tensor = transform(image).unsqueeze(0).to(device)
                        
                        # Extract features
                        with torch.no_grad():
                            embedding = model(input_tensor)
                            embedding = F.normalize(embedding, p=2, dim=1)  # L2 normalize
                        
                        # Store relative path for web serving
                        relative_path = os.path.join('similar_items', class_name, img_name).replace('\\', '/')
                        
                        embeddings_dict[relative_path] = embedding.cpu().numpy()
                        image_paths.append(relative_path)
                        embeddings_list.append(embedding.cpu().numpy())
                        
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
                        continue
    
    # Convert to numpy arrays for efficient similarity search
    if embeddings_list:
        embeddings_array = np.vstack(embeddings_list)
        
        # Save to cache file
        cache_data = {
            'embeddings_dict': embeddings_dict,
            'image_paths': image_paths,
            'embeddings_array': embeddings_array
        }
        
        cache_path = os.path.join(os.path.dirname(feature_extractor_path), 'embeddings_cache.pkl')
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"Created embeddings for {len(image_paths)} images")
        print(f"Cache saved to: {cache_path}")
        
        return cache_data
    else:
        print("No images found to create embeddings")
        return None

def load_embedding_database(feature_extractor_path):
    """
    Load cached embeddings database
    
    Args:
        feature_extractor_path: Path to feature extractor (used to find cache file)
    
    Returns:
        cache_data: Dictionary with embeddings and image paths
    """
    cache_path = os.path.join(os.path.dirname(feature_extractor_path), 'embeddings_cache.pkl')
    
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        print(f"Loaded embeddings for {len(cache_data['image_paths'])} images from cache")
        return cache_data
    else:
        print(f"No cache found at {cache_path}")
        return None

def find_similar_images(query_image_path, feature_extractor_path, similar_items_dir, device='cpu', top_k=3):
    """
    Find similar images using ProtoNet embeddings
    
    Args:
        query_image_path: Path to the query image
        feature_extractor_path: Path to the feature extractor model
        similar_items_dir: Directory containing similar items
        device: Device to run on
        top_k: Number of similar images to return
    
    Returns:
        similar_items: List of dictionaries with similar image information
    """
    try:
        # Try to load cached embeddings first
        cache_data = load_embedding_database(feature_extractor_path)
          # If no cache, create embeddings database
        if cache_data is None:
            cache_data = create_embedding_database(feature_extractor_path, similar_items_dir, device)
            if cache_data is None:
                return []
        
        embeddings_dict = cache_data['embeddings_dict']
        image_paths = cache_data['image_paths']
        embeddings_array = cache_data['embeddings_array']
        
        if len(image_paths) == 0:
            return []
        
        # Load feature extractor
        model = get_protonet_model(feature_extractor_path, device)
        model.eval()
        
        # Define image transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Extract features from query image
        query_image = Image.open(query_image_path).convert('RGB')
        query_tensor = transform(query_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            query_embedding = model(query_tensor)
            query_embedding = F.normalize(query_embedding, p=2, dim=1)  # L2 normalize
        
        query_embedding_np = query_embedding.cpu().numpy()
        
        # Calculate cosine similarities (since embeddings are normalized, dot product = cosine similarity)
        similarities = np.dot(embeddings_array, query_embedding_np.T).flatten()
        
        # Get top k most similar images
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        similar_items = []
        for i, idx in enumerate(top_indices):
            img_path = image_paths[idx]
            similarity_score = similarities[idx]
            
            # Extract class name and image name from path
            path_parts = img_path.split('/')
            class_name = path_parts[1] if len(path_parts) > 1 else 'Unknown'
            img_name = path_parts[-1] if len(path_parts) > 0 else 'Unknown'
            
            similar_items.append({
                'id': i + 1,
                'name': f'{class_name} - {img_name}',
                'description': f'Similarity: {similarity_score:.3f}',
                'image_url': f'/static/{img_path}',
                'similarity': round(float(similarity_score) * 100, 2),
                'class': class_name,
                'embedding_distance': float(1 - similarity_score)  # Convert similarity to distance
            })
        
        return similar_items
        
    except Exception as e:
        print(f"Error in similarity search: {e}")
        return []

def update_embedding_cache(feature_extractor_path, similar_items_dir, device='cpu'):
    """
    Force update the embedding cache (useful when new images are added)
    """
    cache_path = os.path.join(os.path.dirname(feature_extractor_path), 'embeddings_cache.pkl')
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print("Removed old cache")
    
    return create_embedding_database(feature_extractor_path, similar_items_dir, device)
