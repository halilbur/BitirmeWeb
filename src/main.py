import sys
import os

# Add the current directory (src) to Python path to find models
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Also add the parent directory to handle different deployment scenarios
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
from datetime import datetime
import uuid
import random  # random modülü eklendi

# Debug info for troubleshooting
print(f"Main.py Debug - Current dir: {current_dir}")
print(f"Main.py Debug - Parent dir: {parent_dir}")
print(f"Main.py Debug - Models dir exists: {os.path.exists(os.path.join(current_dir, 'models'))}")

# Try different import paths for deployment compatibility
try:
    # First try: relative import from current directory
    from models.resnet18.predict import predict_image
    from models.tripleNetAndXgboost.predict import predict_with_triplenet_xgboost
    print("Successfully imported models using relative imports")
except ImportError as e1:
    print(f"First import attempt failed: {e1}")
    try:
        # Second try: with src prefix
        from src.models.resnet18.predict import predict_image
        from src.models.tripleNetAndXgboost.predict import predict_with_triplenet_xgboost
        print("Successfully imported models using src prefix")
    except ImportError as e2:
        print(f"Second import attempt failed: {e2}")
        try:
            # Third try: add models directory to path and import directly
            models_path = os.path.join(current_dir, 'models')
            if models_path not in sys.path:
                sys.path.insert(0, models_path)
            from resnet18.predict import predict_image
            from tripleNetAndXgboost.predict import predict_with_triplenet_xgboost
            print("Successfully imported models after adding models path")
        except ImportError as e3:
            print(f"All import attempts failed: {e1}, {e2}, {e3}")
            print(f"Current directory: {current_dir}")
            print(f"Python path: {sys.path}")
            print(f"Models path exists: {os.path.exists(os.path.join(current_dir, 'models'))}")
            # Import a dummy function to prevent the app from crashing
            def predict_image(*args, **kwargs):
                return "Error", 0.0, []
            def predict_with_triplenet_xgboost(*args, **kwargs):
                return "Error", 0.0, []


# Create Flask application instance
app = Flask(__name__, static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'static'))
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

# Configuration
UPLOAD_FOLDER = os.path.join(app.static_folder, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Define model paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESNET_MODEL_PATH = os.path.join(BASE_DIR, 'models/resnet18/BestModel_withLoss.pth')
TRIPLET_MODEL_PATH = os.path.join(BASE_DIR, 'models/tripleNetAndXgboost/Triplet50.pth') # Added path
XGB_MODEL_PATH = os.path.join(BASE_DIR, 'models/tripleNetAndXgboost/xgb_model.json') # Added path


# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 12 Clothing classes - update these to match your model's classes
CLOTHING_CLASSES = [
    'Ceket', 'Elbise', 'Etek', 'Gömlek', 'Hırka', 'Kazak', 
    'Mont', 'Pantolon', 'Sweatshirt', 'Tshirt', 'Yelek', 'Şort'
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_clothing_class(image_path):
    """
    ResNet18 modelini ve .pth dosyasını kullanarak gerçek tahmin yapar
    """
    # Use cached model instead of reloading
    model = get_resnet_model()
    
    # Process image directly without reloading model
    from torchvision import transforms
    from PIL import Image
    import torch
    import numpy as np
    
    # Inference transform (same as in predict.py)
    inference_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    device = 'cpu'
    image = Image.open(image_path).convert('RGB')
    input_tensor = inference_transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)
        predicted_class = CLOTHING_CLASSES[pred_idx.item()]
    
    # En yüksek 3 tahmini bul
    probs_numpy = probs.cpu().numpy()
    top_indices = np.argsort(probs_numpy)[::-1][:3]
    top_predictions = []
    for idx in top_indices:
        top_predictions.append({
            'class': CLOTHING_CLASSES[idx],
            'confidence': round(probs_numpy[idx]*100, 2)
        })
    return predicted_class, round(conf.item()*100, 2), top_predictions

def get_similar_items(clothing_class, num_items=8):
    """
    Belirtilen giysi sınıfı için benzer ürünleri bulur.
    Fotoğrafları static/similar_items/{clothing_class}/ klasöründen alır.
    """
    similar_items_dir = os.path.join(app.static_folder, 'similar_items', clothing_class)
    similar_items = []
    
    if os.path.exists(similar_items_dir):
        try:
            available_images = [f for f in os.listdir(similar_items_dir) 
                                if os.path.isfile(os.path.join(similar_items_dir, f)) 
                                and f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
            
            selected_images = random.sample(available_images, min(len(available_images), num_items))
            
            for i, img_name in enumerate(selected_images):
                similar_items.append({
                    'id': i + 1,
                    'name': f'{clothing_class} - {i+1}', # Daha iyi bir isimleme düşünülebilir
                    #'description': f'Benzer {clothing_class} modeli.',
                    'image_url': url_for('static', filename=f'similar_items/{clothing_class}/{img_name}'),
                    #'similarity': random.randint(80, 95), # Bu da rastgele
                    #'price': random.randint(20, 200) if random.choice([True, False]) else None # Bu da rastgele
                })
        except Exception as e:
            print(f"Error reading similar items for {clothing_class}: {e}")
            # Hata durumunda boş liste veya varsayılan birkaç öğe döndürülebilir
            pass # Şimdilik hata olursa boş liste dönecek

    # Eğer hiç benzer ürün bulunamazsa veya klasör yoksa, placeholder birkaç öğe eklenebilir
    if not similar_items:
        for i in range(num_items): # Varsayılan olarak 4 placeholder gösterelim
            similar_items.append({
                'id': -(i + 1), # Placeholder olduğunu belirtmek için negatif id
                'name': f'Placeholder {clothing_class} #{i+1}',
                'description': 'Bu kategori için benzer ürünler yakında eklenecektir.',
                'image_url': url_for('static', filename='resimler/zeki_kus.jpg'), # Genel bir placeholder resmi
                'similarity': 0,
                'price': None
            })
            if len(similar_items) >= 4: # En fazla 4 placeholder
                break
                
    return similar_items

def predict_with_triplenet_xgboost_cached(image_path):
    """
    TripletNet+XGBoost modellerini kullanarak tahmin yapar (cached models)
    """
    triplet_model, xgb_model = get_triplet_and_xgb_models()
    
    from torchvision import transforms
    from PIL import Image
    import torch
    import numpy as np
    
    # Transform (same as in TripletNet predict.py)
    IMG_SIZE = 224
    data_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    device = 'cpu'
    
    # Get embedding from image
    image = Image.open(image_path).convert('RGB')
    image_tensor = data_transforms(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = triplet_model(image_tensor)
    
    embedding_np = embedding.cpu().numpy()
    
    # Predict with XGBoost
    xgb_pred = xgb_model.predict(embedding_np)
    predicted_index = int(xgb_pred[0])
    predicted_class = CLOTHING_CLASSES[predicted_index]
    
    # Get probabilities
    try:
        xgb_probs = xgb_model.predict_proba(embedding_np)[0]
        
        # Get top 3 predictions
        top_indices = np.argsort(xgb_probs)[::-1][:3]
        top_predictions = []
        for idx in top_indices:
            top_predictions.append({
                'class': CLOTHING_CLASSES[idx],
                'confidence': round(float(xgb_probs[idx]) * 100, 2)
            })
        
        primary_confidence = round(float(xgb_probs[predicted_index]) * 100, 2)
        
    except Exception as e:
        print(f"Warning: Could not get probabilities: {e}")
        # Fallback
        top_predictions = [
            {'class': predicted_class, 'confidence': 100.0},
            {'class': CLOTHING_CLASSES[1] if predicted_index != 1 else CLOTHING_CLASSES[0], 'confidence': 0.0},
            {'class': CLOTHING_CLASSES[2] if predicted_index != 2 else CLOTHING_CLASSES[0], 'confidence': 0.0}
        ]
        primary_confidence = 100.0
    
    return predicted_class, primary_confidence, top_predictions
_resnet_model = None
_triplet_model = None
_xgb_model = None

def get_resnet_model():
    """Get cached ResNet18 model or load it if not cached"""
    global _resnet_model
    if _resnet_model is None:
        try:
            print("Loading ResNet18 model...")
            from models.resnet18.model import get_resnet18_model
            _resnet_model = get_resnet18_model(
                num_classes=len(CLOTHING_CLASSES), 
                weights_path=RESNET_MODEL_PATH, 
                device='cpu'
            )
            print("ResNet18 model loaded successfully")
        except Exception as e:
            print(f"Error loading ResNet18 model: {e}")
            _resnet_model = None
            raise e
    return _resnet_model

def get_triplet_and_xgb_models():
    """Get cached TripletNet and XGBoost models or load them if not cached"""
    global _triplet_model, _xgb_model
    if _triplet_model is None or _xgb_model is None:
        try:
            print("Loading TripletNet and XGBoost models...")
            from models.tripleNetAndXgboost.predict import load_triplet_model, load_xgb_model
            _triplet_model = load_triplet_model(TRIPLET_MODEL_PATH, device='cpu')
            _xgb_model = load_xgb_model(XGB_MODEL_PATH)
            print("TripletNet and XGBoost models loaded successfully")
        except Exception as e:
            print(f"Error loading TripletNet/XGBoost models: {e}")
            _triplet_model = None
            _xgb_model = None
            raise e
    return _triplet_model, _xgb_model

def preload_models():
    """Preload all models on app startup to avoid cold start issues"""
    print("Preloading models on startup...")
    try:
        get_resnet_model()
        get_triplet_and_xgb_models()
        print("All models preloaded successfully")
    except Exception as e:
        print(f"Error preloading models: {e}")

# Preload models when app starts
with app.app_context():
    preload_models()

# Routes
@app.route('/')
def home():
    return render_template('index.html', title='Find Similar Clothes')

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Check if models are loaded
        resnet_status = _resnet_model is not None
        triplet_status = _triplet_model is not None
        xgb_status = _xgb_model is not None
        
        return jsonify({
            'status': 'healthy',
            'models': {
                'resnet18': resnet_status,
                'triplet': triplet_status,
                'xgboost': xgb_status
            }
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if image was uploaded
        if 'image' not in request.files:
            flash('No image uploaded')
            return redirect(url_for('home'))
        
        file = request.files['image']
        if file.filename == '':
            flash('No image selected')
            return redirect(url_for('home'))
        
        if file and allowed_file(file.filename):
            # Generate unique filename
            filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Add timeout handling and memory cleanup
            import gc
            
            try:
                # Get predictions from ResNet18 model
                resnet_predicted_class, resnet_confidence, resnet_top_predictions = predict_clothing_class(filepath)
                resnet_similar_items = get_similar_items(resnet_predicted_class)
                
                # Force garbage collection between model predictions to free memory
                gc.collect()
                
                # Get predictions from TripletNet+XGBoost model
                triplet_predicted_class, triplet_confidence, triplet_top_predictions = predict_with_triplenet_xgboost_cached(filepath)
                triplet_similar_items = get_similar_items(triplet_predicted_class)
                
                # Clean up memory after predictions
                gc.collect()
                
            except Exception as model_error:
                print(f"Model prediction error: {model_error}")
                # Fallback to dummy predictions to avoid total failure
                resnet_predicted_class = "Error"
                resnet_confidence = 0.0
                resnet_top_predictions = [{"class": "Error", "confidence": 0.0}]
                resnet_similar_items = []
                
                triplet_predicted_class = "Error"
                triplet_confidence = 0.0
                triplet_top_predictions = [{"class": "Error", "confidence": 0.0}]
                triplet_similar_items = []
                
            # Prepare data for template
            uploaded_image_url = url_for('static', filename=f'uploads/{filename}')
            
            return render_template('results.html',
                                 title='Prediction Results',
                                 uploaded_image_url=uploaded_image_url,
                                 # ResNet18 results
                                 resnet_predicted_class=resnet_predicted_class,
                                 resnet_confidence=resnet_confidence,
                                 resnet_top_predictions=resnet_top_predictions,
                                 resnet_similar_items=resnet_similar_items,
                                 # TripletNet+XGBoost results
                                 triplet_predicted_class=triplet_predicted_class,
                                 triplet_confidence=triplet_confidence,
                                 triplet_top_predictions=triplet_top_predictions,
                                 triplet_similar_items=triplet_similar_items
                                )
        else:
            flash('Invalid file type. Please upload an image.')
            return redirect(url_for('home'))
            
    except Exception as e:
        print(f'Error processing image: {str(e)}')
        flash(f'Error processing image: {str(e)}')
        return redirect(url_for('home'))

@app.route('/about')
def about():
    return render_template('about.html', title='About ClothMatch')

@app.route('/models')
def models():
    return render_template('models.html', title='AI Models')

@app.route('/team')
def team():
    return render_template('team.html', title='Project Team')

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(413)
def too_large(error):
    flash('File too large. Please upload a smaller image.')
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)