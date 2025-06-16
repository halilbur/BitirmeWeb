from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
from datetime import datetime
import os
import uuid
import random  # random modülü eklendi
from models.resnet18.predict import predict_image
from models.tripleNetAndXgboost.predict import predict_with_triplenet_xgboost # Added import
from models.protoNet.predict import predict_with_protonet # Added ProtoNet import


# Create Flask application instance
app = Flask(__name__, static_folder='../static')
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

# Configuration
UPLOAD_FOLDER = '../static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define model paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESNET_MODEL_PATH = os.path.join(BASE_DIR, 'models/resnet18/BestModel_withLoss.pth')
TRIPLET_MODEL_PATH = os.path.join(BASE_DIR, 'models/tripleNetAndXgboost/Triplet50.pth') # Added path
XGB_MODEL_PATH = os.path.join(BASE_DIR, 'models/tripleNetAndXgboost/xgb_model.json') # Added path
# ProtoNet model paths
PROTONET_FEATURE_EXTRACTOR_PATH = os.path.join(BASE_DIR, 'models/protoNet/protonet_feature_extractor.pth')
PROTONET_CLASSIFIER_DATA_PATH = os.path.join(BASE_DIR, 'models/protoNet/protonet_classifier_data.pth')


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
    # Model ağırlık dosyasının yolu
    weights_path = RESNET_MODEL_PATH # Use defined path
    device = 'cpu'  # Sunucuda GPU varsa 'cuda' olarak değiştirilebilir
    predicted_class, confidence, probs = predict_image(image_path, weights_path, device=device)
    # En yüksek 3 tahmini bul
    import numpy as np
    top_indices = np.argsort(probs)[::-1][:3]
    top_predictions = []
    for idx in top_indices:
        top_predictions.append({
            'class': CLOTHING_CLASSES[idx],
            'confidence': round(probs[idx]*100, 2)
        })
    return predicted_class, round(confidence*100, 2), top_predictions

def get_similar_items(clothing_class, num_items=8):
    """
    Belirtilen giysi sınıfı için benzer ürünleri bulur.
    Fotoğrafları src/static/similar_items/{clothing_class}/ klasöründen alır.
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

# Routes
@app.route('/')
def home():
    return render_template('index.html', title='Find Similar Clothes')

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
            
            # Get predictions from ResNet18 model
            resnet_predicted_class, resnet_confidence, resnet_top_predictions = predict_clothing_class(filepath)
            resnet_similar_items = get_similar_items(resnet_predicted_class)
            
            # Get predictions from TripletNet+XGBoost model
            device = 'cpu' # or 'cuda' if available
            triplet_predicted_class, triplet_confidence, triplet_top_predictions = predict_with_triplenet_xgboost(
                filepath,
                TRIPLET_MODEL_PATH,
                XGB_MODEL_PATH,
                device=device
            )
            triplet_similar_items = get_similar_items(triplet_predicted_class) # Assuming get_similar_items can be reused for now
              # Get predictions from ProtoNet model
            try:
                proto_predicted_class, proto_confidence, proto_top_predictions = predict_with_protonet(
                    filepath,
                    PROTONET_FEATURE_EXTRACTOR_PATH,
                    PROTONET_CLASSIFIER_DATA_PATH,
                    device=device
                )
                proto_similar_items = get_similar_items(proto_predicted_class)
            except Exception as e:
                print(f"ProtoNet prediction error: {e}")
                # Fallback values if ProtoNet fails
                proto_predicted_class = "Error"
                proto_confidence = 0.0
                proto_top_predictions = [
                    {'class': 'Error', 'confidence': 0.0},
                    {'class': 'Error', 'confidence': 0.0},
                    {'class': 'Error', 'confidence': 0.0}
                ]
                proto_similar_items = []
            
            # Prepare data for template
            uploaded_image_url = f'/static/uploads/{filename}'
            
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
                                 triplet_similar_items=triplet_similar_items,
                                 # ProtoNet results
                                 proto_predicted_class=proto_predicted_class,
                                 proto_confidence=proto_confidence,
                                 proto_top_predictions=proto_top_predictions,
                                 proto_similar_items=proto_similar_items
                                )
        else:
            flash('Invalid file type. Please upload an image.')
            return redirect(url_for('home'))
            
    except Exception as e:
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