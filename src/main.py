from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
from datetime import datetime
import os
import uuid

# Create Flask application instance
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 12 Clothing classes - update these to match your model's classes
CLOTHING_CLASSES = [
    'dress', 'pants', 'tshirt', 'shirt', 'shorts', 'jeans', 
    'skirt', 'jacket', 'sweater', 'hoodie', 'blouse', 'coat'
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_clothing_class(image_path):
    """
    Placeholder for your actual model prediction
    Replace this with your actual deep learning model inference
    """
    import random
    
    # Simulate model prediction - replace with your actual model
    predicted_class = random.choice(CLOTHING_CLASSES)
    confidence = random.randint(75, 98)
    
    # Simulate top 3 predictions
    top_predictions = []
    remaining_classes = [c for c in CLOTHING_CLASSES if c != predicted_class]
    for i in range(2):
        class_name = random.choice(remaining_classes)
        remaining_classes.remove(class_name)
        top_predictions.append({
            'class': class_name,
            'confidence': random.randint(40, confidence-10)
        })
    
    # Add the main prediction at the top
    top_predictions.insert(0, {
        'class': predicted_class,
        'confidence': confidence
    })
    
    return predicted_class, confidence, top_predictions

def get_similar_items(clothing_class, num_items=8):
    """
    Placeholder for getting similar items from your database
    Replace this with your actual similarity search
    """
    # Mock similar items - replace with your actual database query
    similar_items = []
    
    for i in range(num_items):
        similar_items.append({
            'id': i + 1,
            'name': f'Similar {clothing_class.title()} #{i+1}',
            'description': f'Stylish {clothing_class} with similar design and pattern',
            'image_url': f'/static/placeholder-{clothing_class}.jpg',  # Replace with actual images
            'similarity': random.randint(80, 95),
            'price': random.randint(20, 200) if random.choice([True, False]) else None
        })
    
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
            
            # Get predictions from your models
            predicted_class, confidence, top_predictions = predict_clothing_class(filepath)
            
            # Get similar items
            similar_items = get_similar_items(predicted_class)
            
            # Prepare data for template
            uploaded_image_url = f'/static/uploads/{filename}'
            
            return render_template('results.html',
                                 title=f'Similar {predicted_class.title()}s Found',
                                 predicted_class=predicted_class,
                                 confidence=confidence,
                                 top_predictions=top_predictions,
                                 similar_items=similar_items,
                                 uploaded_image_url=uploaded_image_url)
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