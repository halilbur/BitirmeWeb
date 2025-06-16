#!/usr/bin/env python3
# Test script to verify the Flask app can start without import errors

import sys
import os

# Add the src directory to the Python path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    print("Testing imports...")
    
    # Test basic imports
    from main import app
    print("✓ Successfully imported Flask app")
    
    # Test model imports
    from models.resnet18.predict import predict_image
    print("✓ Successfully imported ResNet18 predict function")
    
    from models.tripleNetAndXgboost.predict import predict_with_triplenet_xgboost
    print("✓ Successfully imported TripletNet+XGBoost predict function")
    
    # Test Flask app configuration
    print(f"✓ Static folder path: {app.static_folder}")
    print(f"✓ Upload folder path: {app.config['UPLOAD_FOLDER']}")
    
    # Test if required directories exist
    if os.path.exists(app.static_folder):
        print("✓ Static folder exists")
    else:
        print("⚠ Static folder does not exist")
    
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        print("✓ Upload folder exists")
    else:
        print("⚠ Upload folder does not exist (will be created automatically)")
    
    print("\n🎉 All tests passed! The app should work on Render.com")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
