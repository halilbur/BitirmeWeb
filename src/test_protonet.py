"""
Test script for ProtoNet integration in Flask app
"""
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.protoNet.predict import predict_with_protonet
from PIL import Image
import numpy as np

def test_protonet():
    """Test ProtoNet prediction functionality"""
    
    # Get paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    feature_extractor_path = os.path.join(current_dir, 'models', 'protoNet', 'protonet_feature_extractor.pth')
    classifier_data_path = os.path.join(current_dir, 'models', 'protoNet', 'protonet_classifier_data.pth')
    
    # Create a test image
    test_image_path = os.path.join(current_dir, '..', 'static', 'uploads', 'test_protonet.jpg')
    
    # Ensure upload directory exists
    upload_dir = os.path.dirname(test_image_path)
    os.makedirs(upload_dir, exist_ok=True)
    
    # Create a dummy test image
    try:
        dummy_img = Image.new('RGB', (224, 224), color='red')
        dummy_img.save(test_image_path)
        print(f"✓ Created test image at {test_image_path}")
    except Exception as e:
        print(f"✗ Error creating test image: {e}")
        return False
    
    # Check if model files exist
    if not os.path.exists(feature_extractor_path):
        print(f"✗ Feature extractor not found at {feature_extractor_path}")
        return False
    
    if not os.path.exists(classifier_data_path):
        print(f"✗ Classifier data not found at {classifier_data_path}")
        return False
    
    print(f"✓ Found feature extractor at {feature_extractor_path}")
    print(f"✓ Found classifier data at {classifier_data_path}")
    
    # Test prediction
    try:
        predicted_class, confidence, top_predictions = predict_with_protonet(
            test_image_path,
            feature_extractor_path,
            classifier_data_path,
            device='cpu'
        )
        
        print(f"✓ ProtoNet prediction successful!")
        print(f"  - Predicted Class: {predicted_class}")
        print(f"  - Confidence: {confidence}%")
        print(f"  - Top 3 Predictions:")
        for i, pred in enumerate(top_predictions, 1):
            print(f"    {i}. {pred['class']} ({pred['confidence']}%)")
        
        # Clean up test image
        try:
            os.remove(test_image_path)
            print(f"✓ Cleaned up test image")
        except:
            pass
        
        return True
        
    except Exception as e:
        print(f"✗ ProtoNet prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("Testing ProtoNet integration...")
    print("=" * 50)
    
    success = test_protonet()
    
    print("=" * 50)
    if success:
        print("✓ ProtoNet integration test PASSED!")
        print("  Your Flask app should now work with ProtoNet predictions.")
    else:
        print("✗ ProtoNet integration test FAILED!")
        print("  Please check the error messages above.")
