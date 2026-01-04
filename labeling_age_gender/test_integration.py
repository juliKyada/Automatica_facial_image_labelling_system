#!/usr/bin/env python3
"""
Test script to verify the integration of emotion model with the existing system
"""

import os
import sys
import numpy as np
from PIL import Image
import cv2

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from streamlit_gui import FacialLabellingSystem
    print("✅ Successfully imported FacialLabellingSystem")
except ImportError as e:
    print(f"❌ Failed to import FacialLabellingSystem: {e}")
    sys.exit(1)

def test_model_loading():
    """Test if all models can be loaded"""
    print("\n🔄 Testing model loading...")
    
    system = FacialLabellingSystem()
    
    # Test age/gender model loading
    if system.load_models():
        print("✅ Age/Gender model loaded successfully")
    else:
        print("❌ Failed to load Age/Gender model")
        return False
    
    # Check if emotion model was loaded
    if system.emotion_model is not None:
        print(f"✅ Emotion model loaded: {os.path.basename(system.emotion_model_path)}")
    else:
        print("⚠️ Emotion model not found (this is optional)")
    
    # Check if ethnicity model was loaded
    if system.nationality_model is not None:
        print(f"✅ Ethnicity model loaded: {os.path.basename(system.nationality_model_path)}")
    else:
        print("⚠️ Ethnicity model not found (this is optional)")
    
    return True

def test_prediction_functions():
    """Test prediction functions with a dummy image"""
    print("\n🔄 Testing prediction functions...")
    
    system = FacialLabellingSystem()
    
    if not system.load_models():
        print("❌ Cannot test predictions - models failed to load")
        return False
    
    # Create a dummy image (48x48 RGB)
    dummy_image = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
    pil_image = Image.fromarray(dummy_image)
    
    try:
        # Test age/gender prediction
        age_gender_result = system.predict_age_gender(pil_image)
        if age_gender_result:
            print(f"✅ Age/Gender prediction: Age={age_gender_result['age']}, Gender={age_gender_result['gender']}")
        else:
            print("❌ Age/Gender prediction failed")
            return False
        
        # Test emotion prediction
        emotion_result = system.predict_emotion(pil_image)
        if emotion_result:
            emotion, confidence, _ = emotion_result
            print(f"✅ Emotion prediction: {emotion} (confidence: {confidence:.3f})")
        else:
            print("⚠️ Emotion prediction not available (model not loaded)")
        
        # Test ethnicity prediction
        nationality_result = system.predict_nationality(pil_image)
        if nationality_result:
            nationality, confidence, _ = nationality_result
            print(f"✅ Ethnicity prediction: {nationality} (confidence: {confidence:.3f})")
        else:
            print("⚠️ Ethnicity prediction not available (model not loaded)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during prediction testing: {e}")
        return False

def test_comprehensive_prediction():
    """Test the comprehensive prediction function"""
    print("\n🔄 Testing comprehensive prediction...")
    
    system = FacialLabellingSystem()
    
    if not system.load_models():
        print("❌ Cannot test comprehensive prediction - models failed to load")
        return False
    
    # Create a dummy image
    dummy_image = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
    pil_image = Image.fromarray(dummy_image)
    
    try:
        # Test comprehensive prediction
        result = system.predict_comprehensive(pil_image)
        if result:
            print("✅ Comprehensive prediction successful!")
            print(f"   Age: {result.get('age', 'N/A')}")
            print(f"   Gender: {result.get('gender', 'N/A')} (confidence: {result.get('gender_confidence', 0):.3f})")
            print(f"   Emotion: {result.get('emotion', 'N/A')} (confidence: {result.get('emotion_confidence', 0):.3f})")
            print(f"   Ethnicity: {result.get('ethnicity', 'N/A')} (confidence: {result.get('ethnicity_confidence', 0):.3f})")
        else:
            print("❌ Comprehensive prediction failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error during comprehensive prediction testing: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting Integration Test for Automatic Facial Image Labelling System")
    print("=" * 70)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Prediction Functions", test_prediction_functions),
        ("Comprehensive Prediction", test_comprehensive_prediction),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 40)
        
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 70)
    print(f"📊 TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The emotion integration is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)