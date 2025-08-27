#!/usr/bin/env python3
"""
Demo script for the Automatic Facial Image Labelling System
This script demonstrates the core functionality without the GUI
"""

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path, target_size=(48, 48)):
    """Load and preprocess an image for the model"""
    try:
        # Load image
        image = Image.open(image_path)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Resize to target size
        img_resized = cv2.resize(img_array, target_size)
        
        # Normalize to [0,1]
        img_normalized = img_resized.astype('float32') / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch, image
        
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return None, None

def predict_age_gender(model, image_input):
    """Predict age and gender from preprocessed image"""
    try:
        # Make prediction
        gender_pred, age_pred = model.predict(image_input, verbose=0)
        
        # Process results
        gender_prob = gender_pred[0][0]
        gender_label = "Male" if gender_prob > 0.5 else "Female"
        gender_confidence = max(gender_prob, 1 - gender_prob)
        age_value = int(round(age_pred[0][0]))
        
        return {
            'age': age_value,
            'gender': gender_label,
            'gender_probability': gender_prob,
            'gender_confidence': gender_confidence,
            'raw_age': age_pred[0][0]
        }
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

def display_results(image, results):
    """Display image with prediction results"""
    plt.figure(figsize=(10, 6))
    
    # Display image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis('off')
    
    # Display results
    plt.subplot(1, 2, 2)
    plt.axis('off')
    
    result_text = f"""
    🔍 PREDICTION RESULTS
    
    👤 Gender: {results['gender']}
    📊 Confidence: {results['gender_confidence']:.2%}
    
    🎂 Age: {results['age']} years
    
    📈 Raw Values:
    Gender Probability: {results['gender_probability']:.4f}
    Age Prediction: {results['raw_age']:.2f}
    """
    
    plt.text(0.1, 0.5, result_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def main():
    print("🤖 Automatic Facial Image Labelling System - Demo")
    print("=" * 60)
    
    # Check if model exists
    model_path = "age_gender_pseudolabel.h5"
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file '{model_path}' not found!")
        print("Please ensure the model file is in the current directory.")
        return
    
    try:
        # Load model
        print("🔄 Loading model...")
        model = load_model(model_path, compile=False)
        model.compile(
            optimizer="adam",
            loss={
                "age_out": "mse",
                "sex_out": "binary_crossentropy"
            },
            metrics={
                "age_out": "mae",
                "sex_out": "accuracy"
            }
        )
        print("✅ Model loaded successfully!")
        
        # Demo with sample image if available
        sample_images = [f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if sample_images:
            print(f"\n📁 Found {len(sample_images)} image(s) in current directory:")
            for i, img in enumerate(sample_images):
                print(f"  {i+1}. {img}")
            
            # Use first image for demo
            demo_image = sample_images[0]
            print(f"\n🎯 Using '{demo_image}' for demonstration...")
            
            # Load and preprocess image
            print("🔄 Processing image...")
            img_input, original_image = load_and_preprocess_image(demo_image)
            
            if img_input is not None:
                # Make prediction
                print("🔍 Making prediction...")
                results = predict_age_gender(model, img_input)
                
                if results:
                    print("\n📊 Results:")
                    print(f"  Gender: {results['gender']} (Confidence: {results['gender_confidence']:.2%})")
                    print(f"  Age: {results['age']} years")
                    print(f"  Raw gender probability: {results['gender_probability']:.4f}")
                    print(f"  Raw age prediction: {results['raw_age']:.2f}")
                    
                    # Display results
                    print("\n🖼️  Displaying results...")
                    display_results(original_image, results)
                else:
                    print("❌ Prediction failed!")
            else:
                print("❌ Failed to load image!")
        else:
            print("\n📁 No image files found in current directory.")
            print("Please place some image files in this directory to test the system.")
            print("Supported formats: .jpg, .jpeg, .png, .bmp")
            
            # Show model summary instead
            print("\n📋 Model Summary:")
            model.summary()
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Please check your installation and model file.")

if __name__ == "__main__":
    main()
