from http.server import BaseHTTPRequestHandler
import json
import os
import base64
from tensorflow.keras.models import load_model
import numpy as np
from utils import preprocess_image, preprocess_for_ethnicity, preprocess_for_emotion

# Model paths - adjust based on your deployment
MODEL_BASE_PATH = os.path.join(os.path.dirname(__file__), '..', 'labeling_age_gender')

# Global model cache
_age_gender_model = None
_ethnicity_model = None
_emotion_model = None

def get_age_gender_model():
    global _age_gender_model
    if _age_gender_model is None:
        age_gender_model_path = os.path.join(MODEL_BASE_PATH, 'age_gender_pseudolabel.h5')
        if os.path.exists(age_gender_model_path):
            _age_gender_model = load_model(age_gender_model_path, compile=False)
            _age_gender_model.compile(
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
    return _age_gender_model

def get_ethnicity_model():
    global _ethnicity_model
    if _ethnicity_model is None:
        ethnicity_model_paths = [
            os.path.join(MODEL_BASE_PATH, 'Ethnicity_lebelling.h5'),
            os.path.join(MODEL_BASE_PATH, 'ethnicity_labelling.h5'),
        ]
        for eth_path in ethnicity_model_paths:
            if os.path.exists(eth_path):
                try:
                    _ethnicity_model = load_model(eth_path, compile=False)
                    break
                except:
                    continue
    return _ethnicity_model

def get_emotion_model():
    global _emotion_model
    if _emotion_model is None:
        emotion_model_paths = [
            os.path.join(MODEL_BASE_PATH, 'emotion_model.h5'),
        ]
        for emo_path in emotion_model_paths:
            if os.path.exists(emo_path):
                try:
                    _emotion_model = load_model(emo_path, compile=False)
                    break
                except:
                    continue
    return _emotion_model

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            # Decode base64 image
            image_str = data.get('image', '')
            if ',' in image_str:
                image_str = image_str.split(',')[1]
            image_data = base64.b64decode(image_str)
            
            # Get age/gender model
            age_gender_model = get_age_gender_model()
            if age_gender_model is None:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'success': False,
                    'error': 'Model file not found'
                }).encode('utf-8'))
                return
            
            # Preprocess image
            img_input = preprocess_image(image_data)
            if img_input is None:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'success': False,
                    'error': 'Failed to preprocess image'
                }).encode('utf-8'))
                return
            
            # Predict age and gender
            gender_pred, age_pred = age_gender_model.predict(img_input, verbose=0)
            
            # Process results
            gender_prob = float(gender_pred[0][0])
            gender_label = "Male" if gender_prob < 0.5 else "Female"
            gender_confidence = max(gender_prob, 1 - gender_prob)
            age_value = int(round(age_pred[0][0]))
            
            # Try to load optional models
            ethnicity_result = None
            emotion_result = None
            
            # Ethnicity model
            ethnicity_model = get_ethnicity_model()
            if ethnicity_model is not None:
                try:
                    img_eth = preprocess_for_ethnicity(image_data)
                    if img_eth is not None:
                        probs = ethnicity_model.predict(img_eth, verbose=0)
                        probs = np.squeeze(probs)
                        if probs.ndim == 0:
                            probs = np.array([1.0 - float(probs), float(probs)])
                        class_index = int(np.argmax(probs))
                        confidence = float(np.max(probs))
                        ethnicity_result = {
                            'label': f"Class_{class_index}",
                            'confidence': confidence
                        }
                except Exception as e:
                    print(f"Ethnicity prediction error: {e}")
            
            # Emotion model
            emotion_model = get_emotion_model()
            emotion_classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
            if emotion_model is not None:
                try:
                    img_emo = preprocess_for_emotion(image_data)
                    if img_emo is not None:
                        probs = emotion_model.predict(img_emo, verbose=0)
                        probs = np.squeeze(probs)
                        if probs.ndim == 0:
                            probs = np.array([1.0 - float(probs), float(probs)])
                        class_index = int(np.argmax(probs))
                        confidence = float(np.max(probs))
                        label = emotion_classes[class_index] if class_index < len(emotion_classes) else f"Emotion_{class_index}"
                        emotion_result = {
                            'label': label,
                            'confidence': confidence
                        }
                except Exception as e:
                    print(f"Emotion prediction error: {e}")
            
            # Combine results
            predictions = {
                'age': age_value,
                'gender': gender_label,
                'gender_confidence': gender_confidence,
                'ethnicity': ethnicity_result,
                'emotion': emotion_result
            }
            
            # Draw labels on image
            from utils import draw_labels_on_image
            labeled_image = draw_labels_on_image(image_data, predictions)
            
            response = {
                'success': True,
                'age': age_value,
                'gender': gender_label,
                'gender_confidence': gender_confidence,
                'ethnicity': ethnicity_result,
                'emotion': emotion_result,
                'labeled_image': labeled_image
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                'success': False,
                'error': f"Prediction error: {str(e)}"
            }).encode('utf-8'))
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

