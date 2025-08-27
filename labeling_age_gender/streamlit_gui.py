import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import os
import tempfile
import zipfile
import io
from tensorflow.keras.models import load_model
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
import glob
from pathlib import Path
from typing import Optional, Tuple
from config import NATIONALITY_MODEL_PATHS

# Page configuration
st.set_page_config(
    page_title="🤖 Automatic Facial Image Labelling System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
    }
    .upload-area {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #fafafa;
        transition: all 0.3s ease;
    }
    .upload-area:hover {
        border-color: #1f77b4;
        background-color: #f0f8ff;
    }
    .folder-upload {
        background-color: #e8f5e8;
        border-color: #28a745;
    }
    .file-upload {
        background-color: #e8f4fd;
        border-color: #1f77b4;
    }
    .stButton > button {
        border-radius: 20px;
        font-weight: bold;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .progress-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .upload-area {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #fafafa;
        transition: all 0.3s ease;
    }
    .upload-area:hover {
        border-color: #1f77b4;
        background-color: #f0f8ff;
    }
    .folder-upload {
        background-color: #e8f5e8;
        border-color: #28a745;
    }
    .file-upload {
        background-color: #e8f4fd;
        border-color: #1f77b4;
    }
    .stButton > button {
        border-radius: 20px;
        font-weight: bold;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .progress-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def upload_folder():
    """Create a folder upload interface"""
    st.markdown("### 📁 Upload Folder")
    
    # Option 1: Zip file upload
    st.markdown("**Option 1: Upload as ZIP file**")
    zip_file = st.file_uploader(
        "Upload a ZIP file containing your images",
        type=['zip'],
        help="Create a ZIP file with your images and upload it here"
    )
    
    if zip_file:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Find all image files
                image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')
                image_files = []
                for ext in image_extensions:
                    image_files.extend(glob.glob(os.path.join(temp_dir, '**', ext), recursive=True))
                
                if image_files:
                    st.success(f"✅ Found {len(image_files)} images in ZIP file")
                    return image_files
                else:
                    st.warning("⚠️ No image files found in ZIP file")
                    return []
        except Exception as e:
            st.error(f"❌ Error processing ZIP file: {str(e)}")
            return []
    
    # Option 2: Manual folder path (for advanced users)
    st.markdown("**Option 2: Enter folder path manually**")
    folder_path = st.text_input(
        "Enter the full path to your image folder:",
        placeholder="C:\\Users\\YourName\\Pictures\\FacialImages",
        help="Enter the complete path to the folder containing your images"
    )
    
    if folder_path and os.path.exists(folder_path):
        # Find all image files
        image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        
        if image_files:
            st.success(f"✅ Found {len(image_files)} images in folder")
            return image_files
        else:
            st.warning("⚠️ No image files found in the specified folder")
            return []
    elif folder_path:
        st.error("❌ Folder path does not exist")
    
    return []


class FacialLabellingSystem:
    def __init__(self):
        self.age_gender_model = None
        self.nationality_model = None
        self.nationality_model_path: Optional[str] = None
        self.models_loaded = False
        self.dataset_info = {}
        self.labeled_data = []
        self.unlabeled_data = []
        self.prediction_results = []
        
    def load_models(self):
        """Load the pre-trained models"""
        try:
            with st.spinner("🔄 Loading models..."):
                model_path = "age_gender_pseudolabel.h5"
                if os.path.exists(model_path):
                    self.age_gender_model = load_model(model_path, compile=False)
                    self.age_gender_model.compile(
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
                    # Try load nationality model (optional)
                    self.nationality_model = self._try_load_nationality_model()
                    self.models_loaded = True
                    return True
                else:
                    st.error("❌ Model file 'age_gender_pseudolabel.h5' not found!")
                    return False
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            return False

    def _try_load_nationality_model(self) -> Optional[object]:
        # First check configured candidates
        candidates = [os.path.abspath(p) for p in NATIONALITY_MODEL_PATHS]
        # Add case-insensitive scans for root and models/
        for root_dir in [os.getcwd(), os.path.join(os.getcwd(), 'models')]:
            try:
                for fname in os.listdir(root_dir):
                    if fname.lower().endswith('.h5') and 'ethnicity' in fname.lower():
                        candidates.append(os.path.join(root_dir, fname))
            except Exception:
                pass
        # Deduplicate preserving order
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)
        for candidate_path in unique_candidates:
            try:
                if os.path.exists(candidate_path):
                    model = load_model(candidate_path, compile=False)
                    self.nationality_model_path = candidate_path
                    return model
            except Exception as e:
                st.warning(f"Failed loading ethnicity model at {os.path.basename(candidate_path)}: {e}")
                continue
        return None

    def load_nationality_model_from_path(self, model_path: str) -> bool:
        try:
            abs_path = os.path.abspath(model_path)
            if not os.path.exists(abs_path):
                st.error(f"❌ Ethnicity model not found at: {abs_path}")
                return False
            self.nationality_model = load_model(abs_path, compile=False)
            self.nationality_model_path = abs_path
            st.success(f"🌍 Ethnicity model loaded: {os.path.basename(abs_path)}")
            return True
        except Exception as e:
            st.error(f"❌ Failed to load ethnicity model: {e}")
            return False

    def load_nationality_model_from_upload(self, uploaded_file) -> bool:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                save_path = os.path.join(tmpdir, uploaded_file.name)
                with open(save_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                # Load directly from temp path
                self.nationality_model = load_model(save_path, compile=False)
                self.nationality_model_path = uploaded_file.name
                st.success(f"🌍 Ethnicity model loaded from upload: {uploaded_file.name}")
                return True
        except Exception as e:
            st.error(f"❌ Failed to load uploaded ethnicity model: {e}")
            return False
    
    def preprocess_image(self, image, target_size=(48, 48)):
        """Preprocess image for model input"""
        try:
            # Convert PIL image to numpy array
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image
            
            # Resize to model input size
            img_resized = cv2.resize(img_array, target_size)
            
            # Normalize to [0,1]
            img_normalized = img_resized.astype('float32') / 255.0
            
            # Add batch dimension
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            return img_batch
        except Exception as e:
            st.error(f"Error preprocessing image: {str(e)}")
            return None

    def _preprocess_for_model(self, image, model, fallback_size=(224, 224)):
        """Preprocess a PIL/numpy image for an arbitrary Keras model.

        - If model has 4D input shape with known HxW, use that size.
        - Otherwise, use fallback_size (defaults to 224x224 for common CNNs).
        """
        try:
            input_shape = None
            try:
                input_shape = getattr(model, 'input_shape', None)
            except Exception:
                input_shape = None

            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image

            # Determine target size
            target_size = fallback_size
            if input_shape is not None:
                # input_shape like (None, H, W, C) for TensorFlow
                if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 2:
                    ishape = input_shape[0] if isinstance(input_shape[0], (list, tuple)) else input_shape
                    if len(ishape) == 4:
                        _, h, w, _ = ishape
                        if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                            target_size = (w, h)

            img_resized = cv2.resize(img_array, target_size)
            img_normalized = img_resized.astype('float32') / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)
            return img_batch
        except Exception as e:
            st.error(f"Error preprocessing for model: {str(e)}")
            return None
    
    def predict_age_gender(self, image):
        """Predict age and gender from image"""
        try:
            # Preprocess image
            img_input = self.preprocess_image(image)
            if img_input is None:
                return None
            
            # Make prediction
            gender_pred, age_pred = self.age_gender_model.predict(img_input, verbose=0)
            
            # Process results
            gender_prob = gender_pred[0][0]
            gender_label = "Male" if gender_prob < 0.5 else "Female"
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
            st.error(f"Error during prediction: {str(e)}")
            return None

    def predict_nationality(self, image) -> Optional[Tuple[str, float, np.ndarray]]:
        """Predict nationality/ethnicity if model is available.

        Returns (label, confidence, raw_probs) or None if model not loaded.
        """
        if self.nationality_model is None:
            return None
        try:
            # Try to adapt input size for the ethnicity model (common CNNs use 224x224)
            img_input = self._preprocess_for_model(image, self.nationality_model, fallback_size=(224, 224))
            if img_input is None:
                return None
            probs = self.nationality_model.predict(img_input, verbose=0)
            # Handle different output shapes
            if isinstance(probs, list) or isinstance(probs, tuple):
                probs = probs[0]
            probs = np.squeeze(probs)
            if probs.ndim == 0:
                # Binary case; map to two classes
                probs = np.array([1.0 - float(probs), float(probs)])
            class_index = int(np.argmax(probs))
            confidence = float(np.max(probs))
            label = f"Class_{class_index}"
            # Optional: human-readable labels via env or sidecar file in future
            return label, confidence, probs
        except Exception as e:
            st.warning(f"Nationality prediction skipped: {str(e)}")
            return None
    
    def process_dataset(self, labeled_files, unlabeled_files):
        """Process the entire dataset"""
        all_results = []
        
        # Process labeled data
        if labeled_files:
            st.info(f"📊 Processing {len(labeled_files)} labeled images...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file_path in enumerate(labeled_files):
                status_text.text(f"Processing labeled image {i+1}/{len(labeled_files)}: {file_path.name}")
                
                try:
                    # Load image
                    image = Image.open(file_path)
                    
                    # Get filename info (assuming format: age_gender_*.jpg)
                    filename = file_path.name
                    parts = filename.split('_')
                    
                    if len(parts) >= 2:
                        try:
                            true_age = int(parts[0])
                            true_gender = int(parts[1])
                            gender_label = "Male" if true_gender == 0 else "Female"
                            
                            # Make prediction
                            prediction = self.predict_age_gender(image)
                            
                            if prediction:
                                nat = self.predict_nationality(image)
                                result = {
                                    'filename': filename,
                                    'image_type': 'Labeled',
                                    'true_age': true_age,
                                    'true_gender': gender_label,
                                    'predicted_age': prediction['age'],
                                    'predicted_gender': prediction['gender'],
                                    'predicted_nationality': nat[0] if nat else None,
                                    'nationality_confidence': nat[1] if nat else None,
                                    'age_error': abs(true_age - prediction['age']),
                                    'gender_correct': (true_gender == 0 and prediction['gender'] == 'Male') or 
                                                    (true_gender == 1 and prediction['gender'] == 'Female'),
                                    'gender_confidence': prediction['gender_confidence'],
                                    'raw_gender_prob': prediction['gender_probability']
                                }
                                all_results.append(result)
                        except ValueError:
                            st.warning(f"Could not parse filename: {filename}")
                    else:
                        st.warning(f"Invalid filename format: {filename}")
                        
                except Exception as e:
                    st.error(f"Error processing {file_path}: {str(e)}")
                    continue
                
                # Update progress
                progress_bar.progress((i + 1) / len(labeled_files))
            
            progress_bar.empty()
            status_text.empty()
        
        # Process unlabeled data
        if unlabeled_files:
            st.info(f"🔍 Processing {len(unlabeled_files)} unlabeled images...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file_path in enumerate(unlabeled_files):
                status_text.text(f"Processing unlabeled image {i+1}/{len(unlabeled_files)}: {file_path.name}")
                
                try:
                    # Load image
                    image = Image.open(file_path)
                    
                    # Make prediction
                    prediction = self.predict_age_gender(image)
                    
                    if prediction:
                        nat = self.predict_nationality(image)
                        result = {
                            'filename': file_path.name,
                            'image_type': 'Unlabeled',
                            'true_age': None,
                            'true_gender': None,
                            'predicted_age': prediction['age'],
                            'predicted_gender': prediction['gender'],
                            'predicted_nationality': nat[0] if nat else None,
                            'nationality_confidence': nat[1] if nat else None,
                            'age_error': None,
                            'gender_correct': None,
                            'gender_confidence': prediction['gender_confidence'],
                            'raw_gender_prob': prediction['gender_probability']
                        }
                        all_results.append(result)
                        
                except Exception as e:
                    st.error(f"Error processing {file_path}: {str(e)}")
                    continue
                
                # Update progress
                progress_bar.progress((i + 1) / len(unlabeled_files))
            
            progress_bar.empty()
            status_text.empty()
        
        return all_results

def upload_folder():
    """Create a folder upload interface"""
    st.markdown("### 📁 Upload Folder")
    
    # Option 1: Zip file upload
    st.markdown("**Option 1: Upload as ZIP file**")
    zip_file = st.file_uploader(
        "Upload a ZIP file containing your images",
        type=['zip'],
        help="Create a ZIP file with your images and upload it here"
    )
    
    if zip_file:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Find all image files
                image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')
                image_files = []
                for ext in image_extensions:
                    image_files.extend(glob.glob(os.path.join(temp_dir, '**', ext), recursive=True))
                
                if image_files:
                    st.success(f"✅ Found {len(image_files)} images in ZIP file")
                    return image_files
                else:
                    st.warning("⚠️ No image files found in ZIP file")
                    return []
        except Exception as e:
            st.error(f"❌ Error processing ZIP file: {str(e)}")
            return []
    
    # Option 2: Manual folder path (for advanced users)
    st.markdown("**Option 2: Enter folder path manually**")
    folder_path = st.text_input(
        "Enter the full path to your image folder:",
        placeholder="C:\\Users\\YourName\\Pictures\\FacialImages",
        help="Enter the complete path to the folder containing your images"
    )
    
    if folder_path and os.path.exists(folder_path):
        # Find all image files
        image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        
        if image_files:
            st.success(f"✅ Found {len(image_files)} images in folder")
            return image_files
        else:
            st.warning("⚠️ No image files found in the specified folder")
            return []
    elif folder_path:
        st.error("❌ Folder path does not exist")
    
    return []

def show_load_dataset_page(facial_system):
    """Display the dataset loading page"""
    st.markdown('<h2 class="sub-header">📁 Load Dataset</h2>', unsafe_allow_html=True)
    
    if not facial_system.models_loaded:
        st.warning("⚠️ Please load the models first from the Home page.")
        return
    
    # Dataset upload section
    st.markdown("### 📤 Upload Your Dataset")
    
    # Upload method selection
    upload_method = st.radio(
        "Choose upload method:",
        ["📁 Upload Folder (Recommended)", "📄 Upload Individual Files"],
        help="Select how you want to upload your images"
    )
    
    if upload_method == "📁 Upload Folder (Recommended)":
        st.markdown("""
        **Benefits of folder upload:**
        - Upload entire datasets at once
        - Faster than selecting individual files
        - Better for large collections
        - Maintains folder structure
        """)
        
        # Folder upload interface
        uploaded_files = upload_folder()
        
        if uploaded_files:
            # Separate labeled and unlabeled images
            labeled_files = []
            unlabeled_files = []
            
            for file_path in uploaded_files:
                filename = os.path.basename(file_path)
                parts = filename.split('_')
                
                # Check if filename follows the labeled format (age_gender_*.jpg)
                if len(parts) >= 2:
                    try:
                        age = int(parts[0])
                        gender = int(parts[1])
                        if 0 <= age <= 100 and gender in [0, 1]:
                            labeled_files.append(file_path)
                            continue
                    except ValueError:
                        pass
                
                # If not labeled, consider it unlabeled
                unlabeled_files.append(file_path)
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Labeled Images", len(labeled_files))
            with col2:
                st.metric("Unlabeled Images", len(unlabeled_files))
            
            # Store dataset info
            facial_system.dataset_info = {
                'labeled_count': len(labeled_files),
                'unlabeled_count': len(unlabeled_files),
                'total_count': len(uploaded_files),
                'labeled_files': labeled_files,
                'unlabeled_files': unlabeled_files
            }
            
            # Show sample images
            if labeled_files or unlabeled_files:
                st.markdown("### 🖼️ Sample Images")
                
                # Show labeled samples
                if labeled_files:
                    st.markdown("**Labeled Samples:**")
                    sample_labeled = labeled_files[:min(5, len(labeled_files))]
                    cols = st.columns(len(sample_labeled))
                    for i, (col, file_path) in enumerate(zip(cols, sample_labeled)):
                        with col:
                            try:
                                image = Image.open(file_path)
                                st.image(image, caption=f"Labeled {i+1}", use_column_width=True)
                            except Exception as e:
                                st.error(f"Error loading image: {str(e)}")
                
                # Show unlabeled samples
                if unlabeled_files:
                    st.markdown("**Unlabeled Samples:**")
                    sample_unlabeled = unlabeled_files[:min(5, len(unlabeled_files))]
                    cols = st.columns(len(sample_unlabeled))
                    for i, (col, file_path) in enumerate(zip(cols, sample_unlabeled)):
                        with col:
                            try:
                                image = Image.open(file_path)
                                st.image(image, caption=f"Unlabeled {i+1}", use_column_width=True)
                            except Exception as e:
                                st.error(f"Error loading image: {str(e)}")
            
            st.success("✅ Dataset loaded successfully! You can now proceed to process the images.")
    
    else:
        # Individual file upload (original method)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏷️ Labeled Images")
            st.markdown("""
            Upload images that already have age and gender labels.
            
            **Expected filename format:** `age_gender_*.jpg`
            - `age`: Integer age (0-100)
            - `gender`: 0 for Male, 1 for Female
            - Example: `25_0_person1.jpg` (25-year-old male)
            """)
            
            labeled_files = st.file_uploader(
                "Choose labeled image files",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                accept_multiple_files=True,
                key="labeled_uploader"
            )
        
        with col2:
            st.markdown("#### 🔍 Unlabeled Images")
            st.markdown("""
            Upload images without age and gender labels.
            
            The system will automatically predict:
            - Age (0-100 years)
            - Gender (Male/Female)
            - Confidence scores
            """)
            
            unlabeled_files = st.file_uploader(
                "Choose unlabeled image files",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                accept_multiple_files=True,
                key="unlabeled_uploader"
            )
        
        # Dataset summary for individual files
        if labeled_files or unlabeled_files:
            st.markdown("---")
            st.markdown("### 📊 Dataset Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Labeled Images", len(labeled_files) if labeled_files else 0)
            
            with col2:
                st.metric("Unlabeled Images", len(unlabeled_files) if unlabeled_files else 0)
            
            with col3:
                total = (len(labeled_files) if labeled_files else 0) + (len(unlabeled_files) if unlabeled_files else 0)
                st.metric("Total Images", total)
            
            with col4:
                if labeled_files:
                    labeled_percentage = (len(labeled_files) / total) * 100
                    st.metric("Labeled %", f"{labeled_percentage:.1f}%")
            
            # Store dataset info
            facial_system.dataset_info = {
                'labeled_count': len(labeled_files) if labeled_files else 0,
                'unlabeled_count': len(unlabeled_files) if unlabeled_files else 0,
                'total_count': total,
                'labeled_files': labeled_files,
                'unlabeled_files': unlabeled_files
            }
            
            # Show sample images
            if labeled_files or unlabeled_files:
                st.markdown("### 🖼️ Sample Images")
                
                # Show labeled samples
                if labeled_files:
                    st.markdown("**Labeled Samples:**")
                    cols = st.columns(min(5, len(labeled_files)))
                    for i, col in enumerate(cols):
                        if i < len(labeled_files):
                            with col:
                                st.image(labeled_files[i], caption=f"Labeled {i+1}", use_column_width=True)
                
                # Show unlabeled samples
                if unlabeled_files:
                    st.markdown("**Unlabeled Samples:**")
                    cols = st.columns(min(5, len(unlabeled_files)))
                    for i, col in enumerate(cols):
                        if i < len(unlabeled_files):
                            with col:
                                st.image(unlabeled_files[i], caption=f"Unlabeled {i+1}", use_column_width=True)
            
            st.success("✅ Dataset loaded successfully! You can now proceed to process the images.")

def show_process_page(facial_system):
    """Display the image processing page"""
    st.markdown('<h2 class="sub-header">🔍 Process Images</h2>', unsafe_allow_html=True)
    
    if not facial_system.models_loaded:
        st.warning("⚠️ Please load the models first from the Home page.")
        return
    
    if not facial_system.dataset_info or facial_system.dataset_info['total_count'] == 0:
        st.warning("⚠️ Please load a dataset first from the 'Load Dataset' page.")
        return
    
    st.markdown("### 🚀 Start Processing")
    
    # Display dataset info in a nice format
    st.markdown("#### 📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Images", facial_system.dataset_info['total_count'])
    with col2:
        st.metric("Labeled Images", facial_system.dataset_info['labeled_count'])
    with col3:
        st.metric("Unlabeled Images", facial_system.dataset_info['unlabeled_count'])
    with col4:
        if facial_system.dataset_info['labeled_count'] > 0:
            labeled_percentage = (facial_system.dataset_info['labeled_count'] / facial_system.dataset_info['total_count']) * 100
            st.metric("Labeled %", f"{labeled_percentage:.1f}%")
    
    # Processing options
    st.markdown("#### ⚙️ Processing Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        batch_size = st.slider(
            "Batch Size",
            min_value=1,
            max_value=50,
            value=10,
            help="Number of images to process at once. Lower values use less memory."
        )
    
    with col2:
        show_progress = st.checkbox(
            "Show detailed progress",
            value=True,
            help="Display real-time progress for each image"
        )
    
    # Process button
    if st.button("🚀 Start Processing", type="primary", use_container_width=True):
        with st.spinner("🔄 Processing images..."):
            # Process the dataset
            results = facial_system.process_dataset(
                facial_system.dataset_info.get('labeled_files', []),
                facial_system.dataset_info.get('unlabeled_files', [])
            )
            
            if results:
                facial_system.prediction_results = results
                st.success(f"✅ Processing complete! Processed {len(results)} images.")
                
                # Show processing summary
                show_processing_summary(results)
                
                # Store results in session state
                st.session_state.results = results
                st.rerun()
            else:
                st.error("❌ No results generated. Please check your dataset and try again.")

def show_processing_summary(results):
    """Display processing summary"""
    st.markdown("### 📊 Processing Summary")
    
    if not results:
        return
    
    # Convert to dataframe
    df = pd.DataFrame(results)
    
    # Calculate metrics
    labeled_results = df[df['image_type'] == 'Labeled']
    unlabeled_results = df[df['image_type'] == 'Unlabeled']
    
    # Display metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Processed", len(results))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if not labeled_results.empty:
            age_mae = labeled_results['age_error'].mean()
            st.metric("Age MAE (Labeled)", f"{age_mae:.1f}")
        else:
            st.metric("Age MAE (Labeled)", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if not labeled_results.empty:
            gender_accuracy = labeled_results['gender_correct'].mean() * 100
            st.metric("Gender Accuracy", f"{gender_accuracy:.1f}%")
        else:
            st.metric("Gender Accuracy", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_confidence = df['gender_confidence'].mean() * 100
        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Ethnicity quick view (if available)
    if 'predicted_nationality' in df.columns and df['predicted_nationality'].notna().any():
        st.markdown("### 🌍 Ethnicity Overview")
        eth_counts = df['predicted_nationality'].fillna('Unknown').value_counts()
        fig_eth = px.bar(
            x=eth_counts.index,
            y=eth_counts.values,
            labels={'x': 'Ethnicity', 'y': 'Count'},
            title="Predicted Ethnicity Distribution"
        )
        st.plotly_chart(fig_eth, use_container_width=True)
    
    # Show detailed results table
    st.markdown("### 📋 Detailed Results")
    
    # Add search and filter options
    col1, col2 = st.columns(2)
    
    with col1:
        search_term = st.text_input("🔍 Search by filename:", placeholder="Enter filename to search...")
    
    with col2:
        filter_type = st.selectbox("📁 Filter by type:", ["All", "Labeled", "Unlabeled"])
    # Ethnicity filter
    eth_options = ["All"]
    if 'predicted_nationality' in df.columns and df['predicted_nationality'].notna().any():
        eth_options += sorted(df['predicted_nationality'].dropna().unique().tolist())
    ethnicity_filter = st.selectbox("🌍 Filter by ethnicity:", eth_options)
    
    # Filter data
    filtered_df = df.copy()
    if search_term:
        filtered_df = filtered_df[filtered_df['filename'].str.contains(search_term, case=False, na=False)]
    if filter_type != "All":
        filtered_df = filtered_df[filtered_df['image_type'] == filter_type]
    if 'predicted_nationality' in filtered_df.columns and ethnicity_filter != "All":
        filtered_df = filtered_df[filtered_df['predicted_nationality'] == ethnicity_filter]
    
    # Display filtered results
    st.dataframe(filtered_df, use_container_width=True)
    
    # Show summary statistics
    if not filtered_df.empty:
        st.markdown(f"**Showing {len(filtered_df)} of {len(df)} results**")

def show_results_page(facial_system):
    """Display the results page"""
    st.markdown('<h2 class="sub-header">📊 Results Analysis</h2>', unsafe_allow_html=True)
    
    if not facial_system.prediction_results:
        st.warning("⚠️ No results available. Please process images first.")
        return
    
    results = facial_system.prediction_results
    df = pd.DataFrame(results)
    
    # Results overview
    st.markdown("### 📈 Results Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        st.markdown("**Age Distribution**")
        fig_age = px.histogram(
            df, 
            x='predicted_age', 
            color='image_type',
            title="Predicted Age Distribution",
            nbins=20,
            color_discrete_map={'Labeled': '#1f77b4', 'Unlabeled': '#ff7f0e'}
        )
        fig_age.update_layout(height=400)
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        # Gender distribution
        st.markdown("**Gender Distribution**")
        gender_counts = df['predicted_gender'].value_counts()
        fig_gender = px.pie(
            values=gender_counts.values,
            names=gender_counts.index,
            title="Predicted Gender Distribution",
            color_discrete_map={'Male': '#1f77b4', 'Female': '#ff7f0e'}
        )
        fig_gender.update_layout(height=400)
        st.plotly_chart(fig_gender, use_container_width=True)
    
    # Ethnicity distribution if available
    if 'predicted_nationality' in df.columns and df['predicted_nationality'].notna().any():
        st.markdown("### 🌍 Ethnicity Distribution")
        eth_counts = df['predicted_nationality'].fillna('Unknown').value_counts()
        col1, col2 = st.columns(2)
        with col1:
            fig_eth_bar = px.bar(
                x=eth_counts.index,
                y=eth_counts.values,
                labels={'x': 'Ethnicity', 'y': 'Count'},
                title="Ethnicity Counts"
            )
            st.plotly_chart(fig_eth_bar, use_container_width=True)
        with col2:
            fig_eth_pie = px.pie(values=eth_counts.values, names=eth_counts.index, title="Ethnicity Share")
            st.plotly_chart(fig_eth_pie, use_container_width=True)
    
    # Performance metrics for labeled data
    labeled_df = df[df['image_type'] == 'Labeled']
    if not labeled_df.empty:
        st.markdown("### 🎯 Performance Metrics (Labeled Data)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age_mae = labeled_df['age_error'].mean()
            st.metric("Mean Age Error", f"{age_mae:.2f} years")
        
        with col2:
            gender_accuracy = labeled_df['gender_correct'].mean() * 100
            st.metric("Gender Accuracy", f"{gender_accuracy:.1f}%")
        
        with col3:
            avg_confidence = labeled_df['gender_confidence'].mean() * 100
            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        
        # Age error analysis
        st.markdown("**Age Prediction Error Analysis**")
        fig_age_error = px.scatter(
            labeled_df,
            x='true_age',
            y='predicted_age',
            title="True vs Predicted Age",
            labels={'true_age': 'True Age', 'predicted_age': 'Predicted Age'},
            color='gender_confidence',
            color_continuous_scale='viridis'
        )
        fig_age_error.add_trace(go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode='lines',
            name='Perfect Prediction',
            line=dict(dash='dash', color='red')
        ))
        st.plotly_chart(fig_age_error, use_container_width=True)
    
    # Confidence analysis
    st.markdown("### 📊 Confidence Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confidence distribution
        fig_conf = px.histogram(
            df,
            x='gender_confidence',
            color='image_type',
            title="Gender Prediction Confidence Distribution",
            nbins=20,
            color_discrete_map={'Labeled': '#1f77b4', 'Unlabeled': '#ff7f0e'}
        )
        st.plotly_chart(fig_conf, use_container_width=True)
    
    with col2:
        # Confidence vs accuracy (for labeled data)
        if not labeled_df.empty:
            fig_conf_acc = px.scatter(
                labeled_df,
                x='gender_confidence',
                y='gender_correct',
                title="Confidence vs Accuracy",
                labels={'gender_confidence': 'Confidence', 'gender_correct': 'Correct Prediction'},
                color='true_age',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_conf_acc, use_container_width=True)

def show_download_page(facial_system):
    """Display the download page"""
    st.markdown('<h2 class="sub-header">📥 Download Results</h2>', unsafe_allow_html=True)
    
    if not facial_system.prediction_results:
        st.warning("⚠️ No results available. Please process images first.")
        return
    
    results = facial_system.prediction_results
    df = pd.DataFrame(results)
    
    st.markdown("### 📊 Export Options")
    
    # Data preview with search and filter
    st.markdown("**Data Preview:**")
    
    col1, col2 = st.columns(2)
    with col1:
        search_term = st.text_input("🔍 Search:", placeholder="Search by filename...")
    with col2:
        filter_type = st.selectbox("📁 Filter:", ["All", "Labeled", "Unlabeled"])
    # Ethnicity filter for export preview
    eth_options = ["All"]
    if 'predicted_nationality' in df.columns and df['predicted_nationality'].notna().any():
        eth_options += sorted(df['predicted_nationality'].dropna().unique().tolist())
    ethnicity_filter = st.selectbox("🌍 Ethnicity:", eth_options)
    
    # Filter data
    filtered_df = df.copy()
    if search_term:
        filtered_df = filtered_df[filtered_df['filename'].str.contains(search_term, case=False, na=False)]
    if filter_type != "All":
        filtered_df = filtered_df[filtered_df['image_type'] == filter_type]
    if 'predicted_nationality' in filtered_df.columns and ethnicity_filter != "All":
        filtered_df = filtered_df[filtered_df['predicted_nationality'] == ethnicity_filter]
    
    st.dataframe(filtered_df.head(10), use_container_width=True)
    st.markdown(f"**Showing {len(filtered_df)} of {len(df)} results**")
    
    # Export options
    st.markdown("### 📁 Export Formats")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📄 CSV Export")
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"facial_labelling_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        st.markdown("#### 📊 Excel Export")
        # Create Excel file in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            filtered_df.to_excel(writer, sheet_name='Results', index=False)
            
            # Create summary sheet
            summary_data = {
                'Metric': ['Total Images', 'Labeled Images', 'Unlabeled Images', 'Age MAE', 'Gender Accuracy', 'Ethnicity Classes'],
                'Value': [
                    len(filtered_df),
                    len(filtered_df[filtered_df['image_type'] == 'Labeled']),
                    len(filtered_df[filtered_df['image_type'] == 'Unlabeled']),
                    filtered_df[filtered_df['image_type'] == 'Labeled']['age_error'].mean() if len(filtered_df[filtered_df['image_type'] == 'Labeled']) > 0 else 'N/A',
                    f"{filtered_df[filtered_df['image_type'] == 'Labeled']['gender_correct'].mean() * 100:.1f}%" if len(filtered_df[filtered_df['image_type'] == 'Labeled']) > 0 else 'N/A',
                    len(filtered_df['predicted_nationality'].dropna().unique()) if 'predicted_nationality' in filtered_df.columns else 0
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        output.seek(0)
        st.download_button(
            label="📥 Download Excel",
            data=output.getvalue(),
            file_name=f"facial_labelling_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    # Dataset statistics
    st.markdown("### 📈 Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Images", len(filtered_df))
    
    with col2:
        labeled_count = len(filtered_df[filtered_df['image_type'] == 'Labeled'])
        st.metric("Labeled Images", labeled_count)
    
    with col3:
        unlabeled_count = len(filtered_df[filtered_df['image_type'] == 'Unlabeled'])
        st.metric("Unlabeled Images", unlabeled_count)
    
    with col4:
        if labeled_count > 0:
            accuracy = filtered_df[filtered_df['image_type'] == 'Labeled']['gender_correct'].mean() * 100
            st.metric("Overall Accuracy", f"{accuracy:.1f}%")
        else:
            st.metric("Overall Accuracy", "N/A")
    
    # Additional export options
    st.markdown("### 🔧 Advanced Export Options")
    
    # Filtered exports
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Export by Image Type:**")
        export_type = st.selectbox("Select image type:", ["All", "Labeled", "Unlabeled"], key="export_type")
        
        if export_type != "All":
            type_filtered_df = filtered_df[filtered_df['image_type'] == export_type]
        else:
            type_filtered_df = filtered_df
        
        if st.button(f"📥 Export {export_type} Images", use_container_width=True):
            csv_data = type_filtered_df.to_csv(index=False)
            st.download_button(
                label=f"Download {export_type} CSV",
                data=csv_data,
                file_name=f"facial_labelling_{export_type.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col2:
        st.markdown("**Export by Gender:**")
        gender_filter = st.selectbox("Select gender:", ["All", "Male", "Female"], key="gender_filter")
        
        if gender_filter != "All":
            gender_df = filtered_df[filtered_df['predicted_gender'] == gender_filter]
        else:
            gender_df = filtered_df
        
        if st.button(f"📥 Export {gender_filter} Results", use_container_width=True):
            csv_data = gender_df.to_csv(index=False)
            st.download_button(
                label=f"Download {gender_filter} CSV",
                data=csv_data,
                file_name=f"facial_labelling_{gender_filter.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

    # Export by ethnicity (if present)
    if 'predicted_nationality' in df.columns and df['predicted_nationality'].notna().any():
        st.markdown("### 🌍 Export by Ethnicity")
        eth_choices = sorted(df['predicted_nationality'].dropna().unique().tolist())
        chosen_eth = st.selectbox("Choose ethnicity:", eth_choices, key="export_ethnicity")
        eth_df = df[df['predicted_nationality'] == chosen_eth]
        if st.button(f"📥 Export {chosen_eth} Results", use_container_width=True):
            csv_data = eth_df.to_csv(index=False)
            st.download_button(
                label=f"Download {chosen_eth} CSV",
                data=csv_data,
                file_name=f"facial_labelling_ethnicity_{chosen_eth}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

def main():
    # Initialize the system
    if 'facial_system' not in st.session_state:
        st.session_state.facial_system = FacialLabellingSystem()
    
    facial_system = st.session_state.facial_system
    
    # Main header
    st.markdown('<h1 class="main-header">🤖 Automatic Facial Image Labelling System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📋 Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["🏠 Home", "📁 Load Dataset", "🔍 Process Images", "📊 Results", "📥 Download"]
        )
        
        st.markdown("---")
        st.markdown("## ℹ️ System Status")
        if facial_system.models_loaded:
            st.success("✅ Models Loaded")
            if facial_system.nationality_model is not None:
                loaded_name = os.path.basename(facial_system.nationality_model_path) if facial_system.nationality_model_path else ""
                st.info(f"🌍 Ethnicity model loaded {f'({loaded_name})' if loaded_name else ''}")
            else:
                st.warning("🌍 Ethnicity model not found (optional)")
                # Manual loader
                st.markdown("#### Load Ethnicity Model")
                eth_upload = st.file_uploader("Upload .h5 file", type=['h5'], key="eth_model_upload")
                eth_path = st.text_input("Or enter path to .h5", value="ethnicity_labelling.h5")
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Load from upload") and eth_upload is not None:
                        facial_system.load_nationality_model_from_upload(eth_upload)
                        st.rerun()
                with col_b:
                    if st.button("Load from path") and eth_path:
                        if facial_system.load_nationality_model_from_path(eth_path):
                            st.rerun()
        else:
            st.error("❌ Models Not Loaded")
        
        st.markdown("---")
        st.markdown("## 📊 Dataset Info")
        if facial_system.dataset_info:
            st.write(f"**Labeled Images:** {facial_system.dataset_info.get('labeled_count', 0)}")
            st.write(f"**Unlabeled Images:** {facial_system.dataset_info.get('unlabeled_count', 0)}")
            st.write(f"**Total Images:** {facial_system.dataset_info.get('total_count', 0)}")
    
    # Load models if not already loaded
    if not facial_system.models_loaded:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 🚀 Get Started")
        st.markdown("Click the button below to load the deep learning models and begin using the system.")
        if st.button("🚀 Load Models", use_container_width=True):
            if facial_system.load_models():
                st.success("✅ Models loaded successfully!")
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "📁 Load Dataset":
        show_load_dataset_page(facial_system)
    elif page == "🔍 Process Images":
        show_process_page(facial_system)
    elif page == "📊 Results":
        show_results_page(facial_system)
    elif page == "📥 Download":
        show_download_page(facial_system)

def show_home_page():
    """Display the home page"""
    st.markdown('<h2 class="sub-header">Welcome to the Automatic Facial Image Labelling System</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        This system uses advanced deep learning models to automatically label facial images with age and gender predictions.
        
        ### 🎯 Key Features:
        - **Semi-supervised Learning**: Combines small labeled datasets with large unlabeled datasets
        - **Age Prediction**: Regression-based age estimation (0-100 years)
        - **Gender Classification**: Binary classification (Male/Female) with confidence scores
        - **Batch Processing**: Handle large datasets efficiently
        - **Results Export**: Download labeled datasets in various formats
        
        ### 🔬 How It Works:
        1. **Load Dataset**: Upload labeled and unlabeled images
        2. **Process Images**: Apply pre-trained models for predictions
        3. **Generate Labels**: Create comprehensive labeled dataset
        4. **Export Results**: Download results for further analysis
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Performance Metrics:
        - **Age Prediction**: Mean Absolute Error (MAE)
        - **Gender Classification**: Accuracy and Confidence
        - **Model Validation**: Cross-validation results
        
        ### 🛠️ Technical Details:
        - **Input Size**: 48x48 pixels (RGB)
        - **Model Architecture**: CNN with dual outputs
        - **Preprocessing**: Resize, normalize, batch processing
        """)
    
    st.markdown("---")
    
    # Quick start section
    st.markdown('<h3 class="sub-header">🚀 Quick Start</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Step 1: Load Models**
        - Click 'Load Models' button
        - Ensure model file is present
        """)
    
    with col2:
        st.markdown("""
        **Step 2: Upload Dataset**
        - Go to 'Load Dataset' page
        - Upload labeled and unlabeled images
        """)
    
    with col3:
        st.markdown("""
        **Step 3: Process & Export**
        - Process images automatically
        - Download labeled results
        """)

if __name__ == "__main__":
    main()
