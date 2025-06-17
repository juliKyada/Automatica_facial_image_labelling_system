import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.models import load_model
from sklearn.cluster import KMeans


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# Set page configuration
st.set_page_config(
    page_title="Age & Gender Detector",
    page_icon="👤",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2563EB;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .result-text {
        font-size: 1.5rem;
        font-weight: 500;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .image-container {
        margin-bottom: 2rem;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: rgba(237, 242, 247, 0.5);
    }
    
    .app-footer {
        text-align: center;
        margin-top: 2rem;
        opacity: 0.7;
    }
    
    .stButton>button {
        background-color: #2563EB;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        border: none;
    }
    
    .stButton>button:hover {
        background-color: #1E40AF;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Function to load the model (with caching for performance)
@st.cache_resource



def load_emotion_model():
    try:
        emotion_model_path = r"models\emotion_model.h5"
        model = load_model(emotion_model_path)
        return model
    except Exception as e:
        st.error(f"Error loading emotion model: {e}")
        return None

def load_age_gender_model():
    try:
        model_path = r"models\Age_Sex_Detection.h5"
        model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def load_nationality_model():
    try:
        model_path = r"models\nationality_model.h5"  
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading nationality model: {e}")
        return None


# Function to preprocess the image
def detect_and_crop_face(image_pil):
    image_np = np.array(image_pil.convert("RGB"))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return None  # No face found

    x, y, w, h = faces[0]  # Use the first detected face
    cropped_face = image_pil.crop((x, y, x + w, y + h))
    return cropped_face, (x, y, w, h)


def preprocess_emotion_image(uploaded_image):
    gray = uploaded_image.convert("L")  # Convert to grayscale
    resized = gray.resize((48, 48))
    img_array = np.array(resized)

    if img_array.dtype == np.float64:
        img_array = (img_array * 255).astype(np.uint8)
    elif img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)

    img_array = img_array.reshape(1, 48, 48, 1).astype("float32") / 255.0
    return img_array


def preprocess_age_gender_image(uploaded_image):
    rgb_image = uploaded_image.convert("RGB")  # Ensure 3 channels
    resized = rgb_image.resize((48, 48))       # Resize to model input size
    img_array = np.array(resized).astype("float32") / 255.0
    img_array = img_array.reshape(1, 48, 48, 3)  # Add batch dimension
    return img_array



def preprocess_nationality_image(uploaded_image):
    if uploaded_image.mode != "RGB":
        uploaded_image = uploaded_image.convert("RGB")
    image = uploaded_image.resize((128, 128)) 
    image_array = np.array(image) / 255.0  
    return np.expand_dims(image_array, axis=0)  # Shape: (1, 64, 64, 3)


# Function to make prediction
def detect_emotion_direct(image, model):
    try:
        gray = image.convert("L")
        resized = gray.resize((48, 48))
        img_array = np.array(resized) / 255.0
        img_array = img_array.reshape(1, 48, 48, 1).astype("float32")
        
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        prediction = model.predict(img_array)
        return emotion_labels[np.argmax(prediction)]
    except Exception as e:
        st.error(f"Emotion detection error: {e}")
        return "Error"



def predict_age_gender(model, image_array):
    try:
        predictions = model.predict(image_array)
        predicted_age = int(np.round(predictions[1][0]))
        gender_prob = predictions[0][0]
        predicted_gender = "Female" if gender_prob > 0.5 else "Male"
        gender_confidence = gender_prob if predicted_gender == "Female" else 1 - gender_prob
        return predicted_age, predicted_gender, float(gender_confidence)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None
    
def predict_nationality(image_array, model):
    labels = ['African','American','Asian','Indian','Other']
    prediction = model.predict(image_array)
    return labels[np.argmax(prediction)]


# Helper function to convert hex color to RGB
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# Estimate hair length
def estimate_hair_length(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    height = gray.shape[0]
    lower_half = gray[int(height/2):, :]
    dark_pixels = np.sum(lower_half < 50)
    total_pixels = lower_half.size
    dark_ratio = dark_pixels / total_pixels
    return 'long' if dark_ratio > 0.3 else 'short'

from collections import Counter
import cv2
import numpy as np

def get_dominant_dress_color(image, face_bbox):
    x, y, w, h = face_bbox

    # Define region below the face
    start_y = int(y + 1.2 * h)
    end_y = int(y + 2.5 * h)
    start_x = int(x)
    end_x = int(x + w)

    # Boundary check
    start_y = max(0, start_y)
    end_y = min(image.shape[0], end_y)
    start_x = max(0, start_x)
    end_x = min(image.shape[1], end_x)

    cropped_region = image[start_y:end_y, start_x:end_x]

    if cropped_region.size == 0:
        return (0, 0, 0), "Not Detected"

    # Convert cropped region to HSV
    hsv_crop = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2HSV)
    reshaped = hsv_crop.reshape(-1, 3)

    counts = Counter(map(tuple, reshaped))
    dominant_hsv = counts.most_common(1)[0][0]

    color_name = hsv_to_color_name(dominant_hsv)
    return dominant_hsv, color_name

def hsv_to_color_name(hsv):
    h, s, v = hsv
    h = int(h)
    s = int(s)
    v = int(v)

    if v < 40:
        return "Black"
    elif s < 40 and v > 200:
        return "White"
    elif s < 40:
        return "Gray"
    elif 0 <= h < 10 or 160 <= h <= 180:
        return "Red"
    elif 10 <= h < 25:
        return "Orange"
    elif 25 <= h < 35:
        return "Yellow"
    elif 35 <= h < 85:
        return "Green"
    elif 85 <= h < 125:
        return "Blue"
    elif 125 <= h < 155:
        return "Purple"
    elif 155 <= h < 160:
        return "Pink"
    else:
        return "Other"



    


# Main function
def main():
    st.markdown('<div class="main-header">Age and Gender Detector</div>', unsafe_allow_html=True)

    with st.spinner("Loading model... This may take a moment."):
        model = load_age_gender_model()
        emotion_model = load_emotion_model()
        nationality_model= load_nationality_model()
    if emotion_model is None:
        st.warning("Please make sure the emotion model file exists at the specified path.")
        return

    if model is None:
        st.warning("Please make sure the model file exists at the specified path.")
        return
    
    if nationality_model is None:
        st.warning("Please make sure the model file exists at the specified path.")
        return  

    st.markdown('<div class="sub-header">Upload Images</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Choose one or more images...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Detect Age & Gender", key="detect_button_files"):
        with st.spinner("Analyzing images..."):
            for i, uploaded_file in enumerate(uploaded_files):
                with st.container():
                    st.markdown(f'<div class="image-container">', unsafe_allow_html=True)
                    st.markdown(f"<h3>Image {i+1}</h3>", unsafe_allow_html=True)
                    col1, col2 = st.columns([1, 1])

                    image = Image.open(uploaded_file)
                    col1.image(image, caption=f"Image {i+1}: {uploaded_file.name}", use_column_width=True)

                    result = detect_and_crop_face(image)
                    if result is None:
                        face=image
                        face_bbox = None
                    else:
                        face, face_bbox = result 


                    processed_image = preprocess_age_gender_image(face)
                    nationality_image = preprocess_nationality_image(face)
                    #preprocess_emotion_image = preprocess_emotion_image(image)
                    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)                   
                                     
                    age, gender, confidence = predict_age_gender(model, processed_image)                    
                    emotion = detect_emotion_direct(image, emotion_model)
                    nationality = predict_nationality(nationality_image, nationality_model)
                 
                     
                    dress_color = None
                        

                    if nationality == "Indian" or nationality == "African":
                        if face_bbox is not None:
                            dominant_color_rgb,dominant_color_name = get_dominant_dress_color(opencv_image, face_bbox)
                            dress_color = dominant_color_name
                        else:
                            dress_color = "Not Detected"
                        

                    col2.markdown('<div class="sub-header">Results:</div>', unsafe_allow_html=True)

                    col2.markdown(
                        f'<div class="result-text" style="background-color: rgba(234, 88, 12, 0.1);">Emotion: {emotion}</div>',
                        unsafe_allow_html=True,
                    )
                    col2.markdown(
                        f'<div class="result-text" style="background-color: rgba(20, 184, 166, 0.1);">Nationality: {nationality}</div>',
                        unsafe_allow_html=True,
                    )

                    if nationality == "Indian":
                        col2.markdown(
                            f'<div class="result-text" style="background-color: rgba(37, 99, 235, 0.1);">Age: {age}</div>',
                            unsafe_allow_html=True,
                        )
                        gender_color = "#9F7AEA" if gender == "Female" else "#4F46E5"
                        col2.markdown(
                            f'<div class="result-text" style="background-color: rgba({", ".join(map(str, hex_to_rgb(gender_color)))}, 0.1);">'
                            f"Gender: {gender}<br>"
                            f"<small>Confidence: {confidence:.2%}</small>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        col2.markdown(
                            f'<div class="result-text" style="background-color: rgba(255, 223, 186, 0.3);">Dress Color: {dress_color}</div>',
                            unsafe_allow_html=True,
                        )

                    elif nationality == "American":
                        col2.markdown(
                            f'<div class="result-text" style="background-color: rgba(37, 99, 235, 0.1);">Age: {age}</div>',
                            unsafe_allow_html=True,
                        )
                        gender_color = "#9F7AEA" if gender == "Female" else "#4F46E5"
                        col2.markdown(
                            f'<div class="result-text" style="background-color: rgba({", ".join(map(str, hex_to_rgb(gender_color)))}, 0.1);">'
                            f"Gender: {gender}<br>"
                            f"<small>Confidence: {confidence:.2%}</small>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    elif nationality == "African":
                        col2.markdown(
                            f'<div class="result-text" style="background-color: rgba(255, 223, 186, 0.3);">Dress Color: {dress_color}</div>',
                            unsafe_allow_html=True,
                        )

# No extra fields shown for other nationalities


                    if i < len(uploaded_files) - 1:
                        st.markdown("<hr>", unsafe_allow_html=True)

    elif st.button("Detect Age & Gender", key="detect_button_empty"):
        st.info("Please upload one or more images first.")

    st.markdown('<div class="app-footer">Powered by NULLCLASS🧑‍💻</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
