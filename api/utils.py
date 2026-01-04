import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from typing import Optional, Tuple, Dict

def preprocess_image(image_bytes: bytes, target_size: Tuple[int, int] = (48, 48)) -> Optional[np.ndarray]:
    """Preprocess image for age/gender model input"""
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert RGBA to RGB if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        # Resize to model input size
        img_resized = cv2.resize(img_array, target_size)
        
        # Normalize to [0,1]
        img_normalized = img_resized.astype('float32') / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

def preprocess_for_ethnicity(image_bytes: bytes, target_size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
    """Preprocess image for ethnicity model input"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(image)
        
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        img_resized = cv2.resize(img_array, target_size)
        img_normalized = img_resized.astype('float32') / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
    except Exception as e:
        print(f"Error preprocessing for ethnicity: {str(e)}")
        return None

def preprocess_for_emotion(image_bytes: bytes, target_size: Tuple[int, int] = (48, 48)) -> Optional[np.ndarray]:
    """Preprocess image for emotion model input (grayscale)"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(image)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            if img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
            elif img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Resize
        img_resized = cv2.resize(img_array, target_size)
        
        # Normalize
        img_normalized = img_resized.astype('float32') / 255.0
        
        # Add channel dimension
        if len(img_normalized.shape) == 2:
            img_normalized = np.expand_dims(img_normalized, axis=-1)
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
    except Exception as e:
        print(f"Error preprocessing for emotion: {str(e)}")
        return None

def draw_labels_on_image(image_bytes: bytes, predictions: Dict) -> Optional[str]:
    """Draw prediction labels on the image and return as base64 string"""
    try:
        # Load original image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Create a copy for drawing
        img_with_labels = image.copy()
        draw = ImageDraw.Draw(img_with_labels)
        
        # Try to load a font, fallback to default if not available
        # Very large font sizes for maximum visibility
        try:
            font_large = ImageFont.truetype("arial.ttf", 80)
            font_medium = ImageFont.truetype("arial.ttf", 65)
            font_small = ImageFont.truetype("arial.ttf", 55)
        except:
            try:
                font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 80)
                font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 65)
                font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 55)
            except:
                # For default font, we'll scale it up
                font_large = ImageFont.load_default()
                font_medium = ImageFont.load_default()
                font_small = ImageFont.load_default()
        
        # Get image dimensions
        width, height = img_with_labels.size
        
        # Background rectangle for text
        # Increased padding and line height for very large text
        padding = 25
        text_y = padding
        line_height = 80
        
        # Draw semi-transparent background
        overlay = Image.new('RGBA', img_with_labels.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # Calculate text box dimensions
        labels = []
        labels.append(f"Age: {predictions.get('age', 'N/A')} years")
        labels.append(f"Gender: {predictions.get('gender', 'N/A')}")
        labels.append(f"Confidence: {predictions.get('gender_confidence', 0) * 100:.1f}%")
        
        if predictions.get('ethnicity'):
            labels.append(f"Ethnicity: {predictions['ethnicity'].get('label', 'N/A')}")
            labels.append(f"Ethnicity Conf: {predictions['ethnicity'].get('confidence', 0) * 100:.1f}%")
        
        if predictions.get('emotion'):
            labels.append(f"Emotion: {predictions['emotion'].get('label', 'N/A')}")
            labels.append(f"Emotion Conf: {predictions['emotion'].get('confidence', 0) * 100:.1f}%")
        
        # Calculate box width (find longest text)
        max_text_width = max([draw.textlength(label, font=font_medium) for label in labels])
        box_width = int(max_text_width) + padding * 2
        box_height = len(labels) * line_height + padding * 2
        
        # Draw background box
        overlay_draw.rectangle(
            [(padding, padding), (padding + box_width, padding + box_height)],
            fill=(0, 0, 0, 180)  # Semi-transparent black
        )
        
        # Composite overlay onto image
        img_with_labels = Image.alpha_composite(
            img_with_labels.convert('RGBA'),
            overlay
        ).convert('RGB')
        draw = ImageDraw.Draw(img_with_labels)
        
        # Draw text labels
        y_offset = padding + 5
        for i, label in enumerate(labels):
            if i == 0:  # Age - larger font
                draw.text((padding + 5, y_offset), label, fill=(255, 255, 255), font=font_large)
            elif i == 1:  # Gender - medium font
                draw.text((padding + 5, y_offset), label, fill=(255, 255, 0), font=font_medium)
            else:  # Other info - smaller font
                draw.text((padding + 5, y_offset), label, fill=(200, 200, 200), font=font_small)
            y_offset += line_height
        
        # Convert to base64
        buffered = io.BytesIO()
        img_with_labels.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/jpeg;base64,{img_str}"
        
    except Exception as e:
        print(f"Error drawing labels: {str(e)}")
        return None

