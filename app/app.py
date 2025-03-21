import streamlit as st
import tensorflow as tf
import numpy as np
import json
from keras.utils import load_img, img_to_array
from PIL import Image
import os
import time

# Get the root directory path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load the trained model
MODEL_PATH = os.path.join(ROOT_DIR, "model", "transfer_learning_model.keras")
CLASS_INDICES_PATH = os.path.join(ROOT_DIR, "model", "class_indices.json")
CURES_PATH = os.path.join(ROOT_DIR, "cures", "cures.json")
IMAGE_UPLOAD_PATH = os.path.join(ROOT_DIR, "images")  # Ensure this folder exists

# Create images directory if it doesn't exist
os.makedirs(IMAGE_UPLOAD_PATH, exist_ok=True)

# Load model and class indices
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)
    class_labels = {v: k for k, v in class_indices.items()}  # Reverse mapping

with open(CURES_PATH, "r") as f:
    cures = json.load(f)

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(64, 64))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize
    return img

def predict_disease(image_path):
    img = preprocess_image(image_path)
    result = model.predict(img)
    confidence = np.max(result)
    predicted_index = np.argmax(result)
    
    if confidence < 0.6:  # Confidence threshold
        return "Not in Database", None
    
    predicted_class = class_labels[predicted_index]
    
    if "healthy" in predicted_class.lower():
        return "Healthy Plant", cures.get("healthy", ["No action needed. Keep maintaining good agricultural practices."])
    
    # Format the disease name for display and cure lookup
    disease_name = predicted_class.replace("_", " ").replace("(", "").replace(")", "")
    # Try different variations of the class name to find a match in cures
    cure_info = cures.get(predicted_class) or cures.get(predicted_class.replace("_", " ")) or cures.get(disease_name)
    
    if not cure_info:
        cure_info = ["No cure information available."]
    
    return disease_name, cure_info

# Streamlit UI
st.set_page_config(page_title="Plant Disease Detector", layout="wide")

# Custom CSS for background colors and animations
st.markdown("""
    <style>
    .healthy-bg {
        background-color: #e6ffe6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .disease-bg {
        background-color: #ffe6e6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .unknown-bg {
        background-color: #f2f2f2;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .debug-info {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        font-size: 0.9em;
    }
    .loading {
        display: inline-block;
        width: 50px;
        height: 50px;
        border: 3px solid #f3f3f3;
        border-radius: 50%;
        border-top: 3px solid #3498db;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .success-icon {
        color: #2ecc71;
        font-size: 50px;
        text-align: center;
        margin: 20px auto;
    }
    .plant-box {
        background-color: #1E1E1E;
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 2px solid #333;
    }
    .plant-box h3 {
        color: #4CAF50;
        text-align: center;
        margin-bottom: 20px;
        font-size: 24px;
    }
    .plant-box p {
        color: #E0E0E0;
        text-align: center;
        margin-bottom: 20px;
    }
    .plant-category {
        background-color: #2D2D2D;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .plant-category h4 {
        color: #81C784;
        margin-bottom: 15px;
        font-size: 20px;
    }
    .plant-list {
        color: #BDBDBD;
        margin-left: 20px;
    }
    .plant-list li {
        margin: 8px 0;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌿 Plant Disease Detection System")

# Create a small box for supported plants
st.markdown("""
    <div class="plant-box">
        <h3>🌱 Supported Plants</h3>
        <p>Fruits: 🍎 Apple, 🫐 Blueberry, 🍒 Cherry, 🍇 Grape, 🍊 Orange, 🍑 Peach, 🫐 Raspberry, 🍓 Strawberry</p>
        <p>Vegetables: 🌽 Corn, 🫑 Pepper, 🥔 Potato, 🫘 Soybean, 🎃 Squash, 🍅 Tomato</p>
        <p style="color: #81C784; font-size: 12px;">Note: Upload clear leaf images for best results</p>
    </div>
""", unsafe_allow_html=True)

st.write("Upload a leaf image to detect diseases and get treatment suggestions.")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Create two columns
    col1, col2 = st.columns(2)
    
    # Left column for image
    with col1:
        st.subheader("📸 Uploaded Image")
        st.image(Image.open(uploaded_file), caption="Leaf Image", use_column_width=True)
    
    # Right column for results
    with col2:
        st.subheader("🔍 Analysis Results")
        
        # Save uploaded image
        image_path = os.path.join(IMAGE_UPLOAD_PATH, uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Add predict button
        if st.button("🔍 Predict Disease", type="primary"):
            # Create a placeholder for the animation
            animation_placeholder = st.empty()
            
            # Show loading animation
            animation_placeholder.markdown('<div class="loading"></div>', unsafe_allow_html=True)
            
            # Process image and get results
            disease, cure = predict_disease(image_path)
            
            # Clear the animation placeholder
            animation_placeholder.empty()
            
            if disease == "Not in Database":
                st.markdown('<div class="unknown-bg">', unsafe_allow_html=True)
                st.error("❌ This leaf is not in our database.")
                st.markdown('</div>', unsafe_allow_html=True)
                
            elif disease == "Healthy Plant":
                st.markdown('<div class="success-icon">✓</div>', unsafe_allow_html=True)
                st.markdown('<div class="healthy-bg">', unsafe_allow_html=True)
                st.success("✅ The plant is healthy!")
                st.write("### 🌱 Maintenance Tips:")
                for tip in cure:
                    st.write(f"• {tip}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            else:
                st.markdown('<div class="success-icon">✓</div>', unsafe_allow_html=True)
                st.markdown('<div class="disease-bg">', unsafe_allow_html=True)
                st.warning(f"⚠️ Detected Disease: {disease}")
                st.write("### 🌱 Recommended Treatment Steps:")
                for step in cure:
                    st.write(f"• {step}")
                st.markdown('</div>', unsafe_allow_html=True)
