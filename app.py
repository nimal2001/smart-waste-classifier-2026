import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Smart Waste Classifier", page_icon="♻️")
st.title("♻️ Smart Waste Classification System")
st.write("Upload an image of waste to classify it into one of the predefined categories.")

# --- 2. LOAD RESOURCES ---
# @st.cache_resource prevents the app from reloading the heavy model every time the user clicks a button
@st.cache_resource
def load_model_and_classes():
    # Load the model
    model = tf.keras.models.load_model('waste_classifier_model.keras')
    
    # Load class names
    with open('class_names.json', 'r') as f:
        classes = json.load(f)
    
    return model, classes

try:
    model, class_mapping = load_model_and_classes()
except Exception as e:
    st.error(f"Error loading model or class names. Ensure 'waste_classifier_model.keras' and 'class_names.json' are in the same folder. Details: {e}")
    st.stop()

# --- 3. UI: IMAGE UPLOAD (Requirement 1) ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # --- 4. DISPLAY IMAGE (Requirement 2) ---
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("Processing...")

    # --- 5. PREPROCESS IMAGE ---
    image = image.convert('RGB')
    image = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) 
    
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # --- 6. PREDICTION (Requirement 3) ---
    predictions = model.predict(img_array)[0]
    
    predicted_class_index = np.argmax(predictions)
    # Using the safer .get() method we added earlier
    predicted_class_name = class_mapping.get(str(predicted_class_index), f"Unknown_ID_{predicted_class_index}")
    confidence = predictions[predicted_class_index] * 100

    # --- 7. DISPLAY PREDICTION & CONFIDENCE (Requirement 4) ---
    st.success(f"### Prediction: **{predicted_class_name}**")
    st.info(f"**Confidence:** {confidence:.2f}%")

    # --- 8. DISPLAY ALL PROBABILITIES (Requirement 5) ---
    st.write("### Probability Breakdown")
    
    # Create a visual bar chart of all probabilities
    prob_dict = {class_mapping.get(str(i), f"Unknown_ID_{i}"): float(prob) * 100 for i, prob in enumerate(predictions)}
    
    # Sort the dictionary by probability for better visualization
    sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))
    
    # Display as progress bars
    for class_name, prob in sorted_probs.items():
        cols = st.columns([2, 8, 2])
        with cols[0]:
            st.write(class_name)
        with cols[1]:
            st.progress(int(prob) if not np.isnan(prob) else 0)
        with cols[2]:
            st.write(f"{prob:.2f}%")
