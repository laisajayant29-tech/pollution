import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Model load karne ke liye function
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("pollution_model.h5")
    return model

# Model ko load karein
model = load_model()

st.title("🌍 Environmental Pollution Classification App")
st.write("Upload an image to detect the type of pollution using CNN.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Image display karein
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Image ko preprocess karein
    image = image.resize((64, 64))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    
    # Prediction karein
    predictions = model.predict(img_array)
    class_names = ['Air_Pollution', 'Noise_Traffic', 'Water_Pollution']
    
    score = tf.nn.softmax(predictions[0])
    
    # Result show karein
    st.write(f"### Prediction: **{class_names[np.argmax(score)]}**")
    st.write(f"### Confidence: **{100 * np.max(score):.2f}%**")
