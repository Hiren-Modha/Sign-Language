import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps



# Load trained model
model = load_model('hs1.h5')


# Mapping of classes to letters
CLASS_MAP = {i: chr(65 + i) for i in range(26) if i != 9}  # Exclude 'J'

# Preprocess uploaded image
def preprocess_image(image):
    image = ImageOps.grayscale(image)  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image_array = np.array(image) / 255.0  # Normalize
    image_array = image_array.reshape(1, 28, 28, 1)  # Reshape for model input
    return image_array

# Streamlit App
st.title("Sign Language to Text Converter")

uploaded_file = st.file_uploader("Upload a sign language image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")
    image_array = preprocess_image(image)
    
    # Make prediction
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)
    st.write(f"Predicted Letter: **{CLASS_MAP[predicted_class]}**")
