import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Title
st.title("✍️ Handwritten Digit Recognition (CNN)")
st.write("Upload an image of a digit (0–9)")

# Build model architecture
def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

# Load model + weights
@st.cache_resource
def load_model():
    model = build_model()
    model.load_weights("mnist_weights.weights.h5")  # ✅ updated name
    return model

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('L')
        image = image.resize((28, 28))

        st.image(image, caption="Uploaded Image", width=150)

        img_array = np.array(image) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        prediction = model.predict(img_array)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        st.success(f"Predicted Digit: {digit}")
        st.info(f"Confidence: {confidence:.2f}")

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.warning("Please upload an image")

st.markdown("---")
st.write("CNN Model trained on MNIST dataset (~98% accuracy)")
