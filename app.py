import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Reduce TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Title
st.title("✍️ Handwritten Digit Recognition (CNN)")
st.write("Upload an image of a digit (0–9)")

# Load model (cached)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_cnn_model.h5")

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # Process image
        image = Image.open(uploaded_file).convert('L')
        image = image.resize((28, 28))

        st.image(image, caption="Uploaded Image", width=150)

        # Prepare for prediction
        img_array = np.array(image) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # Predict
        prediction = model.predict(img_array)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        # Output
        st.success(f"Predicted Digit: {digit}")
        st.info(f"Confidence: {confidence:.2f}")

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.warning("Please upload an image")

# Footer
st.markdown("---")
st.write("CNN Model trained on MNIST dataset (~98% accuracy)")
