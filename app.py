import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Reduce TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# App Title
st.title("✍️ Handwritten Digit Recognition (CNN)")
st.write("Upload an image of a digit (0–9)")

# Load .keras model safely
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_model.keras")

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # Convert to grayscale
        image = Image.open(uploaded_file).convert('L')

        # Resize to 28x28
        image = image.resize((28, 28))

        # Show image
        st.image(image, caption="Uploaded Image", width=150)

        # Convert to numpy array
        img_array = np.array(image)

        # Normalize
        img_array = img_array / 255.0

        # Reshape for CNN
        img_array = img_array.reshape(1, 28, 28, 1)

        # Predict
        prediction = model.predict(img_array)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        # Output
        st.success(f"Predicted Digit: {digit}")
        st.info(f"Confidence: {confidence:.2f}")

    except Exception as e:
        st.error(f"Error processing image: {e}")

else:
    st.warning("Please upload an image to predict.")

# Footer
st.markdown("---")
st.write("Model: CNN trained on MNIST dataset (~98% accuracy)")
