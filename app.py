import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model('brain_tumor_model.h5')

# Class labels
class_names = ['Glioma', 'Pituitary', 'Meningioma', 'No Tumor']

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize sesuai input model (ubah kalau beda)
    image = np.array(image)
    if image.shape[-1] == 4:  # Kalau RGBA, ubah ke RGB
        image = image[..., :3]
    image = image / 255.0  # Normalisasi
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit UI
st.title("Brain Tumor Classification")
st.write("Upload an MRI image to predict the tumor type:")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Predict button
    if st.button("Predict"):
        with st.spinner('Predicting...'):
            preprocessed_image = preprocess_image(image)
            prediction = model.predict(preprocessed_image)[0]
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

            st.success(f"Prediction: **{predicted_class}**")
            st.write(f"Confidence: {confidence:.2f}%")

            # Show all class probabilities
            st.subheader("Prediction Probabilities:")
            for i, prob in enumerate(prediction):
                st.write(f"{class_names[i]}: {prob*100:.2f}%")
