import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.title("Brain Tumor Classification")
st.write("Upload an MRI image to predict the tumor type:")

# Cek apakah file model ada di direktori kerja
if 'brain_tumor_model.h5' not in os.listdir():
    st.error("File model 'brain_tumor_model.h5' tidak ditemukan di server. Pastikan file sudah di-upload dengan benar.")
    st.stop()

# Load model dengan penanganan error
try:
    model = tf.keras.models.load_model('brain_tumor_model.h5')
    st.success("Model berhasil dimuat.")
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}")
    st.stop()

# Label kelas
class_names = ['Glioma', 'Pituitary', 'Meningioma', 'No Tumor']

# Fungsi preprocessing gambar
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    if image.shape[-1] == 4:
        image = image[..., :3]
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Upload file gambar MRI
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Predict"):
        with st.spinner('Predicting...'):
            preprocessed_image = preprocess_image(image)
            prediction = model.predict(preprocessed_image)[0]
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

            st.success(f"Prediction: **{predicted_class}**")
            st.write(f"Confidence: {confidence:.2f}%")

            st.subheader("Prediction Probabilities:")
            for i, prob in enumerate(prediction):
                st.write(f"{class_names[i]}: {prob*100:.2f}%")
