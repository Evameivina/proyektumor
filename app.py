import streamlit as st
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.models import load_model
import gdown
import os

# Config halaman
st.set_page_config(page_title="Deteksi Tumor Otak", layout="wide")

# Path dan URL model
model_path = 'brain_tumor_model.h5'
file_id = '18lLL4vDzXS9gdDXksyJhuY5MedaafKv7'
url = f'https://drive.google.com/uc?id={file_id}'
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
confidence_threshold = 0.6

# Download model kalau belum ada
if not os.path.exists(model_path):
    with st.spinner("Mengunduh model..."):
        gdown.download(url, model_path, quiet=False)

# Load model
model = load_model(model_path)

def is_probably_mri(img_pil):
    img = np.array(img_pil)
    if img is None or img.shape[0] < 100 or img.shape[1] < 100:
        return False

    # Jika RGB
    if len(img.shape) == 3 and img.shape[2] == 3:
        b, g, r = cv2.split(img)
        stds = [np.std(c) for c in (b, g, r)]
        min_std, max_std = min(stds), max(stds)
        ratio = min_std / (max_std + 1e-6)
        if ratio > 0.9:
            return True

        mean_total = np.mean(img)
        green_ratio = np.mean(g) / (mean_total + 1e-6)
        if green_ratio > 0.5:
            return False

    # Kalau grayscale 2D
    if len(img.shape) == 2:
        return True

    return True

# Sidebar menu
menu = st.sidebar.radio("Menu", ["Home", "Info Tumor"])

if menu == "Home":
    st.title("Deteksi Jenis Tumor Otak dari Citra MRI")
    st.write("Unggah gambar MRI otak (jpg/jpeg/png) untuk mengetahui jenis tumor menggunakan model deep learning.")

    uploaded_file = st.file_uploader("Unggah gambar MRI otak", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)

        if not is_probably_mri(image):
            st.warning("Gambar ini kemungkinan bukan citra MRI otak. Mohon unggah gambar MRI otak yang valid.")
        else:
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            pred_index = np.argmax(prediction)
            confidence = prediction[0][pred_index]

            if confidence < confidence_threshold:
                st.warning("Model tidak yakin dengan prediksi untuk gambar ini.")
            else:
                pred_class = class_names[pred_index]
                st.success(f"Prediksi tumor: {pred_class.upper()}")
                st.write(f"Confidence: {confidence:.2f}")

elif menu == "Info Tumor":
    st.title("Informasi Jenis Tumor Otak")
    pilihan = st.selectbox("Pilih jenis tumor untuk melihat penjelasan:", class_names)

    if pilihan == "glioma":
        st.subheader("Glioma")
        st.write("""
            Glioma adalah tumor yang berasal dari sel glial di otak atau sumsum tulang belakang.
            Glioma dapat bersifat jinak maupun ganas. Banyak kasus glioma bersifat agresif dan tumbuh cepat.
        """)

    elif pilihan == "meningioma":
        st.subheader("Meningioma")
        st.write("""
            Meningioma merupakan tumor yang tumbuh dari meninges, lapisan pelindung otak dan sumsum tulang belakang.
            Umumnya bersifat jinak dan tumbuh lambat, tetapi tetap bisa menimbulkan gejala sesuai lokasi pertumbuhannya.
        """)

    elif pilihan == "pituitary":
        st.subheader("Tumor Pituitari")
        st.write("""
            Tumor ini tumbuh di kelenjar pituitari (hipofisis) yang berperan dalam produksi berbagai hormon.
            Dapat menyebabkan gangguan hormonal dan memengaruhi fungsi tubuh secara luas.
        """)

    elif pilihan == "notumor":
        st.subheader("Tidak Ada Tumor")
        st.write("""
            Tidak terdeteksi adanya tumor pada citra MRI otak yang diunggah.
            Namun, hasil ini tidak menggantikan diagnosis profesional. Konsultasikan dengan dokter untuk pemeriksaan lebih lanjut.
        """)
