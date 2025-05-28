import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model
import gdown
import os
import tensorflow as tf

# Konfigurasi halaman
st.set_page_config(page_title="Deteksi Tumor Otak", layout="wide")

# Path dan URL model
model_path = 'brain_tumor_model.h5'
file_id = '18lLL4vDzXS9gdDXksyJhuY5MedaafKv7'
url = f'https://drive.google.com/uc?id={file_id}'
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Download model jika belum tersedia
if not os.path.exists(model_path):
    with st.spinner('Mengunduh model...'):
        success = gdown.download(url, model_path, quiet=False)
        if not success:
            st.error("Gagal mengunduh model dari Google Drive.")
            st.stop()

# Load model
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# Sidebar navigasi
menu = st.sidebar.radio("Menu", ["Home", "Info Tumor"])

# ==================== HOME ====================
if menu == "Home":
    st.title("Deteksi Jenis Tumor Otak dari Citra MRI")
    st.write("Silakan unggah gambar MRI otak untuk mengetahui jenis tumor menggunakan model deep learning.")

    uploaded_file = st.file_uploader("Unggah gambar MRI (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Gambar yang Diunggah', use_column_width=True)

            # Preprocessing
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediksi
            prediction = model.predict(img_array)
            pred_index = np.argmax(prediction)
            confidence = prediction[0][pred_index]

            # Validasi confidence
            if confidence < 0.6:
                st.warning("Gambar tidak dikenali sebagai MRI otak atau kualitas gambar rendah.")
            else:
                predicted_class = class_names[pred_index]
                st.success(f"Jenis tumor terdeteksi: {predicted_class.upper()}")
                st.write(f"Confidence: {confidence:.2f}")

        except UnidentifiedImageError:
            st.error("File yang diunggah bukan gambar yang valid.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses gambar: {e}")

# ==================== INFO TUMOR ====================
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
