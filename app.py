import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model
import gdown
import os
import tensorflow as tf

# Konfigurasi halaman
st.set_page_config(page_title="Deteksi Tumor Otak", layout="wide", page_icon="🧠")

# Path dan URL model
model_path = 'brain_tumor_model.h5'
file_id = '18lLL4vDzXS9gdDXksyJhuY5MedaafKv7'
url = f'https://drive.google.com/uc?id={file_id}'
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Download model jika belum ada
if not os.path.exists(model_path):
    with st.spinner('Mengunduh model dari Google Drive...'):
        success = gdown.download(url, model_path, quiet=False)
        if not success:
            st.error("❌ Gagal mengunduh model dari Google Drive.")
            st.stop()

# Load model
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# Navigasi
menu = st.sidebar.radio("Navigasi", ["🏠 Home", "📖 Info Tumor", "ℹ️ Tentang Aplikasi"])

# ==================== HOME ====================
if menu == "🏠 Home":
    st.title("🧠 Deteksi Jenis Tumor Otak dari Citra MRI")
    st.markdown("Upload gambar MRI otak untuk mendeteksi jenis tumor menggunakan model deep learning.")

    uploaded_file = st.file_uploader("Unggah gambar MRI (jpg/jpeg/png)...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='🖼️ Gambar yang Diunggah', use_column_width=True)

            # Preprocessing
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediksi
            prediction = model.predict(img_array)
            pred_index = np.argmax(prediction)
            confidence = prediction[0][pred_index]

            if confidence < 0.6:
                st.warning("⚠️ Gambar tidak dikenali sebagai MRI otak atau kualitas gambar rendah.")
            else:
                predicted_class = class_names[pred_index]
                st.success(f"✅ Jenis tumor terdeteksi: **{predicted_class.upper()}**")
                st.info(f"Confidence: **{confidence:.2f}**")

        except UnidentifiedImageError:
            st.error("❌ File yang diunggah bukan gambar yang valid.")
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat memproses gambar: {e}")

# ==================== INFO TUMOR ====================
elif menu == "📖 Info Tumor":
    st.title("📚 Informasi Jenis Tumor Otak")
    pilihan = st.selectbox("Pilih jenis tumor untuk melihat penjelasan:", class_names)

    if pilihan == "glioma":
        st.subheader("🧬 Glioma")
        st.write("""
        Glioma adalah tumor yang berasal dari sel glial di otak atau sumsum tulang belakang. 
        Jenis ini bisa bersifat jinak maupun ganas, namun banyak di antaranya bersifat agresif dan tumbuh cepat.
        """)

    elif pilihan == "meningioma":
        st.subheader("🧬 Meningioma")
        st.write("""
        Meningioma adalah tumor yang berasal dari meninges, yaitu selaput yang melindungi otak dan sumsum tulang belakang.
        Biasanya jinak dan tumbuh lambat, tetapi bisa menyebabkan gejala tergantung lokasi tumbuhnya.
        """)

    elif pilihan == "pituitary":
        st.subheader("🧬 Pituitary Tumor")
        st.write("""
        Tumor ini tumbuh di kelenjar pituitari (hipofisis), yang berperan penting dalam produksi hormon.
        Bisa memengaruhi berbagai fungsi tubuh dan terkadang menyebabkan gangguan hormonal.
        """)

    elif pilihan == "notumor":
        st.subheader("✅ No Tumor")
        st.write("""
        Tidak terdeteksi adanya tumor dalam citra MRI otak. Namun, untuk diagnosis pasti tetap diperlukan pemeriksaan oleh dokter spesialis.
        """)

# ==================== TENTANG APLIKASI ====================
elif menu == "ℹ️ Tentang Aplikasi":
    st.title("ℹ️ Tentang Aplikasi Ini")

    st.markdown("""
    **Deteksi Tumor Otak dari Citra MRI** adalah aplikasi berbasis deep learning yang bertujuan untuk membantu mengenali jenis tumor otak
    dari citra MRI secara cepat dan sederhana.

    Aplikasi ini **tidak menggantikan diagnosis medis** dan hanya ditujukan untuk **edukasi dan pembelajaran**.

    ---
    🔧 **Fitur Aplikasi:**
    - Prediksi jenis tumor dari gambar MRI
    - Penjelasan edukatif tentang jenis-jenis tumor
    - Tampilan responsif & mudah digunakan

    💡 **Teknologi:**  
    Dibuat menggunakan TensorFlow, Streamlit, dan model CNN.

    ✨ Didesain dengan harapan bisa membantu kamu belajar teknologi AI di dunia medis secara ringan dan menyenangkan.

    ---
    👩‍💻 Developer: Eva Meivina Dwiana  
    📫 Kontak: [eva@email.com](mailto:eva@email.com)
    """)

