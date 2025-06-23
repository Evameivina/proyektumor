import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model
import gdown
import os

# --- Page configuration ---
st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

# --- Custom CSS Styling ---
st.markdown("""
<style>
    body, html, #root > div:nth-child(1) {
        height: 100vh;
        overflow-y: auto;
        background: #f9fafb;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #333;
        margin: 0;
        padding: 0;
    }
    .menu-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a73e8;
        text-align: center;
        margin: 0.3rem 0 0.8rem 0;
        border-bottom: 2px solid #4285f4;
        padding-bottom: 0.6rem;
        user-select: none;
    }
    .instruction-box {
        background-color: #e8f0fe;
        border-left: 4px solid #1a73e8;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        max-width: 650px;
        margin: 0 auto 1.5rem auto; 
        font-size: 1rem;
        line-height: 1.3;
        color: #202124;
        user-select: none;
    }
    .prediction-success {
        text-align: center;
        font-size: 1.2rem;
        font-weight: 700;
        color: #188038;
        margin-top: 0.8rem;
        user-select: none;
        border: 2px solid #188038;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        max-width: 480px;
        margin-left: auto;
        margin-right: auto;
        background-color: #e6f4ea;
    }
    .prediction-info {
        text-align: center;
        font-size: 1rem;
        font-weight: 600;
        color: #155ab3;
        margin-top: 0.3rem;
        user-select: none;
    }
    .sidebar-menu-label {
        font-weight: 700;
        font-size: 1.4rem;
        color: #1a73e8;
        margin-bottom: 0.2rem; 
        padding-left: 12px;
        border-bottom: 2px solid #4285f4;
        padding-bottom: 0.3rem;
        user-select: none;
    }
    .main {
        max-width: 850px;
        margin: 0 auto 2rem auto;
        padding: 0 15px;
    }
</style>
""", unsafe_allow_html=True)

# --- Download and Load Model ---
file_id = '153Pi99NMlc7e-YgHw1V7mW5GZV_B9QJq'
download_url = f'https://drive.google.com/uc?id={file_id}'
model_path = "brain_tumor_model.h5"
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

if not os.path.exists(model_path):
    with st.spinner("Mengunduh model dari Google Drive..."):
        downloaded = gdown.download(download_url, model_path, quiet=False)
        if not downloaded:
            st.error("Gagal mengunduh model.")
            st.stop()

try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# --- MRI Validity Check ---
def is_probably_mri(image_pil):
    if image_pil.width < 100 or image_pil.height < 100:
        return False
    img_np = np.array(image_pil)
    if len(img_np.shape) == 2:
        return True
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        stds = np.std(img_np, axis=(0,1))
        ratio = stds.min() / (stds.max() + 1e-6)
        if ratio > 0.9:
            return True
    return False

# --- Sidebar Menu ---
st.sidebar.markdown('<div class="sidebar-menu-label">Menu</div>', unsafe_allow_html=True)
page = st.sidebar.radio("", ["Panduan Penggunaan", "Deteksi Tumor", "Informasi Tumor"])

# --- Panduan Penggunaan ---
if page == "Panduan Penggunaan":
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<div class="menu-title">Panduan Penggunaan Aplikasi</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="instruction-box">
    <h4>Langkah-langkah:</h4>
    <ol>
        <li>Siapkan gambar MRI otak dengan format JPG/JPEG/PNG.</li>
        <li>Pastikan gambar jelas dan tidak buram.</li>
        <li>Pilih menu <strong>Deteksi Tumor</strong> dan unggah gambar.</li>
        <li>Sistem akan melakukan prediksi dan menampilkan jenis tumor serta tingkat kepercayaannya.</li>
        <li>Gunakan informasi ini sebagai indikasi awal. Tetap konsultasikan dengan dokter spesialis.</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Deteksi Tumor ---
elif page == "Deteksi Tumor":
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<div class="menu-title">Deteksi Tumor Otak</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Unggah Gambar MRI Otak", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        try:
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, caption='Gambar yang Diunggah', use_column_width=True)

            if not is_probably_mri(img):
                st.warning("Gambar yang diunggah tidak sesuai dan tidak terdeteksi")
            else:
                img_resized = img.resize((224, 224))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction = model.predict(img_array)
                pred_index = np.argmax(prediction)
                confidence = prediction[0][pred_index]

                if confidence < 0.6:
                    st.warning("Model tidak yakin dengan prediksi. Coba gambar lain.")
                else:
                    predicted_class = class_names[pred_index]
                    st.markdown(f'<div class="prediction-success">Jenis tumor terdeteksi: <strong>{predicted_class.upper()}</strong></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="prediction-info">Tingkat kepercayaan: <strong>{confidence:.2f}</strong></div>', unsafe_allow_html=True)

        except UnidentifiedImageError:
            st.error("File yang diunggah bukan gambar yang valid.")
        except Exception as e:
            st.error(f"Kesalahan saat memproses gambar: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# --- Informasi Tumor ---
elif page == "Informasi Tumor":
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<div class="menu-title">Informasi Jenis Tumor Otak</div>', unsafe_allow_html=True)

    pilihan = st.selectbox("Pilih jenis tumor untuk informasi:", class_names)

    if pilihan == "glioma":
        st.markdown('<div class="menu-title">Glioma</div>', unsafe_allow_html=True)
        st.markdown("""...""", unsafe_allow_html=True)  # Tetap seperti sebelumnya

    elif pilihan == "meningioma":
        st.markdown('<div class="menu-title">Meningioma</div>', unsafe_allow_html=True)
        st.markdown("""...""", unsafe_allow_html=True)  # Tetap seperti sebelumnya

    elif pilihan == "pituitary":
        st.markdown('<div class="menu-title">Tumor Pituitary</div>', unsafe_allow_html=True)
        st.markdown("""...""", unsafe_allow_html=True)  # Tetap seperti sebelumnya

    elif pilihan == "notumor":
        st.markdown('<div class="menu-title">Tidak Ada Tumor</div>', unsafe_allow_html=True)
        st.markdown("""...""", unsafe_allow_html=True)  # Tetap seperti sebelumnya

    st.markdown("</div>", unsafe_allow_html=True)
