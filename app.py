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
        color: #0f9d58;
        text-align: center;
        margin: 0.3rem 0 0.8rem 0;
        border-bottom: 2px solid #0b8043;
        padding-bottom: 0.6rem;
        user-select: none;
    }
    .instruction-box {
        background-color: #e6f4ea;
        border-left: 4px solid #0f9d58;
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
        color: #0b8043;
        margin-top: 0.8rem;
        user-select: none;
        border: 2px solid #0b8043;
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
        color: #0f9d58;
        margin-top: 0.3rem;
        user-select: none;
    }
    .sidebar-menu-label {
        font-weight: 700;
        font-size: 1.4rem;
        color: #0f9d58;
        margin-bottom: 0.2rem; 
        padding-left: 12px;
        border-bottom: 2px solid #0b8043;
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

# --- MRI Check ---
def is_probably_mri(image_pil):
    try:
        img_np = np.array(image_pil)

        if image_pil.width < 100 or image_pil.height < 100:
            return False

        if len(img_np.shape) == 2:
            return True

        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            b, g, r = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]
            channel_std = [np.std(b), np.std(g), np.std(r)]
            max_std = max(channel_std)
            min_std = min(channel_std)
            ratio = min_std / (max_std + 1e-6)
            if ratio > 0.9:
                return True

            mean_total = np.mean(img_np)
            green_ratio = np.mean(g) / (mean_total + 1e-6)
            if green_ratio > 0.5:
                return False

        return False
    except:
        return False

# --- Sidebar Menu ---
st.sidebar.markdown('<div class="sidebar-menu-label">Brain Tumor Detection</div>', unsafe_allow_html=True)
page = st.sidebar.radio("", ["Panduan Penggunaan Aplikasi", "Deteksi Tumor", "Informasi Tumor"])

# --- Panduan Penggunaan ---
if page == "Panduan Penggunaan Aplikasi":
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<div class="menu-title">Panduan Penggunaan Aplikasi</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="instruction-box">
    <div style="text-align: justify;">
        <ul>
            <li>Siapkan gambar MRI otak dalam format JPG, JPEG, atau PNG.</li>
            <li>Pastikan gambar jelas, tidak buram, dan memiliki kualitas baik.</li>
            <li>Pilih menu <strong>Deteksi Tumor</strong> untuk analisis dan hasil prediksi.</li>
            <li>Sistem akan memprediksi dan menampilkan apakah terdapat indikasi tumor atau tidak, disertai tingkat kepercayaan dan penjelasan singkat jika tumor terdeteksi.</li>
            <li><strong>Catatan:</strong> Hasil ini hanya sebagai acuan awal. Tetap konsultasikan dengan dokter spesialis untuk kepastian medis.</li>
        </ul>
    </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Deteksi Tumor ---
elif page == "Deteksi Tumor":
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<div class="menu-title">Deteksi Tumor Otak</div>', unsafe_allow_html=True)

    confidence_threshold = 0.9
    uploaded_file = st.file_uploader("Mulai analisis dengan mengunggah gambar MRI otak", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        try:
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, caption='Gambar yang Diunggah', use_column_width=True)

            if not is_probably_mri(img):
                st.warning("⚠️ Gambar yang Anda unggah kemungkinan besar bukan MRI otak atau tidak valid. Coba unggah gambar lain.")
            else:
                img_resized = img.resize((224, 224))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction = model.predict(img_array)
                pred_index = np.argmax(prediction)
                confidence = prediction[0][pred_index]

                if confidence < confidence_threshold:
                    st.warning(f"⚠️ Prediksi tidak meyakinkan. Confidence hanya {confidence:.2f}, mohon unggah gambar lain yang lebih jelas.")
                else:
                    predicted_class = class_names[pred_index]
                    st.markdown(f'<div class="prediction-success">Jenis tumor terdeteksi: <strong>{predicted_class.upper()}</strong></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="prediction-info">Tingkat kepercayaan: <strong>{confidence:.2f}</strong></div>', unsafe_allow_html=True)

                    definitions = {
                        "glioma": "Glioma adalah tumor otak yang berasal dari sel glia, bisa jinak atau ganas, dan merupakan salah satu tumor otak primer yang paling umum.",
                        "meningioma": "Meningioma adalah tumor jinak yang tumbuh lambat di meninges (lapisan pelindung otak), dapat membesar dan menekan jaringan otak.",
                        "pituitary": "Tumor pituitary merupakan pertumbuhan abnormal di kelenjar pituitary yang umumnya jinak, namun bisa memengaruhi hormon dan fungsi saraf sekitarnya.",
                        "notumor": "Tidak terdeteksi adanya tumor otak pada hasil MRI. Meski begitu, pemeriksaan lanjutan tetap disarankan jika ada gejala."
                    }
                    st.markdown(f"""
                    <div style="text-align: justify;">
                    {definitions[predicted_class]} Untuk informasi lebih lengkap, silakan buka menu <strong>Informasi Tumor</strong>.
                    </div>
                    """, unsafe_allow_html=True)

        except UnidentifiedImageError:
            st.error("File yang diunggah bukan gambar yang valid.")
        except Exception as e:
            st.error(f"Kesalahan saat memproses gambar: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
