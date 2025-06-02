import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model
import gdown
import os

# ==================== Konfigurasi Halaman ====================
st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

# ==================== Styling CSS ====================
with open("style.css", "w") as f:
    f.write("""
    <style>
        body, html, #root > div:nth-child(1) {
            height: 100vh;
            overflow-y: auto;
            background: #f9fafb;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #333;
            margin: 0; padding: 0;
        }
        .menu-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1a73e8;
            text-align: center;
            margin: 0.3rem 0 1.5rem 0;
            border-bottom: 2px solid #4285f4;
            padding-bottom: 1.6rem;
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
        div[data-testid="stFileUploader"] > div:first-child {
            border: 2px dashed #1a73e8 !important;
            border-radius: 12px !important;
            padding: 15px !important;
            background-color: #fff !important;
            max-width: 600px;
            margin: 0 auto 0.2rem auto; 
            transition: background-color 0.3s ease;
            user-select: none;
        }
        div[data-testid="stFileUploader"]:hover > div:first-child {
            background-color: #f1f8ff !important;
        }
        label[for="upload"] {
            font-weight: 600;
            font-size: 1.2rem;
            color: #1a73e8;
            display: block;
            text-align: center;
            margin-bottom: 0.1rem;
            user-select: none;
        }
        .prediction-box {
            background-color: #e6f4ea;
            border-left: 5px solid #188038;
            border-radius: 10px;
            padding: 1rem;
            margin-top: 1.5rem;
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            user-select: none;
        }
        .prediction-success {
            text-align: center;
            font-size: 1.2rem;
            font-weight: 700;
            color: #188038;
            margin-bottom: 0.3rem;
        }
        .prediction-info {
            text-align: center;
            font-size: 1rem;
            font-weight: 600;
            color: #155ab3;
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
    """)

with open("style.css", "r") as f:
    st.markdown(f.read(), unsafe_allow_html=True)

# ==================== Unduh Model Jika Belum Ada ====================
MODEL_PATH = 'brain_tumor_model.h5'
FILE_ID = '18lLL4vDzXS9gdDXksyJhuY5MedaafKv7'
DOWNLOAD_URL = f'https://drive.google.com/uc?id={FILE_ID}'
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

if not os.path.exists(MODEL_PATH):
    with st.spinner("Mengunduh model dari Google Drive..."):
        if not gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False):
            st.error("Gagal mengunduh model.")
            st.stop()

# ==================== Load Model ====================
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# ==================== Fungsi Validasi MRI ====================
def is_probably_mri(image_pil):
    if image_pil.width < 100 or image_pil.height < 100:
        return False
    img_np = np.array(image_pil)
    if len(img_np.shape) == 2:
        return True
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        stds = np.std(img_np, axis=(0, 1))
        ratio = stds.min() / (stds.max() + 1e-6)
        green_ratio = np.mean(img_np[:, :, 1]) / (np.mean(img_np) + 1e-6)
        return ratio > 0.9 and green_ratio < 0.5
    return False

# ==================== Sidebar Navigation ====================
st.sidebar.markdown('<div class="sidebar-menu-label">Menu</div>', unsafe_allow_html=True)
page = st.sidebar.radio("", ["Home", "Tumor Info"])

# ==================== Halaman Home ====================
if page == "Home":
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<div class="menu-title">Brain Tumor Detection</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="instruction-box">
    <h4>📌 Tata Cara Penggunaan:</h4>
    <ol>
        <li>Unggah gambar MRI otak (JPG/JPEG/PNG).</li>
        <li>Pastikan gambar jelas dan terlihat struktur otaknya.</li>
        <li>Model akan otomatis menganalisis dan memprediksi jenis tumor.</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<label for="upload">Upload Gambar MRI</label>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="upload")

    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Gambar yang Diunggah", use_column_width=True)

            if not is_probably_mri(image):
                st.warning("Gambar tidak terdeteksi sebagai MRI otak.")
            else:
                resized = image.resize((224, 224))
                array = np.expand_dims(np.array(resized) / 255.0, axis=0)
                pred = model.predict(array)
                pred_idx = np.argmax(pred)
                confidence = pred[0][pred_idx]

                if confidence < 0.6:
                    st.warning("Prediksi kurang yakin. Silakan coba gambar lain.")
                else:
                    st.markdown(f"""
                    <div class="prediction-box">
                        <div class="prediction-success">Jenis tumor: <strong>{CLASS_NAMES[pred_idx].upper()}</strong></div>
                        <div class="prediction-info">Tingkat kepercayaan: {confidence:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)

        except UnidentifiedImageError:
            st.error("File bukan gambar yang valid.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses gambar: {e}")

# ==================== Halaman Informasi Tumor ====================
elif page == "Tumor Info":
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<div class="menu-title">Informasi Jenis Tumor Otak</div>', unsafe_allow_html=True)

    pilihan = st.selectbox("Pilih jenis tumor untuk informasi:", CLASS_NAMES)

    if pilihan == "glioma":
        st.subheader("Glioma")
        st.write("Tumor berasal dari sel glial. Bisa jinak atau ganas, dan umumnya tumbuh cepat.")

    elif pilihan == "meningioma":
        st.subheader("Meningioma")
        st.write("""Meningioma adalah tumor jinak yang berasal dari sel arachnoid, bagian dari meninges.
        Meski jinak, ukurannya bisa membesar dan menyebabkan tekanan di otak. Biasanya tumbuh lambat.""")

    elif pilihan == "notumor":
        st.subheader("Tidak Ada Tumor")
        st.write("Model tidak mendeteksi keberadaan tumor pada citra MRI yang diberikan.")

    elif pilihan == "pituitary":
        st.subheader("Pituitary Tumor")
        st.write("Tumor yang berasal dari kelenjar pituitari. Bisa memengaruhi hormon dan fungsi tubuh lainnya.")
