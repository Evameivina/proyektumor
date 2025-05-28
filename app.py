import streamlit as st  
import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model
import gdown
import os

# Page config
st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

# Minimalist & elegant CSS styling
st.markdown("""
<style>
    /* Body & root container */
    body, html, #root > div:nth-child(1) {
        height: 100vh;
        overflow-y: auto;
        background: #f9fafb;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #333;
    }

    /* Title */
    .menu-title {
        font-size: 3rem;
        font-weight: 700;
        color: #1a73e8;
        text-align: center;
        margin: 1rem 0 2rem 0;
        border-bottom: 3px solid #4285f4;
        padding-bottom: 0.5rem;
        user-select: none;
    }

    /* Instruction box */
    .instruction-box {
        background-color: #e8f0fe;
        border-left: 5px solid #1a73e8;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        max-width: 650px;
        margin: 0 auto 2.5rem auto;
        font-size: 1.1rem;
        line-height: 1.5;
        color: #202124;
        user-select: none;
    }

    /* Upload container */
    div[data-testid="stFileUploader"] > div:first-child {
        border: 2.5px dashed #1a73e8 !important;
        border-radius: 15px !important;
        padding: 25px 20px !important;
        background-color: #ffffff !important;
        max-width: 600px;
        margin: 0 auto 1.5rem auto;
        transition: background-color 0.3s ease;
        user-select: none;
    }
    div[data-testid="stFileUploader"]:hover > div:first-child {
        background-color: #f1f8ff !important;
    }

    /* Upload label */
    label[for="upload"] {
        font-weight: 600;
        font-size: 1.3rem;
        color: #1a73e8;
        display: block;
        text-align: center;
        margin-bottom: 0.4rem;
        user-select: none;
    }

    /* Image caption */
    .image-caption {
        text-align: center;
        font-size: 0.9rem;
        color: #555;
        font-style: italic;
        margin-top: 0.5rem;
        user-select: none;
    }

    /* Prediction results */
    .prediction-success {
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        color: #188038;
        margin-top: 1.2rem;
        user-select: none;
    }
    .prediction-info {
        text-align: center;
        font-size: 1.1rem;
        font-weight: 600;
        color: #155ab3;
        margin-top: 0.4rem;
        user-select: none;
    }

    /* Sidebar menu label */
    .sidebar-menu-label {
        font-weight: 700;
        font-size: 1.6rem;
        color: #1a73e8;
        margin-bottom: 1rem;
        padding-left: 15px;
        user-select: none;
    }

    /* Main container */
    .main {
        max-width: 850px;
        margin: 0 auto 3rem auto;
        padding: 0 20px;
    }

    /* Spacing between radio buttons in sidebar */
    .css-1n76uvr {
        margin-bottom: 0.7rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Download dan load model
model_path = 'brain_tumor_model.h5'
file_id = '18lLL4vDzXS9gdDXksyJhuY5MedaafKv7'
url = f'https://drive.google.com/uc?id={file_id}'
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

if not os.path.exists(model_path):
    with st.spinner('Mengunduh model...'):
        success = gdown.download(url, model_path, quiet=False)
        if not success:
            st.error("Gagal mengunduh model dari Google Drive.")
            st.stop()

try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

def is_probably_mri(image_pil):
    if image_pil.width < 100 or image_pil.height < 100:
        return False
    img_np = np.array(image_pil)
    if len(img_np.shape) == 2:
        return True
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        r, g, b = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]
        std_r, std_g, std_b = np.std(r), np.std(g), np.std(b)
        ratio = min(std_r, std_g, std_b) / (max(std_r, std_g, std_b) + 1e-6)
        if ratio > 0.9:
            return True
        mean_total = np.mean(img_np)
        green_ratio = np.mean(g) / (mean_total + 1e-6)
        if green_ratio > 0.5:
            return False
    return True

# Sidebar menu with clear label
st.sidebar.markdown('<div class="sidebar-menu-label">Menu</div>', unsafe_allow_html=True)
page = st.sidebar.radio("", ["Home", "Tumor Info"])

# =============== HOME PAGE ===============
if page == "Home":
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<div class="menu-title">Brain Tumor Detection dari Citra MRI</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="instruction-box">
    <h4>📌 Tata Cara Penggunaan:</h4>
    <ol>
        <li>Siapkan <strong>gambar MRI otak</strong> (format JPG, JPEG, atau PNG).</li>
        <li>Pastikan gambar jelas dan memperlihatkan struktur otak.</li>
        <li>Klik <em>"Browse files"</em> atau seret gambar ke kotak unggah.</li>
        <li>Sistem akan otomatis memeriksa validitas gambar.</li>
        <li>Model akan memprediksi jenis tumor jika ditemukan.</li>
        <li>Hasil prediksi akan menampilkan jenis tumor dan tingkat kepercayaan.</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<label for="upload">Upload Gambar MRI</label>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="upload")

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Gambar yang Diunggah', use_column_width=True)

            if not is_probably_mri(image):
                st.warning("Gambar yang diunggah kemungkinan bukan citra MRI otak.")
            else:
                img = image.resize((224, 224))
                img_array = np.array(img) / 255.0
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
            st.error("File bukan gambar yang valid.")
        except Exception as e:
            st.error(f"Kesalahan saat memproses gambar: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# =============== TUMOR INFO PAGE ===============
elif page == "Tumor Info":
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<div class="menu-title">Informasi Jenis Tumor Otak</div>', unsafe_allow_html=True)

    pilihan = st.selectbox("Pilih jenis tumor untuk informasi:", class_names)

    if pilihan == "glioma":
        st.markdown('<div class="menu-title">Glioma</div>', unsafe_allow_html=True)
        st.write("Tumor berasal dari sel glial. Bisa jinak atau ganas, dan umumnya tumbuh cepat.")
    elif pilihan == "meningioma":
        st.markdown('<div class="menu-title">Meningioma</div>', unsafe_allow_html=True)
        st.write("Tumor dari meninges, biasanya jinak, tapi dapat menekan otak tergantung lokasi.")
    elif pilihan == "pituitary":
        st.markdown('<div class="menu-title">Tumor Pituitari</div>', unsafe_allow_html=True)
        st.write("Tumbuh di kelenjar pituitari yang mengatur hormon. Bisa mengganggu keseimbangan hormon.")
    elif pilihan == "notumor":
        st.markdown('<div class="menu-title">Tidak Ada Tumor</div>', unsafe_allow_html=True)
        st.write("Tidak ditemukan tumor pada gambar MRI. Selalu konsultasi dengan dokter untuk hasil pasti.")

    st.markdown("</div>", unsafe_allow_html=True)
