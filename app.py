import streamlit as st  
import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model
import gdown
import os

# Page config
st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

# CSS styling
st.markdown("""
<style>
    /* Memperbesar judul utama */
    .menu-title {
        font-size: 48px !important;   /* lebih besar dari 34px */
        font-weight: 900;
        color: #0077b6;
        margin-bottom: 20px;
        user-select: none;
        border-bottom: 4px solid #00b4d8;
        padding-bottom: 10px;
        text-align: center;
        margin-top: 0px !important;   /* hilangkan jarak atas */
    }

    /* Kontainer utama agar tinggi maksimal 100vh (tinggi layar) dan tidak scroll */
    .main {
        max-width: 800px;
        margin: 0 auto;
        padding: 15px 25px 25px 25px;
        border-radius: 15px;
        min-height: 100vh;          /* supaya tinggi penuh layar */
        overflow: hidden;           /* hilangkan scroll vertikal di kontainer */
        box-sizing: border-box;
    }

    /* Hilangkan scroll utama di body dan html */
    html, body, #root, .appview-container, .main {
        overflow-y: hidden !important;
        height: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Instruksi kotak tetap bagus */
    .instruction-box {
        background-color: #caf0f8;
        border-left: 6px solid #023e8a;
        padding: 12px 20px;
        border-radius: 12px;
        font-size: 16px;
        line-height: 1.4;
        margin-bottom: 25px;
        user-select: none;
    }

    /* Styling uploader */
    div[data-testid="stFileUploader"] > div:first-child {
        border: 3px dashed #0077b6 !important;
        border-radius: 15px !important;
        padding: 20px 15px !important;
        background-color: #e0f7fa !important;
        transition: background-color 0.3s ease;
        user-select: none;
        max-width: 600px;
        margin: 0 auto;
    }

    label[for="upload"] {
        font-weight: 700;
        font-size: 22px;
        color: #0077b6;
        margin-bottom: 6px;
        display: block;
        text-align: center;
        user-select: none;
    }

    .image-caption {
        font-size: 14px;
        color: #444;
        text-align: center;
        margin-top: 8px;
        font-style: italic;
        user-select: none;
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

# Sidebar menu dengan label "Menu" yang rapi dan ke kiri
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
                    st.markdown(f'<div class="prediction-success">✅ Jenis tumor terdeteksi: <strong>{predicted_class.upper()}</strong></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="prediction-info">🔎 Tingkat kepercayaan: <strong>{confidence:.2f}</strong></div>', unsafe_allow_html=True)

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
