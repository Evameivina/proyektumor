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
st.markdown("""
    <style>
        .instructions-box {
            background-color: #d4f0f9;
            padding: 10px 18px;
            border-radius: 8px;
            margin-bottom: 10px;
            font-size: 16px;
        }

        /* Hapus margin atas bawaan Streamlit */
        section.main > div {padding-top: 10px;}
    </style>

    <div class="instructions-box">
        <ol>
            <li>Siapkan <b>gambar MRI otak</b> (format JPG, JPEG, atau PNG).</li>
            <li>Pastikan gambar jelas dan memperlihatkan struktur otak.</li>
            <li>Klik <i>"Browse files"</i> atau seret gambar ke kotak unggah.</li>
            <li>Sistem akan otomatis memeriksa validitas gambar.</li>
            <li>Model akan memprediksi jenis tumor jika ditemukan.</li>
            <li>Hasil prediksi akan menampilkan jenis tumor dan tingkat kepercayaan.</li>
        </ol>
    </div>
""", unsafe_allow_html=True)

# Gunakan ini agar label "Upload Gambar MRI" langsung jadi satu bagian dengan uploader
uploaded_file = st.file_uploader("### Upload Gambar MRI", type=["jpg", "jpeg", "png"])

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

# Sidebar menu dengan label "Menu" yang rapi dan bold
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
