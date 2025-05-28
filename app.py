import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model
import gdown
import os

# Page config dengan layout wide dan disable menu/sidebar biar maksimal
st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

# CSS styling no scroll, compact, rapat tanpa margin besar
st.markdown("""
    <style>
        /* Reset margin/padding dan buat container penuh tanpa scroll */
        html, body, #root > div:nth-child(1), .main, .block-container {
            height: 100vh !important;  /* tinggi penuh viewport */
            margin: 0; padding: 0;
            overflow: hidden;  /* hilangkan scroll */
            background-color: #f9fcff;
        }
        /* Container utama dengan padding minimal */
        .main {
            max-width: 800px;
            margin: 10px auto 10px auto !important;
            padding: 10px 20px 20px 20px !important;
            border-radius: 15px;
        }
        /* Hilangkan semua padding/margin dari app container */
        [data-testid="stAppViewContainer"],
        [data-testid="stVerticalBlock"],
        .css-18e3th9 {
            padding: 0 !important;
            margin: 0 !important;
        }
        /* Judul */
        .menu-title {
            font-size: 26px;
            font-weight: 700;
            color: #0077b6;
            margin-bottom: 15px;
            border-bottom: 3px solid #00b4d8;
            padding-bottom: 6px;
            user-select: none;
        }
        /* Instruction box */
        .instruction-box {
            background-color: #caf0f8;
            border-left: 6px solid #023e8a;
            padding: 12px 15px;
            border-radius: 12px;
            font-size: 14px;
            line-height: 1.3;
            margin-bottom: 20px;
            max-height: 170px;
            overflow-y: auto; /* kalau panjang, scroll di box saja */
        }
        /* File uploader */
        div[data-testid="stFileUploader"] > div:first-child {
            border: 3px dashed #0077b6 !important;
            border-radius: 15px !important;
            padding: 20px 10px !important;
            background-color: #e0f7fa !important;
            max-width: 600px;
            margin: 0 auto;
            height: 120px;  /* fixed height supaya compact */
            display: flex;
            align-items: center;
            justify-content: center;
        }
        /* Label uploader rapat ke uploader */
        label[for="upload"] {
            font-weight: 700;
            font-size: 20px;
            color: #0077b6;
            margin-bottom: 4px;
            display: block;
            text-align: center;
            user-select: none;
        }
        /* Caption gambar */
        .image-caption {
            font-size: 14px;
            color: #444;
            text-align: center;
            margin-top: 8px;
            font-style: italic;
            user-select: none;
        }
        /* Gambar yang diupload dibatasi tinggi */
        .uploaded-image {
            max-height: 300px;
            width: auto;
            margin: 0 auto;
            display: block;
        }
    </style>
""", unsafe_allow_html=True)

# Load model dan setup (sama dengan yang kamu buat sebelumnya)

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
    # sama dengan fungsi kamu

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

# Sidebar dan page layout

st.sidebar.markdown('<div class="sidebar-menu-label">Menu</div>', unsafe_allow_html=True)
page = st.sidebar.radio("", ["Home", "Tumor Info"])

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
            # tampilkan gambar dengan kelas css agar max heightnya terjaga
            st.markdown(f'<img src="data:image/png;base64,{st.image_to_bytes(image)}" class="uploaded-image" />', unsafe_allow_html=True)
            # fallback biasa:
            st.image(image, caption='Gambar yang Diunggah', use_column_width=False, width=600)

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
                    st.success(f"Jenis tumor terdeteksi: **{predicted_class.upper()}**")
                    st.info(f"Tingkat kepercayaan: **{confidence:.2f}**")

        except UnidentifiedImageError:
            st.error("File bukan gambar yang valid.")
        except Exception as e:
            st.error(f"Kesalahan saat memproses gambar: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Tumor Info":
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<div class="menu-title">Informasi Jenis Tumor Otak</div>', unsafe_allow_html=True)

    pilihan = st.selectbox("Pilih jenis tumor untuk informasi:", class_names)

    if pilihan == "glioma":
        st.markdown('<div class="feature-title"> Glioma</div>', unsafe_allow_html=True)
        st.write("Tumor berasal dari sel glial. Bisa jinak atau ganas, dan umumnya tumbuh cepat.")
    elif pilihan == "meningioma":
        st.markdown('<div class="feature-title"> Meningioma</div>', unsafe_allow_html=True)
        st.write("Tumor dari meninges, biasanya jinak, tapi dapat menekan otak tergantung lokasi.")
    elif pilihan == "pituitary":
        st.markdown('<div class="feature-title"> Pituitary Tumor</div>', unsafe_allow_html=True)
        st.write("Tumor di kelenjar pituitari, dapat memengaruhi hormon tubuh.")
    else:
        st.markdown('<div class="feature-title"> Tidak Ada Tumor</div>', unsafe_allow_html=True)
        st.write("Citra MRI tidak menunjukkan adanya tumor.")

    st.markdown("</div>", unsafe_allow_html=True)
