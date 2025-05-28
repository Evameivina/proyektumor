import streamlit as st  
import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model
import gdown
import os

# Konfigurasi halaman
st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

# Tambahkan CSS untuk mempercantik tampilan
st.markdown("""
    <style>
        .main {
            background-color: #f4f9fd;
            padding: 30px 40px 40px 40px;
            border-radius: 12px;
            max-width: 800px;
            margin: auto;
            box-shadow: 0 0 15px rgba(0,0,0,0.1);
        }
        .menu-title {
            font-size: 36px;
            font-weight: 700;
            color: #005f73;
            margin-bottom: 30px;
            text-align: center;
        }
        .feature-title {
            font-size: 26px;
            font-weight: 600;
            color: #0a9396;
            margin-top: 40px;
        }
        .instruction-box {
            background-color: #e0f7fa;
            border-left: 6px solid #0288d1;
            padding: 20px 25px;
            border-radius: 10px;
            font-size: 16px;
            line-height: 1.5;
            margin-bottom: 40px;
        }
        /* Styling drag and drop box */
        div[data-testid="stFileUploader"] > div:first-child {
            border: 3px dashed #0288d1 !important;
            border-radius: 12px !important;
            padding: 40px !important;
            background-color: #ffffffcc !important;
            transition: background-color 0.3s ease;
        }
        div[data-testid="stFileUploader"] > div:first-child:hover {
            background-color: #b2ebf2cc !important;
        }
        /* Upload label styling */
        label[for="upload"] {
            font-weight: 600;
            font-size: 22px;
            color: #005f73;
            display: block;
            margin-bottom: 15px;
            text-align: center;
        }
        /* Image preview caption */
        .image-caption {
            font-size: 14px;
            color: #555;
            text-align: center;
            margin-top: 10px;
            font-style: italic;
        }
    </style>
""", unsafe_allow_html=True)

# Path dan URL model
model_path = 'brain_tumor_model.h5'
file_id = '18lLL4vDzXS9gdDXksyJhuY5MedaafKv7'
url = f'https://drive.google.com/uc?id={file_id}'
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Unduh model jika belum ada
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

# Sidebar pilihan halaman tanpa judul
menu = st.sidebar.radio("", ["🏠 Home", "📚 Tumor Info"])

# ==================== HOME ====================
if menu == "🏠 Home":
    st.markdown("<div class='main'>", unsafe_allow_html=True)
    st.markdown("<div class='menu-title'>Brain Tumor Detection dari Citra MRI</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="instruction-box">
    <h4>📌 Tata Cara Penggunaan:</h4>
    <ol>
        <li>Siapkan <strong>gambar MRI otak</strong> (format JPG, JPEG, atau PNG).</li>
        <li>Pastikan gambar <strong>jelas</strong> dan menunjukkan struktur otak.</li>
        <li>Klik <em>"Browse files"</em> atau seret gambar ke kolom unggah.</li>
        <li>Sistem akan mengecek apakah gambar valid.</li>
        <li>Jika valid, model akan memprediksi <strong>jenis tumor</strong> (jika ada).</li>
        <li>Hasil menampilkan <strong>jenis tumor</strong> dan <strong>tingkat kepercayaan</strong>.</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<label for="upload">📤 Upload Gambar MRI</label>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="upload")

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='🖼️ Gambar yang Diunggah', use_column_width=True)
            
            if not is_probably_mri(image):
                st.warning("⚠️ Gambar tidak dikenali sebagai MRI otak.")
            else:
                img = image.resize((224, 224))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction = model.predict(img_array)
                pred_index = np.argmax(prediction)
                confidence = prediction[0][pred_index]

                if confidence < 0.6:
                    st.warning("❗ Model tidak cukup yakin dengan hasil prediksi.")
                else:
                    predicted_class = class_names[pred_index]
                    st.success(f"✅ Jenis tumor terdeteksi: **{predicted_class.upper()}**")
                    st.info(f"🔍 Tingkat Kepercayaan: **{confidence:.2f}**")

        except UnidentifiedImageError:
            st.error("🚫 File yang diunggah bukan gambar yang valid.")
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat memproses gambar: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# ==================== TUMOR INFO ====================
elif menu == "📚 Tumor Info":
    st.markdown("<div class='main'>", unsafe_allow_html=True)
    st.markdown("<div class='menu-title'>Informasi Jenis Tumor Otak</div>", unsafe_allow_html=True)

    pilihan = st.selectbox("🧬 Pilih jenis tumor untuk informasi lebih lanjut:", class_names)

    if pilihan == "glioma":
        st.markdown("<div class='feature-title'>🧠 Glioma</div>", unsafe_allow_html=True)
        st.write("Tumor berasal dari sel glial. Bisa jinak atau ganas, dan umumnya tumbuh cepat.")
    elif pilihan == "meningioma":
        st.markdown("<div class='feature-title'>🧠 Meningioma</div>", unsafe_allow_html=True)
        st.write("Berasal dari meninges, biasanya jinak namun bisa menekan bagian otak tergantung lokasi.")
    elif pilihan == "pituitary":
        st.markdown("<div class='feature-title'>🧠 Tumor Pituitari</div>", unsafe_allow_html=True)
        st.write("Tumbuh di kelenjar pituitari yang mengatur hormon. Bisa menyebabkan gangguan hormonal.")
    elif pilihan == "notumor":
        st.markdown("<div class='feature-title'>🧠 Tidak Ada Tumor</div>", unsafe_allow_html=True)
        st.write("Tidak ditemukan tumor dalam gambar MRI. Namun, selalu konsultasikan hasil ini dengan profesional medis.")

    st.markdown("</div>", unsafe_allow_html=True)
