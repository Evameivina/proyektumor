import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model
import gdown
import os

# Page config
st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

# Custom CSS styling sidebar dan halaman
st.markdown("""
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
    .image-caption {
        text-align: center;
        font-size: 0.85rem;
        color: #555;
        font-style: italic;
        margin-top: 0.3rem;
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
        display: flex;
        flex-direction: column;
        align-items: center;  
    }
    .stRadio > div > div {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
        user-select: none;
    }
    .stRadio > div > div > label {
        margin-left: 0.5rem;
        cursor: pointer;
        font-weight: 600;
        color: #1a73e8;
    }
</style>
""", unsafe_allow_html=True)

# Path dan ID file model di Google Drive
model_path = 'brain_tumor_model.h5'
file_id = '18lLL4vDzXS9gdDXksyJhuY5MedaafKv7'
download_url = f'https://drive.google.com/uc?id={file_id}'
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Unduh model jika belum tersedia
if not os.path.exists(model_path):
    with st.spinner('Mengunduh model...'):
        downloaded = gdown.download(download_url, model_path, quiet=False)
        if not downloaded:
            st.error("Gagal mengunduh model dari Google Drive.")
            st.stop()

# Load model dengan penanganan error
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# Fungsi sederhana untuk mendeteksi apakah gambar kemungkinan MRI (grayscale atau sedikit warna)
def is_probably_mri(image_pil):
    if image_pil.width < 100 or image_pil.height < 100:
        return False
    img_np = np.array(image_pil)
    if len(img_np.shape) == 2:  # grayscale
        return True
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        stds = np.std(img_np, axis=(0,1))
        ratio = stds.min() / (stds.max() + 1e-6)
        if ratio > 0.9:
            return True
        green_ratio = np.mean(img_np[:,:,1]) / (np.mean(img_np) + 1e-6)
        if green_ratio > 0.5:
            return False
    return True

# Sidebar menu
st.sidebar.markdown('<div class="sidebar-menu-label">Menu</div>', unsafe_allow_html=True)
page = st.sidebar.radio("", ["Home", "Tumor Info"])

# ========== Halaman Home ==========
if page == "Home":
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<div class="menu-title">Brain Tumor Detection</div>', unsafe_allow_html=True)

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

    if uploaded_file:
    try:
        img = Image.open(uploaded_file).convert('RGB')
        # Bungkus gambar dengan div flex agar center
        st.markdown(
            '''
            <div style="display: flex; justify-content: center; margin-top: 1rem; margin-bottom: 1rem;">
            ''', unsafe_allow_html=True)
        st.image(img, caption='Gambar yang Diunggah', width=400)
        st.markdown('</div>', unsafe_allow_html=True)

            if not is_probably_mri(img):
                st.warning("Gambar yang diunggah kemungkinan bukan citra MRI otak.")
            else:
                img_resized = img.resize((224, 224))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction = model.predict(img_array)
                pred_index = np.argmax(prediction)
                confidence = prediction[0][pred_index]

                if confidence < 0.6:
                    st.warning("Model tidak yakin dengan prediksi. Silakan coba gambar lain.")
                else:
                    predicted_class = class_names[pred_index]
                    st.markdown(f"""
                        <div class="prediction-box">
                            <div class="prediction-success">Jenis tumor terdeteksi: <strong>{predicted_class.upper()}</strong></div>
                            <div class="prediction-info">Tingkat kepercayaan: <strong>{confidence:.2f}</strong></div>
                        </div>
                    """, unsafe_allow_html=True)

        except UnidentifiedImageError:
            st.error("File yang diunggah bukan gambar yang valid.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses gambar: {e}")

# ========== Halaman Informasi Tumor ==========
elif page == "Tumor Info":
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<div class="menu-title">Informasi Jenis Tumor Otak</div>', unsafe_allow_html=True)

    pilihan = st.selectbox("Pilih jenis tumor untuk informasi:", class_names)

    if pilihan == "glioma":
        st.markdown('<div class="menu-title">Glioma</div>', unsafe_allow_html=True)
        st.write("""
        Glioma adalah jenis tumor otak yang berasal dari sel glial.
        Tumor ini bisa bersifat ganas atau jinak, tergantung subtipenya.
        Gejala umum termasuk sakit kepala, kejang, dan gangguan neurologis.
        Penanganan meliputi operasi, kemoterapi, dan radioterapi.
        """)
        
    # ========== Halaman Informasi Tumor ==========
elif page == "Tumor Info":
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<div class="menu-title">Informasi Jenis Tumor Otak</div>', unsafe_allow_html=True)

    pilihan = st.selectbox("Pilih jenis tumor untuk informasi:", class_names)

    if pilihan == "glioma":
        st.markdown('<div class="menu-title">Glioma</div>', unsafe_allow_html=True)
        st.write("""
        Glioma adalah jenis tumor otak yang berasal dari sel glial.
        Tumor ini bisa bersifat ganas atau jinak, tergantung subtipenya.
        Gejala umum termasuk sakit kepala, kejang, dan gangguan neurologis.
        Penanganan meliputi operasi, kemoterapi, dan radioterapi.
        """)

    elif pilihan == "meningioma":
        st.markdown('<div class="menu-title">Meningioma</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="text-align: justify;">
            Meningioma adalah tumor jinak intrakranial yang tumbuh lambat dan berasal dari sel arachnoid, yaitu bagian dari meninges yang melindungi otak dan sumsum tulang belakang. 
            Meski bersifat jinak, tumor ini dapat membesar dan berpotensi mengancam jiwa. Meningioma ganas sering dikaitkan dengan mutasi kromosom yang mempercepat pertumbuhan tumor. 
            Biasanya muncul tunggal, namun bisa juga ditemukan di beberapa lokasi secara bersamaan.<br><br>
            Gejala klinis meningioma seringkali tidak jelas, kecuali bila tumor sudah berukuran cukup besar, karena pertumbuhannya yang lambat.
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <div style="text-align: justify;">
            <b>Referensi:</b> <a href="https://e-journal.trisakti.ac.id/index.php/abdimastrimedika/article/view/19011" target="_blank">
            Jurnal Abdimas Trimedika - Universitas Trisakti</a>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif pilihan == "notumor":
        st.markdown('<div class="menu-title">Tidak Ada Tumor</div>', unsafe_allow_html=True)
        st.write("Gambar menunjukkan kondisi otak tanpa tumor yang terdeteksi.")

    elif pilihan == "pituitary":
        st.markdown('<div class="menu-title">Pituitary Tumor</div>', unsafe_allow_html=True)
        st.write("""
        Tumor pituitari tumbuh pada kelenjar pituitari di dasar otak.
        Bisa menyebabkan gangguan hormon dan masalah kesehatan lain.
        Penanganan dapat meliputi operasi, terapi hormon, dan radioterapi.
        """)

    st.markdown('</div>', unsafe_allow_html=True)
