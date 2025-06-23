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
        color: #0f9d58;  /* hijau */
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
st.sidebar.markdown('<div class="sidebar-menu-label">Brain Tumor Detection</div>', unsafe_allow_html=True)
page = st.sidebar.radio("", ["Panduan Penggunaan Aplikasi", "Deteksi Tumor", "Informasi Tumor"])

# --- Panduan Penggunaan ---
if page == "Panduan Penggunaan Aplikasi":
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<div class="menu-title">Panduan Penggunaan Aplikasi</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="instruction-box">
    
    <ul>
        <li>Siapkan gambar MRI otak dalam format JPG, JPEG, atau PNG.</li>
        <li>Pastikan gambar jelas, tidak buram, dan memiliki kualitas baik.</li>
        <li>Pilih menu <strong>Deteksi Tumor</strong> untuk analisis dan hasil prediksi.</li>
        <li>Sistem akan memprediksi dan menampilkan apakah terdapat indikasi tumor atau tidak, disertai tingkat kepercayaan dan penjelasan singkat jika tumor terdeteksi.</li>
        <li><strong>Catatan:</strong> Hasil ini hanya sebagai acuan awal. Tetap konsultasikan dengan dokter spesialis untuk kepastian medis.</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Deteksi Tumor ---
elif page == "Deteksi Tumor":
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<div class="menu-title">Deteksi Tumor Otak</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Mulai analisis dengan mengunggah gambar MRI otak", type=["jpg", "jpeg", "png"])

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

                    definitions = {
                        "glioma": "Glioma adalah tumor otak yang berasal dari sel glia, bisa jinak atau ganas, dan merupakan salah satu tumor otak primer yang paling umum.",
                        "meningioma": "Meningioma adalah tumor jinak yang tumbuh lambat di meninges (lapisan pelindung otak), dapat membesar dan menekan jaringan otak.",
                        "pituitary": "Tumor pituitari merupakan Pertumbuhan abnormal di kelenjar pituitari yang umumnya jinak, namun bisa memengaruhi hormon dan fungsi saraf sekitarnya.",
                        "notumor": "Tidak terdeteksi adanya tumor otak pada hasil MRI. Meski begitu, pemeriksaan lanjutan tetap disarankan jika ada gejala."
                    }
                    st.info(f"{definitions[predicted_class]} Untuk informasi lebih lengkap, silakan buka menu *Informasi Tumor*.")


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
        st.markdown("""
            <div style="text-align: justify;">
            Glioma adalah jenis tumor yang tumbuh di otak dan sumsum tulang belakang yang berasal dari sel glia, yaitu sel pendukung jaringan saraf. Tumor ini bisa bersifat jinak atau ganas dan merupakan salah satu tumor otak primer yang paling umum.<br><br>
            Glioma terbagi menjadi beberapa jenis, seperti astrositoma, oligodendroglioma, dan glioblastoma, yang berbeda tingkat keganasan dan pola pertumbuhannya. Gejala glioma biasanya tergantung pada lokasi dan ukuran tumor, seperti sakit kepala, kejang, gangguan penglihatan, atau kelemahan pada bagian tubuh tertentu.<br><br>
            Diagnosis dilakukan dengan pemeriksaan pencitraan seperti CT scan atau MRI, dan terkadang konfirmasi melalui biopsi jaringan tumor. Penanganan glioma meliputi operasi pengangkatan tumor, radioterapi, dan kemoterapi, tergantung pada jenis dan tingkat keparahan tumor.
            </div>
        """, unsafe_allow_html=True)
        st.markdown("""
            <div style="text-align: justify;">
            <br><br>
            <b>Referensi:</b><br>
            <a href="https://www.ncbi.nlm.nih.gov/books/NBK441874/" target="_blank">NCBI Bookshelf - Glioma</a><br>
            <a href="https://jurnal.ar-raniry.ac.id/index.php/jurnalphi/article/download/8302/5016" target="_blank">Jurnal Phi - Ar-Raniry</a>
            </div>
        """, unsafe_allow_html=True)
        
    elif pilihan == "meningioma":
        st.markdown('<div class="menu-title">Meningioma</div>', unsafe_allow_html=True)
        st.markdown("""
            <div style="text-align: justify;">
            Meningioma adalah tumor jinak intrakranial yang tumbuh lambat dan berasal dari sel arachnoid, bagian dari meninges yang melindungi otak dan sumsum tulang belakang. 
            Meskipun bersifat jinak, tumor ini bisa tumbuh besar dan menyebabkan tekanan pada jaringan otak. 
            Tumor ini biasanya muncul tunggal, tetapi bisa juga muncul di beberapa lokasi sekaligus.<br><br>
            Gejalanya bergantung pada ukuran dan lokasi tumor, seperti sakit kepala, gangguan penglihatan, telinga berdenging, atau mual-muntah. 
            Pemeriksaan penunjang seperti CT Scan dan MRI digunakan untuk diagnosis, dan bisa dikonfirmasi melalui pemeriksaan patologi anatomi jika hasil pencitraan belum jelas.<br><br>
            Penanganan meningioma bisa berupa observasi (jika gejala minimal), operasi, radioterapi, atau terapi tambahan lain. 
            Pencegahan dilakukan dengan mengontrol faktor risiko seperti hipertensi dan diabetes, serta menjalani pola hidup sehat.
            </div>
        """, unsafe_allow_html=True)
        st.markdown("""
            <div style="text-align: justify;">
            <br><br>
            <b>Referensi:</b> <a href="https://e-journal.trisakti.ac.id/index.php/abdimastrimedika/article/view/19011" target="_blank">
            Jurnal Abdimas Trimedika - Universitas Trisakti</a>
            </div>
        """, unsafe_allow_html=True)

    elif pilihan == "pituitary":
        st.markdown('<div class="menu-title">Tumor Pituitary</div>', unsafe_allow_html=True)
        st.markdown("""
            <div style="text-align: justify;">
            Tumor pituitary adalah pertumbuhan sel abnormal yang terjadi pada kelenjar pituitari, yaitu kelenjar kecil di dasar otak yang berperan penting dalam mengatur berbagai hormon tubuh. 
            Sebagian besar tumor pituitari bersifat jinak (adenoma) dan tidak menyebar ke bagian tubuh lain, namun dapat memengaruhi produksi hormon dan menekan struktur sekitarnya sehingga menyebabkan gangguan hormonal maupun neurologis.<br><br>
            Gejala tumor ini bervariasi tergantung jenis hormon yang diproduksi atau ditekan, antara lain gangguan penglihatan, sakit kepala, perubahan siklus menstruasi, hingga gangguan pertumbuhan. 
            Diagnosis dilakukan melalui pemeriksaan pencitraan (MRI/CT) dan tes laboratorium hormon. 
            Penanganan meliputi pemberian obat, tindakan bedah, atau radioterapi, tergantung ukuran, lokasi, dan aktivitas tumor.
            </div>
        """, unsafe_allow_html=True)
        st.markdown("""
            <div style="text-align: justify;">
            <br><br>
            <b>Referensi:</b><br>
            <a href="https://ejournal.ukrida.ac.id/index.php/Meditek/article/view/1266/1383" target="_blank">
            Jurnal Meditek – Universitas Kristen Krida Wacana</a><br>
            <a href="https://www.cancer.org/cancer/types/pituitary-tumors/about/what-is-pituitary-tumor.html" target="_blank">
            American Cancer Society – What Is a Pituitary Tumor?</a>
            </div>
        """, unsafe_allow_html=True)

    elif pilihan == "notumor":
        st.markdown('<div class="menu-title">Notumor</div>', unsafe_allow_html=True)
        st.markdown("""
            <div style="text-align: justify;">
            Pada pemeriksaan MRI atau CT scan, jika tidak ditemukan adanya massa atau pertumbuhan sel abnormal di otak, maka dikatakan tidak ada tumor otak. 
            Kondisi ini menunjukkan bahwa otak dalam keadaan normal tanpa adanya tumor yang bisa mengganggu fungsi saraf atau kesehatan otak.<br><br>
            Namun, penting untuk selalu konsultasi dengan dokter atau ahli saraf untuk memastikan diagnosis dan pemantauan jika terdapat gejala yang mencurigakan.
            Pemeriksaan lanjutan mungkin diperlukan untuk memastikan penyebab gejala yang dialami jika ada.
            </div>
        """, unsafe_allow_html=True)


    st.markdown("</div>", unsafe_allow_html=True)
