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

@st.cache_resource
def load_cached_model():
    try:
        return load_model(model_path)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

model = load_cached_model()


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
            <li>Silakan unggah <strong>satu</strong> gambar MRI otak dalam format JPG, JPEG, atau PNG. Sistem ini hanya menerima satu gambar untuk setiap proses deteksi</li>
            <li>Pastikan gambar jelas, tidak buram, dan memiliki kualitas baik</li>
            <li>Pilih menu <strong>Deteksi Tumor</strong> untuk memulai analisis dan melihat hasil prediksi</li>
            <li>Sistem akan memprediksi dan menampilkan apakah terdapat indikasi tumor atau tidak, disertai tingkat kepercayaan dan penjelasan singkat jika tumor terdeteksi.</li>
            <li>Jika gambar sesuai dan dikenali oleh sistem, maka hasil deteksi akan ditampilkan. Jika tidak, sistem mungkin tidak dapat memprosesnya dengan tepat</li>
            <li><strong>Catatan:</strong> Hasil ini hanya sebagai referensi awal berbasis kecerdasan buatan. Untuk diagnosis medis yang valid, tetap konsultasikan dengan dokter spesialis.</li>
        </ul>
    </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Deteksi Tumor ---
elif page == "Deteksi Tumor":
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<div class="menu-title">Deteksi Tumor Otak</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Mulai analisis dengan mengunggah satu gambar MRI otak (JPG, JPEG, atau PNG)",
        type=["jpg", "jpeg", "png"]
    )

    def apply_temperature(probabilities, temperature=2.0):
        logits = np.log(probabilities + 1e-8)
        scaled_logits = logits / temperature
        exp_logits = np.exp(scaled_logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    if uploaded_file:
        try:
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, caption='Gambar yang Diunggah', use_container_width=True)

            if not is_probably_mri(img):
                st.warning("⚠️ Gambar yang diunggah tidak valid dan tidak dapat di deteksi")
            else:
                img_resized = img.resize((224, 224))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction = model.predict(img_array)
                scaled_pred = apply_temperature(prediction, temperature=2.0)
                pred_index = np.argmax(scaled_pred)
                confidence = float(scaled_pred[0][pred_index])

                if confidence < 0.6:
                    st.warning(f"⚠️ Prediksi tidak meyakinkan (Confidence: {confidence:.2f}). Silakan coba unggah gambar lain yang lebih jelas.")
                else:
                    predicted_class = class_names[pred_index]
                    st.markdown(f'<div class="prediction-success">Jenis tumor terdeteksi: <strong>{predicted_class.upper()}</strong></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="prediction-info">Tingkat kepercayaan: <strong>{confidence:.2f}</strong></div>', unsafe_allow_html=True)

                    explanations = {
                        "glioma": "Glioma adalah tumor otak yang berasal dari sel glia. Bisa bersifat jinak maupun ganas, dan merupakan salah satu jenis yang paling umum ditemukan.",
                        "meningioma": "Meningioma adalah tumor jinak yang tumbuh di selaput pelindung otak. Umumnya tumbuh lambat, tetapi bisa menekan jaringan otak.",
                        "pituitary": "Tumor pituitary muncul di kelenjar pituitari dan dapat memengaruhi produksi hormon tubuh. Sebagian besar bersifat jinak.",
                        "notumor": "Tidak ditemukan indikasi adanya tumor otak pada gambar yang diunggah. Namun, tetap disarankan konsultasi ke dokter jika ada gejala."
                    }

                    st.markdown(f"""
                    <div style="text-align: justify; margin-top: 1.2rem;">
                        <strong>Tingkat Kepercayaan (Confidence)</strong><br>
                        Nilai ini menunjukkan seberapa yakin model terhadap hasil prediksi, berdasarkan kemiripan gambar MRI yang diunggah dengan data pelatihan.
                        Semakin tinggi nilainya (maksimal 1.00), semakin besar keyakinan model terhadap hasil tersebut.
                        Hasil ini bersifat prediktif dan tidak menggantikan diagnosis medis resmi.
                        <br><br>
                        <strong>Penjelasan Singkat:</strong><br>
                        {explanations[predicted_class]}
                        Untuk informasi lebih lengkap, silakan buka menu <strong>Informasi Tumor</strong>.
                    </div>
                    """, unsafe_allow_html=True)


        except UnidentifiedImageError:
            st.error("⚠️ File yang diunggah bukan gambar yang valid.")
        except Exception as e:
            st.error(f"⚠️ Kesalahan saat memproses gambar: {e}")

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
        st.markdown('<div class="menu-title">Pituitary</div>', unsafe_allow_html=True)
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

