import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model
import gdown
import os

# Konfigurasi halaman
st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

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
        std_r = np.std(r)
        std_g = np.std(g)
        std_b = np.std(b)
        max_std = max(std_r, std_g, std_b)
        min_std = min(std_r, std_g, std_b)
        ratio = min_std / (max_std + 1e-6)

        if ratio > 0.9:
            return True

        mean_total = np.mean(img_np)
        green_ratio = np.mean(g) / (mean_total + 1e-6)
        if green_ratio > 0.5:
            return False

    return True

# Sidebar navigasi
menu = st.sidebar.radio("Menu", ["Home", "Tumor Info"])

# ==================== HOME ====================
if menu == "Home":
    st.title("Brain Tumor Detection from MRI Image")
    st.markdown("""
    ### 📌 Tata Cara Penggunaan:
    1. Siapkan **gambar MRI otak** dalam format **JPG, JPEG, atau PNG**.
    2. Pastikan gambar terlihat **jelas** dan menampilkan struktur otak dengan baik.
    3. Klik tombol **"Browse files"** atau **seret dan jatuhkan** gambar ke area unggahan di bawah.
    4. Sistem akan memeriksa apakah gambar tersebut valid sebagai citra MRI otak.
    5. Jika valid, model akan memprediksi **jenis tumor otak** (atau menyatakan tidak ada tumor).
    6. Hasil akan menampilkan **jenis tumor yang terdeteksi** dan nilai **tingkat kepercayaan (confidence)**.
    """)

    uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            if not is_probably_mri(image):
                st.warning("Gambar tidak dikenali sebagai MRI otak.")
            else:
                img = image.resize((224, 224))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction = model.predict(img_array)
                pred_index = np.argmax(prediction)
                confidence = prediction[0][pred_index]

                if confidence < 0.6:
                    st.warning("Model tidak cukup yakin dengan hasil prediksi.")
                else:
                    predicted_class = class_names[pred_index]
                    st.success(f"Jenis tumor terdeteksi: {predicted_class.upper()}")
                    st.write(f"Tingkat Kepercayaan: {confidence:.2f}")

        except UnidentifiedImageError:
            st.error("File yang diunggah bukan gambar yang valid.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses gambar: {e}")

# ==================== TUMOR INFO ====================
elif menu == "Tumor Info":
    st.title("Brain Tumor Information")
    pilihan = st.selectbox("Select tumor type to read more:", class_names)

    if pilihan == "glioma":
        st.subheader("Glioma")
        st.write("""
            Glioma adalah tumor yang berasal dari sel glial di otak atau sumsum tulang belakang.
            Dapat bersifat jinak maupun ganas. Banyak kasus bersifat agresif dan tumbuh cepat.
        """)
    elif pilihan == "meningioma":
        st.subheader("Meningioma")
        st.write("""
            Meningioma tumbuh dari meninges, lapisan pelindung otak dan sumsum tulang belakang.
            Umumnya jinak dan tumbuh lambat, namun dapat menyebabkan gejala tergantung lokasi.
        """)
    elif pilihan == "pituitary":
        st.subheader("Pituitary Tumor")
        st.write("""
            Tumor ini tumbuh di kelenjar pituitari yang berperan dalam produksi hormon.
            Dapat menyebabkan gangguan hormonal dan memengaruhi banyak fungsi tubuh.
        """)
    elif pilihan == "notumor":
        st.subheader("No Tumor Detected")
        st.write("""
            Tidak ditemukan tumor pada citra MRI otak yang diunggah.
            Namun, hasil ini tidak menggantikan diagnosis dari tenaga medis profesional.
        """)
