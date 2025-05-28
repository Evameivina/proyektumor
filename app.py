import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model
import gdown
import os
import tensorflow as tf

# --------- Setup ---------
st.set_page_config(page_title="Prediksi Tumor Otak MRI", page_icon="🧠", layout="wide")

st.title("Prediksi Jenis Tumor Otak dari Citra MRI")
st.markdown("""
Upload gambar MRI otak dalam format jpg, jpeg, atau png. Sistem akan memprediksi jenis tumor otak jika ada,
atau menyatakan tidak ada tumor (No tumor). Gunakan gambar MRI yang jelas dan fokus pada area otak.
""")

# Sidebar dengan info edukasi
with st.sidebar:
    st.header("Tentang Tumor Otak")
    st.write("""
    Tumor otak adalah massa abnormal dari sel yang tumbuh di dalam atau sekitar otak.
    Berikut beberapa jenis tumor yang umum terdeteksi:
    """)
    
    tumor_info = {
        "glioma": "Glioma adalah tumor yang berasal dari sel glial di otak. Tumor ini dapat bersifat ganas dan menyerang jaringan otak sekitar.",
        "meningioma": "Meningioma adalah tumor yang berkembang di selaput pelindung otak dan sumsum tulang belakang. Biasanya bersifat jinak.",
        "pituitary": "Tumor pituitary berasal dari kelenjar pituitary di dasar otak. Tumor ini dapat memengaruhi produksi hormon.",
        "notumor": "Tidak terdeteksi tumor pada gambar MRI yang diunggah."
    }
    
    for tumor, desc in tumor_info.items():
        st.markdown(f"**{tumor.capitalize()}**")
        st.write(desc)
        st.write("---")

# --------- Download dan Load Model ---------
model_path = 'brain_tumor_model.h5'
file_id = '18lLL4vDzXS9gdDXksyJhuY5MedaafKv7'  # Ganti sesuai modelmu
url = f'https://drive.google.com/uc?id={file_id}'

if not os.path.exists(model_path):
    with st.spinner("Mengunduh model dari Google Drive..."):
        success = gdown.download(url, model_path, quiet=True)
        if not success:
            st.error("Gagal mengunduh model dari Google Drive.")
            st.stop()

try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# --------- Upload dan Prediksi ---------
uploaded_file = st.file_uploader("Upload gambar MRI (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Gambar MRI yang diunggah", use_column_width=True)

        # Resize dan preprocessing
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediksi
        prediction = model.predict(img_array)
        pred_index = np.argmax(prediction)
        confidence = prediction[0][pred_index]

        st.markdown("---")
        st.subheader("Hasil Prediksi")

        if confidence < 0.6:
            st.info("Gambar tidak dikenali sebagai MRI otak yang valid atau kualitas gambar kurang baik. Prediksi: No tumor.")
        else:
            predicted_class = class_names[pred_index]
            if predicted_class == 'notumor':
                st.success(f"Tidak terdeteksi tumor pada gambar ini dengan confidence {confidence:.2f}")
            else:
                st.error(f"Jenis tumor terdeteksi: **{predicted_class.capitalize()}** dengan confidence {confidence:.2f}")

            # Tampilkan deskripsi singkat
            st.write(tumor_info[predicted_class])

    except UnidentifiedImageError:
        st.error("File yang diupload bukan gambar yang valid.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses gambar: {e}")

# --------- Footer ---------
st.markdown("---")
st.caption("Aplikasi ini hanya sebagai alat bantu prediksi dan tidak menggantikan diagnosis medis profesional.")
