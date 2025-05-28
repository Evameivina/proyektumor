import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
from keras.models import load_model
import gdown
import os

# Download model dari Google Drive jika belum ada
model_path = 'brain_tumor_model.h5'
file_id = '1wy92im1lFWabckaU244AI2nLhaSEp-BI'
url = f'https://drive.google.com/uc?id={file_id}'

if not os.path.exists(model_path):
    st.info('Mengunduh model dari Google Drive...')
    gdown.download(url, model_path, quiet=False)

# Load model
model = load_model(model_path)

# Label kelas sesuai urutan output model
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

st.title("Prediksi Jenis Tumor Otak dari Citra MRI 🧠")
st.write("Upload gambar MRI otak yang jelas untuk memprediksi jenis tumornya.")

uploaded_file = st.file_uploader("Pilih gambar MRI (jpg, jpeg, png)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Baca dan konversi gambar
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Gambar yang diunggah', use_column_width=True)

        # Preprocessing gambar
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # bentuk (1, 224, 224, 3)

        # Prediksi model
        prediction = model.predict(img_array)
        pred_index = np.argmax(prediction)
        confidence = prediction[0][pred_index]

        # Threshold confidence supaya yakin prediksinya valid
        if confidence < 0.6:
            st.error("Gambar tidak dikenali atau kualitas gambar kurang baik untuk prediksi.")
        else:
            predicted_class = class_names[pred_index]
            st.success(f"Jenis tumor terdeteksi: **{predicted_class}** dengan confidence {confidence:.2f}")

    except UnidentifiedImageError:
        st.error("File yang diupload bukan gambar yang valid.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
