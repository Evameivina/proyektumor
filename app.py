from google.colab import files
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# Load the model
model = load_model('/content/model_tumor_otak_v2.h5')

confidence_threshold = 0.9

def is_probably_mri(img):
    if img is None:
        return False

    # Ukuran minimum (jika perlu)
    h, w = img.shape[:2]
    if h < 100 or w < 100:
        return False

    # Grayscale-like check: channel variance rendah → kemungkinan grayscale
    if len(img.shape) == 3 and img.shape[2] == 3:
        b, g, r = cv2.split(img)
        channel_std = [np.std(b), np.std(g), np.std(r)]
        max_std = max(channel_std)
        min_std = min(channel_std)
        ratio = min_std / (max_std + 1e-6)
        if ratio > 0.9:
            # Hampir grayscale → kemungkinan MRI
            return True

        # Tambahan filter kasar: kalau terlalu hijau, bukan MRI
        mean_total = np.mean(img)
        green_ratio = np.mean(g) / (mean_total + 1e-6)
        if green_ratio > 0.5:  # lebih longgar
            return False

    # Kalau grayscale 2D saja
    if len(img.shape) == 2:
        return True

    return True  # fallback kalau unsure


def predictData(model, class_indices, target_size=(224, 224)):
    uploaded = files.upload()

    for filename in uploaded.keys():
        img = cv2.imread(filename)

        if img is None:
            print(f"Error: Gagal membaca gambar {filename}")
            continue

        if not is_probably_mri(img):
            print(f"{filename} kemungkinan besar bukan gambar MRI.")
            continue

        img_resized = cv2.resize(img, target_size)
        img_normalized = img_resized / 255.0
        img_expanded = np.expand_dims(img_normalized, axis=0)  # (1, 224, 224, 3)

        prediction = model.predict(img_expanded)
        predicted_class_idx = np.argmax(prediction)
        confidence = np.max(prediction)

        if not is_probably_mri(img):
            print(f"{filename} kemungkinan besar bukan gambar MRI.")
            continue

        if confidence < confidence_threshold:
            print(f"Gambar {filename} tidak dikenali sebagai MRI otak (confidence: {confidence:.2f}).")
            continue

        idx_to_class = {v: k for k, v in class_indices.items()}
        predicted_class_name = idx_to_class[predicted_class_idx]

        # Tampilkan gambar + hasil prediksi
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Predicted: {predicted_class_name} (Confidence: {confidence:.2f})")
        plt.axis('off')
        plt.show()

        print(f"Predicted class: {predicted_class_name}")
        print(f"Confidence: {confidence:.2f}")

# Contoh inisialisasi class_to_idx
class_names = sorted(os.listdir(train_dir))  # Pastikan train_dir sudah didefinisikan
class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

# Jalankan prediksi
predictData(model, class_to_idx)
