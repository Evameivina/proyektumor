# Proyek Tumor Otak

Aplikasi deteksi jenis tumor otak menggunakan deep learning dan Streamlit.  
🔗 Coba aplikasinya: [proyektumor.streamlit.app](https://proyektumor.streamlit.app/)

## Navigasi Aplikasi

Aplikasi memiliki 3 menu utama yang dapat diakses melalui sidebar:

### 1. Panduan Penggunaan Aplikasi
Berisi petunjuk penggunaan sistem deteksi.  
Pengguna mendapatkan informasi tentang format gambar yang didukung, kualitas gambar yang disarankan, serta penjelasan penting sebelum melakukan deteksi.

### 2. Deteksi Tumor
Halaman utama untuk mengunggah gambar MRI otak.  
Sistem akan:
- Menampilkan gambar yang diunggah
- Melakukan klasifikasi ke dalam salah satu dari empat kelas
- Memberikan tingkat kepercayaan (confidence score)
- Menampilkan penjelasan singkat hasil deteksi

Jika gambar tidak sesuai (misalnya buram atau bukan MRI otak), sistem menolak input dan memberikan peringatan.

### 3. Informasi Tumor
Menampilkan informasi lengkap terkait masing-masing jenis tumor:
- Glioma
- Meningioma
- Pituitary
- Notumor

Setiap informasi mencakup pengertian, gejala, metode diagnosis, penanganan, dan referensi ilmiah.

## Sumber Dataset

Model dilatih menggunakan dua dataset publik dari Kaggle:

- [Brain Tumor Classification MRI - Sartaj](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)  
- [Brain Tumor MRI Dataset - Masoud Nickparvar](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

## Kelas yang Terdeteksi

- Glioma  
- Meningioma  
- Notumor  
- Pituitary

## Hasil Evaluasi Model

### Training dan Validasi
- **Akurasi Data Latih**: 97.68%  
- **Loss Data Latih**: 0.1146  
- **Akurasi Validasi**: 96.73%  
- **Loss Validasi**: 0.1555  
- **Learning Rate**: 5e-6  

### Testing
- **Akurasi Data Uji**: 97.16%  
- **Loss Data Uji**: 0.1291  

### Classification Report

| Kelas       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Glioma      | 0.99      | 0.96   | 0.98     | 163     |
| Meningioma  | 0.99      | 0.92   | 0.95     | 165     |
| Notumor     | 0.99      | 1.00   | 1.00     | 200     |
| Pituitary   | 0.92      | 1.00   | 0.96     | 176     |
| **Accuracy**|           |        | **0.97** | 704     |
| Macro Avg   | 0.97      | 0.97   | 0.97     | 704     |
| Weighted Avg| 0.97      | 0.97   | 0.97     | 704     |
