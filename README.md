# Proyek Tumor Otak

Aplikasi deteksi jenis tumor otak menggunakan deep learning dan Streamlit.  
Coba aplikasinya di sini: [proyektumor.streamlit.app](https://proyektumor.streamlit.app/)

## Kelas yang Dideteksi
- Glioma  
- Meningioma  
- Notumor  
- Pituitary

## Hasil Evaluasi Model

### Hasil Training & Validasi
- **Akurasi Data Latih**: 97.68%  
- **Loss Data Latih**: 0.1146  
- **Akurasi Validasi**: 96.73%  
- **Loss Validasi**: 0.1555  
- **Learning Rate**: 5e-6  

### Hasil Pengujian (Testing)
- **Akurasi Data Uji**: 97.16%  
- **Loss Data Uji**: 0.1291  

### Classification Report
Model menunjukkan performa sangat baik dengan nilai precision, recall, dan f1-score tinggi di semua kelas:

| Kelas       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Glioma      | 0.99      | 0.96   | 0.98     | 163     |
| Meningioma  | 0.99      | 0.92   | 0.95     | 165     |
| Notumor     | 0.99      | 1.00   | 1.00     | 200     |
| Pituitary   | 0.92      | 1.00   | 0.96     | 176     |
| Accuracy    |           |        | 0.97     | 704     |
| Macro Avg   | 0.97      | 0.97   | 0.97     | 704     |
| Weighted Avg| 0.97      | 0.97   | 0.97     | 704     |


