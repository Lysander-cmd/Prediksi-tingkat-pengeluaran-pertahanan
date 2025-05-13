# Sistem Prediksi Pengeluaran Militer

Aplikasi berbasis Python untuk analisis dan prediksi pengeluaran militer negara-negara di dunia menggunakan machine learning.

## Fitur Aplikasi

- Analisis tren pengeluaran militer berdasarkan data historis
- Prediksi pengeluaran militer menggunakan beberapa model pembelajaran mesin
- Visualisasi perbandingan antara nilai aktual dan hasil prediksi
- Metrik evaluasi model yang komprehensif

## Struktur Proyek

```
Prediksi tingkat pengeluaran pertahanan/
│
├── data/
│   └── military_expenditure.csv  # Data pengeluaran militer negara-negara
│
├── src/
│   ├── __init__.py              # File inisialisasi package
│   ├── preprocessing.py         # Modul untuk persiapan data
│   └── train.py                 # Modul untuk pelatihan dan evaluasi model
│
├── models/                      # Direktori penyimpanan model terlatih
│
├── main_gui.py                  # Aplikasi utama dengan antarmuka pengguna grafis
└── README.md                    # Dokumentasi proyek
```
