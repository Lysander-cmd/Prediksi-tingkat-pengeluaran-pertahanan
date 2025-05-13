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

## Cara Menjalankan Aplikasi

1. Pastikan semua dependensi terinstal:
   ```
   pip install pandas numpy scikit-learn matplotlib joblib pillow
   ```

2. Siapkan dataset Anda dengan format berikut dan simpan di folder `data/`:
   - File CSV dengan kolom minimal: `country`, `year`, `Military expenditure (current USD)`
   - Kolom tambahan yang didukung: `Military expenditure (% of GDP)`, `Military expenditure (% of general government expenditure)`

3. Jalankan aplikasi:
   ```
   python main_gui.py
   ```

## Format Data

Dataset harus berupa file CSV dengan struktur kolom sebagai berikut:

- `country`: Nama negara
- `iso3c`: Kode ISO 3 karakter (opsional)
- `iso2c`: Kode ISO 2 karakter (opsional)
- `year`: Tahun data
- `Military expenditure (current USD)`: Pengeluaran militer dalam USD
- `Military expenditure (% of general government expenditure)`: Persentase dari pengeluaran pemerintah (opsional)
- `Military expenditure (% of GDP)`: Persentase dari GDP (opsional)
- `adminregion`: Wilayah administratif (opsional)
- `incomeLevel`: Tingkat pendapatan (opsional)

## Model Pembelajaran Mesin yang Digunakan

Aplikasi ini menggunakan tiga model pembelajaran mesin:
1. **Random Forest Regressor**: Model ensemble yang cocok untuk pola non-linear dan outlier
2. **Linear Regression**: Model dasar untuk hubungan linear
3. **Polynomial Regression**: Model untuk menangkap pola kurva dan tren non-linear 

## Metrik Evaluasi

Aplikasi ini mengukur performa model menggunakan metrik-metrik berikut:
- **R² Score**: Menunjukkan kemampuan model dalam menjelaskan variasi data (0-1, semakin tinggi semakin baik)
- **RMSE (Root Mean Square Error)**: Rata-rata akar kuadrat kesalahan (nilai rendah lebih baik)

## Cara Mengembangkan Aplikasi

Jika Anda ingin menambahkan model baru:
1. Tambahkan implementasi model di `src/train.py` dalam fungsi `train_and_save_models`
2. Perbarui tampilan hasil di `main_gui.py`

## Catatan Penting

- Pastikan format dataset sesuai dengan yang dijelaskan di atas
- Gunakan versi Python 3.7 atau yang lebih baru
- Perhatikan penamaan kolom dalam dataset yang harus tepat