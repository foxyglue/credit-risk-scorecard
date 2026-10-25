# Credit Risk Scoring

Portofolio Data Scientist bidang kredit menggunakan **Home Credit Default Risk**.
Tahap yang selesai: EDA, feature engineering, baseline WOE + Logistic Regression,
dan LightGBM dengan fitur numerik serta kategorikal. Semua notebook 01–04 sudah dijalankan
berurutan dengan kernel baru dan menyimpan tabel, grafik, serta kesimpulan aktual.

## Problem dan batas target

Tujuan jangka panjangnya menghubungkan kemampuan model membedakan aplikasi berisiko
dengan explainability dan trade-off kebijakan kredit. Tahap ini mengevaluasi ranking dan
diagnostik probabilitas; belum membuat keputusan approval, scorecard points, atau estimasi laba.

Data dictionary mendefinisikan `TARGET=1` sebagai **payment difficulties**: keterlambatan
lebih dari X hari pada setidaknya satu dari Y cicilan awal. X, Y, dan horizon lengkap tidak
dipublikasikan. `TARGET=0` adalah kasus lain, bukan bukti pinjaman telah lunas. Karena itu
output belum diklaim sebagai PD dengan horizon tertentu atau probabilitas terkalibrasi.

## Data dan notebook

Dataset [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
berisi 307.511 aplikasi berlabel dengan sekitar 8,07% target positif.
Unduh dataset ke `data/raw/`; data dan model lokal diabaikan oleh Git.

Tiga tabel yang dipakai: `application_train.csv` (satu aplikasi per `SK_ID_CURR`),
`bureau.csv` (banyak kredit eksternal per aplikasi), dan `previous_application.csv`
(banyak aplikasi historis per aplikasi saat ini). Tabel pendukung diagregasi menjadi
satu baris per ID sebelum left join dengan validasi one-to-one.
`HomeCredit_columns_description.csv` menjadi rujukan definisi fitur.

`application_test.csv` kompetisi tidak dipakai. Bureau balance, installments payments,
credit card balance, dan POS cash balance belum menjadi sumber fitur tahap ini.
`EXT_SOURCE_*` hanya didefinisikan sebagai normalized score dari sumber eksternal;
sumber dan metodologinya tidak ditebak dari namanya.

| Notebook | Pertanyaan / output | Status |
|---|---|---|
| [01 — EDA](notebooks/01_eda.ipynb) | Kualitas data, imbalance, missing/sentinel, numerik dan kategorikal pada training | Selesai, output tersimpan |
| [02 — Feature engineering](notebooks/02_feature_engineering.ipynb) | 49 fitur turunan, master table 180 kolom, manifest split; tanpa seleksi target | Selesai, output tersimpan |
| [03 — Baseline](notebooks/03_modeling_baseline.ipynb) | WOE, IV, korelasi, perbandingan weighting, evaluasi dan artefak | Selesai, output tersimpan |
| [04 — LightGBM](notebooks/04_modeling_gbm.ipynb) | 10 trial × 5 fold, fitur native, perbandingan kedua pipeline | Selesai, output tersimpan |
| 05 — Explainability | SHAP global dan individual | Rencana, belum dibuat |
| 06 — Business impact | Cutoff, approval rate versus bad rate, asumsi biaya | Rencana, belum dibuat |

## Metodologi

- Satu split stratified berdasarkan ID: 246.008 training (80%), 61.503 test (20%), seed 42.
  Manifest stabil terhadap urutan baris. EDA yang memengaruhi keputusan fitur memakai training.
- Notebook 02 mempertahankan fitur asli, rasio, usia/masa kerja bersih, employment anomaly,
  dan flag missing. Pembagi rasio nol/negatif/tidak finite menjadi NaN.
  `DAYS_BIRTH`/`DAYS_EMPLOYED` disimpan untuk penelusuran tetapi dikeluarkan dari kandidat model.
- Baseline: OptBinning numerik/biner/kategorikal, maksimum lima bin, minimum bin size 5%,
  arah ascending/descending untuk numerik kontinu; missing memakai WOE empiris training,
  kategori baru memakai WOE netral 0. IV ≥ 0,02 adalah screening heuristik, bukan signifikansi
  atau detektor leakage. Korelasi WOE absolut ≥ 0,90 disaring dengan prioritas IV lebih tinggi.
- Seluruh binning dan seleksi baseline dipelajari ulang dalam training tiap fold.
  Bandingkan `class_weight=None` dan `balanced` pada 5 fold sama; pilih mean ROC-AUC tertinggi,
  kemudian mean Brier terendah jika AUC seri. Tidak ada filter otomatis missing >30%.
- LightGBM memakai kandidat sebelum seleksi IV baseline; daftar kategori dipelajari dari training
  tiap fold dan kategori baru menjadi missing. Optuna TPE seed 42 menjalankan 10 trial,
  termasuk pilihan weighting. Rasio kelas dihitung dalam training fold, row sampling aktif,
  early stopping 30 ronde, dan final fit memakai median best iteration lima fold trial terbaik.
- Semua konfigurasi dipilih dari CV sebelum evaluasi test. Angka mentah dipakai dalam optimasi;
  pembulatan hanya untuk penyajian. Mean CV terbaik sesudah tuning merupakan skor seleksi,
  bukan estimasi tanpa bias. Model disimpan bersama preprocessing dan metadata split.

## Hasil aktual pada test yang sama

| Model | ROC-AUC | AP | Gini | KS (0–100) | Brier ↓ | Log loss ↓ | Fitur |
|---|---:|---:|---:|---:|---:|---:|---:|
| LR + WOE | 0.7418 | 0.2207 | 0.4836 | 36.2493 | 0.2054 | 0.5974 | 37 |
| LightGBM | 0.7713 | 0.2639 | 0.5427 | 40.5996 | 0.0669 | 0.2420 | 176 |

| Model | CV ROC-AUC mean ± SD | Konfigurasi |
|---|---:|---|
| LR + WOE | 0.7388 ± 0.0013 | class_weight=balanced |
| LightGBM | 0.7679 ± 0.0012 | weighted=False, 332 boosting rounds |

AP adalah average precision. AUC/AP/Gini/KS lebih tinggi lebih baik; Brier/log loss lebih rendah
lebih baik. Gini adalah transformasi AUC. Perbandingan mencakup preprocessing dan seleksi fitur
masing-masing pipeline, bukan efek algoritma saja. Ketidakpastian selisih model belum diuji.

**Temuan kalibrasi aktual:** rata-rata CV AUC `balanced` hanya +0.000042 dibanding tanpa weighting; ini jauh lebih kecil daripada variasi antar-fold. CV Brier memburuk dari 0.0691 menjadi 0.2064, dan log loss dari 0.2520 menjadi 0.5996. Baseline berbobot tetap dipilih mengikuti aturan mean AUC yang ditetapkan sebelum evaluasi test. Reliability diagram menunjukkan prediksi terlalu tinggi terhadap observed event rate; output ini belum dapat digunakan sebagai PD atau dasar expected loss tanpa pekerjaan kalibrasi lanjutan.

LightGBM menggunakan weighting **False** dan final **332**
boosting rounds. Selisih test ROC-AUC terhadap baseline adalah
**+0.0295**. Baca pula Brier/log loss
dan reliability diagram di notebook 04; tidak ada klaim bahwa peningkatan ranking membuat
model otomatis siap dipakai untuk keputusan kredit.

**Gap generalisasi aktual:** ROC-AUC train 0.8381 versus test 0.7713, selisih 0.0667. Mean CV 0.7679 relatif dekat dengan test, tetapi gap train–test tetap menunjukkan optimisme in-sample yang perlu diperhatikan. Model tidak diklaim bebas overfitting; konfigurasi tidak diubah berdasarkan hasil test.

## Struktur dan artefak

- `src/config.py`, `src/data.py`: path, konstanta, agregasi, dan pembagian ID bersama.
- `src/features.py`: pembentukan fitur deterministik dan rasio aman.
- `src/modeling.py`: preprocessing train-fitted yang sama untuk CV, refit, dan inference.
- `src/evaluation.py`: metrik tanpa pembulatan dan grafik ROC/PR/KS/reliability.
- `tests/test_pipeline.py`: enam regression checks untuk binning, kategori/missing,
  isolasi fold, pembagian data, rasio, metrik, dan model round-trip.
- `scripts/run_notebooks.py`: eksekusi notebook 01–04 dengan kernel baru tiap notebook.

Notebook menghasilkan `data/processed/master_table.parquet`, `split_manifest.csv`, tabel CV/IV/bin,
`model_comparison.csv`, metadata metrik JSON, serta prediksi test Parquet berisi ID, target,
dan probabilitas. `models/baseline_woe.joblib` dan `models/lightgbm_final.joblib` berisi pipeline,
urutan fitur, kategori, parameter, manifest split, dan versi paket. Input pipeline adalah
fitur hasil notebook 02. Prediksi model yang dimuat ulang sudah dibandingkan pada seluruh test.

## Menjalankan ulang

Gunakan Python 3.12; eksekusi ini memakai **Python 3.12.13**.
`requirements-tested.txt` mencatat versi paket yang benar-benar dipakai, bukan lock lintas platform.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# Letakkan tiga CSV sumber dan data dictionary di data/raw/
python -m unittest discover -s tests -v
python scripts/run_notebooks.py
```

Untuk membaca secara interaktif: `jupyter notebook notebooks/01_eda.ipynb`, lalu jalankan
01 → 02 → 03 → 04 menggunakan environment yang sama. Runner memakai interpreter pemanggil
untuk kernel lokal dan menimpa output notebook serta artefak turunan. Ia berhenti pada error.
`--start 3` atau `--start 4` hanya untuk melanjutkan jika artefak tahap sebelumnya masih sesuai.

Pada Windows ini build Conda ZeroMQ mengalami crash saat kernel mulai. Eksekusi berhasil
setelah mengganti build `pyzmq==27.1.0` dengan wheel PyPI versi sama; jika mengalami error identik,
gunakan `python -m pip install --force-reinstall --only-binary=:all: pyzmq==27.1.0`.

## Keterbatasan dan langkah berikutnya

Test sudah pernah dilihat pada eksperimen sebelum perbaikan. Hasil ini adalah evaluasi internal
random split, bukan validasi temporal atau eksternal. Informasi kalender untuk validasi lintas
waktu dan definisi horizon target lengkap tidak tersedia pada alur ini. Waktu ketersediaan fitur
dan kemampuan generalisasi pada populasi baru belum dibuktikan hanya oleh IV/AUC.

Kalibrasi ulang, pemilihan cutoff, SHAP, fairness/subgroup evaluation, serta simulasi bisnis
belum dilakukan. Tidak ada ambang IV/Gini/KS universal yang membuktikan kesiapan produksi.
Pekerjaan sekarang berhenti di notebook 04; notebook 05–06 tetap rencana.

Referensi metode: [OptBinning](https://gnpalencia.org/optbinning/binning_process.html),
[pencegahan leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage),
[kalibrasi probabilitas](https://scikit-learn.org/stable/modules/calibration.html).
