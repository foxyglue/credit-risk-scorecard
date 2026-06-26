# Credit Risk Scoring

Model credit scoring untuk memprediksi probabilitas default nasabah pinjaman, dengan fokus pada **explainability** dan **business impact** — bukan sekadar memaksimalkan skor leaderboard.

## Problem Statement

Lembaga pemberi pinjaman (fintech lending, multifinance) perlu menyeimbangkan dua hal yang saling bertentangan: menyetujui sebanyak mungkin nasabah yang mampu membayar (financial inclusion), sambil meminimalkan kerugian dari nasabah yang berpotensi default. Project ini membangun model klasifikasi risiko kredit, dengan dua penekanan yang sering dilewatkan oleh notebook publik serupa:

1. **Explainability** — model harus bisa dijelaskan ke pihak non-teknis (risk committee, regulator), bukan cuma akurat.
2. **Business framing** — performa model diterjemahkan ke trade-off bisnis (approval rate vs bad rate), bukan cuma metric statistik.

## Dataset

[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) — data aplikasi pinjaman beserta riwayat kredit dari biro kredit dan pengajuan sebelumnya. ~307.500 nasabah, target imbalanced (~8% default).

> Dataset tidak disertakan di repo ini (lihat `.gitignore`). Download dari Kaggle dan letakkan di `data/raw/`.

### Penjelasan Dataset

Dataset terdiri dari beberapa tabel relasional yang saling terhubung lewat `SK_ID_CURR` (ID nasabah/aplikasi) dan `SK_ID_PREV` (ID aplikasi sebelumnya). Setiap tabel merepresentasikan granularitas data yang berbeda:

| File | Deskripsi | Granularitas |
|------|-----------|---------------|
| `application_train.csv` / `application_test.csv` | Tabel utama: data statis aplikasi pinjaman. Train berisi kolom `TARGET`, test tidak (dipakai untuk submission kompetisi, tidak dipakai di project ini). | Satu baris = satu aplikasi pinjaman |
| `bureau.csv` | Riwayat kredit nasabah di lembaga keuangan lain, sebagaimana dilaporkan ke Credit Bureau. | Satu baris = satu kredit sebelumnya di luar Home Credit (bisa banyak baris per nasabah) |
| `bureau_balance.csv` | Saldo bulanan dari kredit-kredit di atas. | Satu baris = satu bulan riwayat dari satu kredit di `bureau.csv` |
| `previous_application.csv` | Riwayat pengajuan pinjaman sebelumnya ke Home Credit sendiri (bukan lembaga lain). | Satu baris = satu aplikasi sebelumnya |
| `POS_CASH_balance.csv` | Snapshot saldo bulanan dari pinjaman POS (point of sales) dan cash loan sebelumnya di Home Credit. | Satu baris = satu bulan riwayat dari satu kredit sebelumnya |
| `credit_card_balance.csv` | Snapshot saldo bulanan dari kartu kredit sebelumnya di Home Credit. | Satu baris = satu bulan riwayat dari satu kartu kredit sebelumnya |
| `installments_payments.csv` | Riwayat pembayaran cicilan dari kredit-kredit Home Credit sebelumnya, termasuk cicilan yang terlambat/tidak dibayar. | Satu baris = satu pembayaran cicilan (atau satu cicilan yang gagal dibayar) |
| `HomeCredit_columns_description.csv` | Deskripsi tiap kolom di seluruh file di atas — referensi penting saat feature engineering. | - |

> **Catatan kerja:** sebelum membuat asumsi soal arti suatu kolom (terutama
> kolom dengan nama ambigu seperti `EXT_SOURCE_1/2/3` atau kolom hasil
> agregasi), cek dulu definisinya di `HomeCredit_columns_description.csv`.
> Beberapa kolom (seperti `EXT_SOURCE_*`) didefinisikan resmi hanya sebagai
> "normalized score from external data source" tanpa detail metodologi —
> ini perlu disebutkan eksplisit di insight ketimbang menebak sumber/cara
> hitungnya.

**Implikasi untuk feature engineering:** karena tabel pendukung (`bureau`, `previous_application`, dst) punya relasi *one-to-many* terhadap `SK_ID_CURR`, semua tabel ini harus diagregasi dulu (count, mean, max, dsb) menjadi satu baris per nasabah sebelum di-merge ke tabel utama — inilah yang dilakukan oleh `src/data.py::build_master_table()`.

## Struktur Project

```
credit-risk-scoring/
├── src/                          # kode reusable (deterministik)
│   ├── config.py                 # path & konstanta
│   ├── data.py                   # load & merge tabel
│   ├── features.py               # feature engineering domain-specific
│   └── evaluation.py             # custom metric: KS, Gini, AUC
├── notebooks/                    # eksplorasi, eksperimen, narasi
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling_baseline.ipynb
│   ├── 04_modeling_gbm.ipynb
│   ├── 05_explainability.ipynb
│   └── 06_business_impact.ipynb
├── models/                       # model artifact terlatih (gitignored)
├── data/                         # dataset (gitignored)
└── requirements.txt
```

## Metodologi

1. **EDA** — pola missing value, distribusi fitur, imbalance target
2. **Feature Engineering** — rasio finansial (debt-to-income, dst), agregasi riwayat kredit dari bureau & previous_application
3. **Modeling** — dua pendekatan dibandingkan:
   - Logistic Regression + WOE binning (standar industri, regulatory-friendly)
   - LightGBM (performa lebih tinggi, ditangani trade-off interpretability)
4. **Explainability** — SHAP untuk feature importance global & individual
5. **Business Impact** — simulasi cutoff threshold dan dampaknya ke approval rate vs bad rate

## Hasil

*(diisi setelah eksperimen selesai — tabel perbandingan AUC/Gini/KS antar model)*

## Cara Menjalankan

```bash
git clone https://github.com/<username>/credit-risk-scoring.git
cd credit-risk-scoring
pip install -r requirements.txt
# download dataset dari Kaggle ke data/raw/
jupyter notebook notebooks/01_eda.ipynb
```
