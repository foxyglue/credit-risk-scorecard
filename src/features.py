"""
Feature engineering domain-specific untuk credit risk scoring.

Fungsi-fungsi di sini membuat fitur turunan (derived feature) yang
punya makna bisnis jelas di dunia credit scoring -- bukan sekadar
transformasi statistik generik. Setiap fungsi didokumentasikan dengan
alasan bisnis di baliknya supaya mudah dijelaskan ulang di README/
saat interview.
"""

import numpy as np
import pandas as pd


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menambahkan fitur rasio finansial dasar yang umum dipakai di
    credit scoring industri (debt-to-income, credit-to-income, dsb).

    Semua rasio dihitung dengan penanganan pembagian oleh nol/NaN.
    """
    df = df.copy()

    # Debt-to-income: makin tinggi, makin berisiko
    df["DEBT_TO_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)

    # Annuity terhadap income: seberapa berat beban cicilan bulanan
    # relatif ke pendapatan nasabah
    df["ANNUITY_TO_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)

    # Credit terhadap annuity: mengindikasikan tenor implisit pinjaman
    df["CREDIT_TO_ANNUITY_RATIO"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"].replace(0, np.nan)

    return df


def add_employment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menambahkan fitur terkait masa kerja dan usia.

    Catatan penting (anomali dikenal di dataset ini): DAYS_EMPLOYED punya
    nilai anomali 365243 untuk nasabah yang tidak bekerja (pensiunan, dll).
    Nilai ini di-flag sebagai kategori terpisah, bukan dibiarkan merusak
    skala numerik.
    """
    df = df.copy()

    # Flag anomali DAYS_EMPLOYED sebelum dikonversi
    df["IS_EMPLOYMENT_ANOMALY"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    days_employed_clean = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    df["EMPLOYMENT_LENGTH_YEARS"] = -days_employed_clean / 365.25
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25

    # Rasio masa kerja terhadap usia -- proxy stabilitas pekerjaan
    df["EMPLOYMENT_TO_AGE_RATIO"] = df["EMPLOYMENT_LENGTH_YEARS"] / df["AGE_YEARS"]

    return df


def add_bureau_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menambahkan fitur turunan dari hasil agregasi tabel bureau
    (lihat src/data.py::aggregate_bureau).

    Kolom bureau_* di sini diasumsikan sudah ada hasil dari proses
    merge di build_master_table().
    """
    df = df.copy()

    if {"bureau_amt_credit_sum_mean", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["BUREAU_CREDIT_TO_INCOME_RATIO"] = (
            df["bureau_amt_credit_sum_mean"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
        )

    if "bureau_credit_count" in df.columns:
        # Tidak ada riwayat di bureau sama sekali -- bisa jadi sinyal
        # "thin file" (nasabah tanpa riwayat kredit), relevan untuk
        # narasi financial inclusion di README
        df["HAS_NO_BUREAU_HISTORY"] = df["bureau_credit_count"].isna().astype(int)
        df["bureau_credit_count"] = df["bureau_credit_count"].fillna(0)

    return df

def add_property_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menambahkan flag missing untuk kolom-kolom terkait properti tempat
    tinggal (COMMONAREA, LIVINGAPARTMENTS, YEARS_BUILD, dst).

    Insight dari EDA: NaN pada kolom-kolom ini punya makna berbeda
    dengan nilai 0 (misal COMMONAREA_MEDI -- NaN bisa berarti informasi
    tidak tersedia/tidak relevan untuk jenis hunian tertentu, sedangkan
    0 berarti memang tidak ada common area sama sekali). Flag ini
    dibuat agar perbedaan itu tidak hilang saat kolom nanti diimputasi.

    Pola ini konsisten dengan penanganan IS_EMPLOYMENT_ANOMALY pada
    add_employment_features() -- flag dulu sebelum nilai aslinya diisi.
    """
    df = df.copy()

    property_cols = [c for c in df.columns if any(
        keyword in c for keyword in [
            "COMMONAREA", "LIVINGAPARTMENTS", "LIVINGAREA",
            "NONLIVINGAPARTMENTS", "NONLIVINGAREA",
            "YEARS_BUILD", "FLOORSMIN", "FLOORSMAX",
            "LANDAREA", "BASEMENTAREA", "ELEVATORS",
            "ENTRANCES", "APARTMENTS", "OWN_CAR_AGE",
        ]
    )]

    for col in property_cols:
        df[f"{col}_MISSING"] = df[col].isnull().astype(int)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline feature engineering lengkap. Dipanggil dari notebook
    setelah build_master_table() di src/data.py.
    """
    df = add_ratio_features(df)
    df = add_employment_features(df)
    df = add_bureau_derived_features(df)
    df = add_property_missing_flags(df)
    return df
