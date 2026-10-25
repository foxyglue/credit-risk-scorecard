"""
Fungsi untuk memuat dan menggabungkan tabel-tabel pada dataset
Home Credit Default Risk.

Dataset terdiri dari beberapa tabel terkait (application, bureau,
previous_application, dst). Modul ini menyediakan fungsi reusable
untuk load raw data dan melakukan agregasi dasar sebelum feature
engineering lebih lanjut dilakukan di src/features.py.

Catatan desain:
- Fungsi di sini bersifat deterministik (tidak ada eksperimen/trial-error).
  Eksplorasi dan analisis tetap dilakukan di notebook.
- Agregasi yang dilakukan di sini adalah agregasi dasar (count, mean, sum
  sederhana). Agregasi yang lebih kompleks/derived feature domain-specific
  ada di src/features.py.
"""

from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RAW_FILES, ID_COL, TARGET_COL, RANDOM_SEED, TEST_SIZE


def split_application_ids(application: pd.DataFrame) -> pd.DataFrame:
    """Stable stratified split; independent of input row order."""
    rows = application[[ID_COL, TARGET_COL]].sort_values(ID_COL)
    if rows[ID_COL].isna().any() or rows[ID_COL].duplicated().any():
        raise ValueError("Application IDs must be present and unique.")
    if rows[TARGET_COL].isna().any() or set(rows[TARGET_COL].unique()) != {0, 1}:
        raise ValueError("TARGET must contain both binary classes, without missing values.")
    train_ids, test_ids = train_test_split(
        rows[ID_COL], test_size=TEST_SIZE, random_state=RANDOM_SEED,
        stratify=rows[TARGET_COL],
    )
    manifest = rows[[ID_COL]].copy()
    manifest["split"] = manifest[ID_COL].isin(test_ids).map({False: "train", True: "test"})
    assert set(train_ids).isdisjoint(test_ids)
    return manifest.reset_index(drop=True)


def load_raw_tables(files: Dict[str, Path] = RAW_FILES) -> Dict[str, pd.DataFrame]:
    """
    Memuat seluruh tabel mentah Home Credit dari CSV ke dalam dict of DataFrame.

    Parameters
    ----------
    files : dict[str, Path]
        Mapping nama tabel -> path file csv. Default mengambil dari
        src.config.RAW_FILES.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping nama tabel -> DataFrame. Tabel yang file-nya belum ada
        di disk akan di-skip dengan warning (supaya proses awal tetap
        bisa jalan walau belum semua file di-download).
    """
    tables = {}
    for name, path in files.items():
        if not path.exists():
            print(f"[WARNING] File tidak ditemukan, di-skip: {path}")
            continue
        tables[name] = pd.read_csv(path)
        print(f"[INFO] Loaded {name}: {tables[name].shape}")
    return tables


def aggregate_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    """
    Agregasi tabel bureau (riwayat kredit dari biro kredit lain) menjadi
    satu baris per SK_ID_CURR, supaya bisa di-join ke tabel aplikasi utama.

    Agregasi dasar: jumlah kredit sebelumnya, rata-rata/maks credit amount,
    rata-rata days overdue. Feature yang lebih derived (rasio, dsb)
    didefinisikan di src/features.py.
    """
    agg = bureau.groupby(ID_COL).agg(
        bureau_credit_count=("SK_ID_BUREAU", "count"),
        bureau_amt_credit_sum_mean=("AMT_CREDIT_SUM", "mean"),
        bureau_amt_credit_sum_max=("AMT_CREDIT_SUM", "max"),
        bureau_credit_day_overdue_mean=("CREDIT_DAY_OVERDUE", "mean"),
        bureau_credit_day_overdue_max=("CREDIT_DAY_OVERDUE", "max"),
    ).reset_index()
    return agg


def aggregate_previous_application(prev: pd.DataFrame) -> pd.DataFrame:
    """
    Agregasi tabel previous_application (riwayat pengajuan pinjaman
    sebelumnya ke Home Credit) menjadi satu baris per SK_ID_CURR.
    """
    agg = prev.groupby(ID_COL).agg(
        prev_application_count=("SK_ID_PREV", "count"),
        prev_amt_credit_mean=("AMT_CREDIT", "mean"),
        prev_amt_annuity_mean=("AMT_ANNUITY", "mean"),
        prev_days_decision_mean=("DAYS_DECISION", "mean"),
    ).reset_index()
    return agg


def build_master_table(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Menggabungkan application_train dengan agregasi tabel-tabel pendukung
    (bureau, previous_application) menjadi satu master table siap pakai
    untuk feature engineering dan modeling.

    Parameters
    ----------
    tables : dict[str, pd.DataFrame]
        Hasil dari load_raw_tables().

    Returns
    -------
    pd.DataFrame
        Master table dengan satu baris per SK_ID_CURR.
    """
    if "application_train" not in tables:
        raise KeyError(
            "application_train tidak ditemukan di `tables`. "
            "Pastikan file application_train.csv sudah di-load."
        )

    master = tables["application_train"].copy()
    if master[ID_COL].isna().any() or master[ID_COL].duplicated().any():
        raise ValueError("Application IDs must be present and unique before merging.")

    if "bureau" in tables:
        bureau_agg = aggregate_bureau(tables["bureau"])
        master = master.merge(bureau_agg, on=ID_COL, how="left", validate="one_to_one")
        print(f"[INFO] Setelah merge bureau: {master.shape}")

    if "previous_application" in tables:
        prev_agg = aggregate_previous_application(tables["previous_application"])
        master = master.merge(prev_agg, on=ID_COL, how="left", validate="one_to_one")
        print(f"[INFO] Setelah merge previous_application: {master.shape}")

    return master
