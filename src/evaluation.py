"""
Custom evaluation metric untuk credit risk scoring.

Selain metric standar (AUC-ROC), industri credit scoring punya metric
khas sendiri: KS statistic dan Gini coefficient. Metric ini lebih mudah
dikomunikasikan ke stakeholder non-teknis (risk committee, regulator)
dibanding AUC murni, karena punya interpretasi langsung terhadap
separasi antara nasabah baik vs nasabah default.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def gini_coefficient(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Menghitung Gini coefficient dari AUC-ROC.

    Gini = 2 * AUC - 1

    Konvensi industri credit scoring: Gini > 0.4 umumnya dianggap
    model yang cukup kuat untuk dipakai produksi (rule of thumb,
    bukan aturan baku).
    """
    auc = roc_auc_score(y_true, y_pred_proba)
    return 2 * auc - 1


def ks_statistic(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Tuple[float, pd.DataFrame]:
    """
    Menghitung Kolmogorov-Smirnov (KS) statistic: jarak maksimum antara
    distribusi kumulatif TPR (nasabah default yang tertangkap) dan FPR
    (nasabah baik yang salah ditangkap) di sepanjang threshold.

    KS statistic adalah metric standar di industri perbankan untuk
    mengukur kemampuan model memisahkan kelas baik vs default.
    Rule of thumb industri: KS > 40 (dalam skala 0-100) dianggap baik.

    Returns
    -------
    ks_value : float
        Nilai KS statistic dalam skala 0-100.
    curve_df : pd.DataFrame
        DataFrame berisi threshold, TPR, FPR, dan selisihnya -- berguna
        untuk plotting KS curve di notebook.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    ks_values = tpr - fpr
    ks_value = float(np.max(ks_values) * 100)

    curve_df = pd.DataFrame({
        "threshold": thresholds,
        "tpr": tpr,
        "fpr": fpr,
        "ks": ks_values * 100,
    })

    return ks_value, curve_df


def evaluation_summary(y_true: np.ndarray, y_pred_proba: np.ndarray) -> pd.Series:
    """
    Ringkasan metric evaluasi standar untuk satu model: AUC, Gini, KS.

    Dirancang untuk dipanggil berulang per model (baseline, GBM, dst)
    lalu hasilnya dikumpulkan jadi satu tabel perbandingan di notebook
    modeling -- ini yang akan ditampilkan di README sebagai hasil utama.
    """
    auc = roc_auc_score(y_true, y_pred_proba)
    gini = gini_coefficient(y_true, y_pred_proba)
    ks, _ = ks_statistic(y_true, y_pred_proba)

    return pd.Series({
        "AUC": round(auc, 4),
        "Gini": round(gini, 4),
        "KS": round(ks, 2),
    })
