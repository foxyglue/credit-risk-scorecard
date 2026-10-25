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
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve


def gini_coefficient(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Menghitung Gini coefficient dari AUC-ROC.

    Gini = 2 * AUC - 1

    Gini merupakan transformasi AUC, bukan bukti kesiapan produksi.
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
    Tidak ada cutoff KS universal yang membuktikan kesiapan produksi.

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
    Ringkasan AUC, Gini, KS, average precision, Brier, dan log loss tanpa pembulatan.

    Dirancang untuk dipanggil berulang per model (baseline, GBM, dst)
    lalu hasilnya dikumpulkan jadi satu tabel perbandingan di notebook
    modeling -- ini yang akan ditampilkan di README sebagai hasil utama.
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)
    if y_true.ndim != 1 or y_pred_proba.ndim != 1 or y_true.shape != y_pred_proba.shape:
        raise ValueError("Labels and probabilities must be aligned one-dimensional arrays.")
    if set(np.unique(y_true)) != {0, 1} or not np.isfinite(y_pred_proba).all():
        raise ValueError("Both binary classes and finite probabilities are required.")
    if ((y_pred_proba < 0) | (y_pred_proba > 1)).any():
        raise ValueError("Probabilities must lie in [0, 1].")
    auc = roc_auc_score(y_true, y_pred_proba)
    gini = 2 * auc - 1
    ks, _ = ks_statistic(y_true, y_pred_proba)

    return pd.Series({
        "AUC": auc,
        "Gini": gini,
        "KS": ks,
        "AP": average_precision_score(y_true, y_pred_proba),
        "Brier": brier_score_loss(y_true, y_pred_proba),
        "LogLoss": log_loss(y_true, y_pred_proba),
    })


def plot_diagnostics(y_true, predictions):
    """Compare aligned predictions without choosing a cutoff; quantile calibration bins."""
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import precision_recall_curve

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    roc_ax, pr_ax, ks_ax, calibration_ax = axes.ravel()
    for name, probabilities in predictions.items():
        fpr, tpr, _ = roc_curve(y_true, probabilities)
        roc_ax.plot(fpr, tpr, label=name)
        precision, recall, _ = precision_recall_curve(y_true, probabilities)
        pr_ax.plot(recall, precision, label=name)
        _, curve = ks_statistic(y_true, probabilities)
        curve = curve[np.isfinite(curve.threshold)]
        ks_ax.plot(curve.threshold, curve.ks, label=name)
        observed, predicted = calibration_curve(y_true, probabilities, n_bins=10, strategy="quantile")
        calibration_ax.plot(predicted, observed, marker="o", label=name)
    roc_ax.plot([0, 1], [0, 1], "k--", alpha=.4)
    pr_ax.axhline(np.mean(y_true), color="black", linestyle="--", label="Prevalensi target")
    calibration_ax.plot([0, 1], [0, 1], "k--", alpha=.4)
    for ax, title, xlabel, ylabel in [
        (roc_ax, "ROC", "False positive rate", "True positive rate"),
        (pr_ax, "Precision–recall", "Recall", "Precision"),
        (ks_ax, "Separasi TPR − FPR", "Threshold skor", "KS (0–100)"),
        (calibration_ax, "Reliability diagram (bin kuantil)", "Rata-rata prediksi", "Observed event rate"),
    ]:
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    return fig
