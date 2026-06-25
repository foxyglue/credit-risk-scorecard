"""
Konfigurasi terpusat untuk project Credit Risk Scoring.

Semua path, nama kolom kunci, dan konstanta proyek didefinisikan di sini
supaya tidak ada hardcoded value yang tersebar di banyak notebook/script.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Pastikan folder ada (aman dipanggil berkali-kali)
for _dir in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Nama file mentah (sesuai nama asli dari Kaggle: Home Credit Default Risk)
# ---------------------------------------------------------------------------
RAW_FILES = {
    "application_train": RAW_DATA_DIR / "application_train.csv",
    "application_test": RAW_DATA_DIR / "application_test.csv",
    "bureau": RAW_DATA_DIR / "bureau.csv",
    "bureau_balance": RAW_DATA_DIR / "bureau_balance.csv",
    "previous_application": RAW_DATA_DIR / "previous_application.csv",
    "installments_payments": RAW_DATA_DIR / "installments_payments.csv",
    "credit_card_balance": RAW_DATA_DIR / "credit_card_balance.csv",
    "pos_cash_balance": RAW_DATA_DIR / "POS_CASH_balance.csv",
}

# ---------------------------------------------------------------------------
# Kolom kunci
# ---------------------------------------------------------------------------
ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"

# ---------------------------------------------------------------------------
# Konstanta eksperimen
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.2
N_FOLDS = 5

# Threshold default untuk klasifikasi (akan dieksplorasi ulang di notebook
# business impact — ini hanya nilai default untuk eksperimen awal)
DEFAULT_THRESHOLD = 0.5
