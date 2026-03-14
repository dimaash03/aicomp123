from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = BASE_DIR / "reports"

TRAIN_TRANSACTIONS = RAW_DIR / "train_transactions.csv"
TEST_TRANSACTIONS = RAW_DIR / "test_transactions.csv"
TRAIN_USERS = RAW_DIR / "train_users.csv"
TEST_USERS = RAW_DIR / "test_users.csv"
