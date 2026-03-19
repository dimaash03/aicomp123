import pandas as pd
from . import config


def load_raw():
    """Load raw CSVs from data/raw."""
    train_tx = pd.read_csv(config.TRAIN_TRANSACTIONS)
    test_tx = pd.read_csv(config.TEST_TRANSACTIONS)
    train_users = pd.read_csv(config.TRAIN_USERS)
    test_users = pd.read_csv(config.TEST_USERS)
    return train_tx, test_tx, train_users, test_users


def preview(df: pd.DataFrame, name: str, n: int = 5) -> None:
    """Print a quick preview and schema."""
    print(f"\n=== {name} ===")
    print(df.head(n))
    print("\nINFO")
    print(df.info())
    print("\nMISSING")
    print(df.isna().sum().sort_values(ascending=False).head(20))
    print("\nDUPLICATES", df.duplicated().sum())
