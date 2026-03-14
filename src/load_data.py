import pandas as pd
from . import config


def load_raw():
    """Load raw CSVs from data/raw."""
    train_tx = pd.read_csv(config.TRAIN_TRANSACTIONS)
    test_tx = pd.read_csv(config.TEST_TRANSACTIONS)
    train_users = pd.read_csv(config.TRAIN_USERS)
    test_users = pd.read_csv(config.TEST_USERS)
    return train_tx, test_tx, train_users, test_users
