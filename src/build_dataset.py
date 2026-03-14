import pandas as pd
from .load_data import load_raw
from .preprocess import clean_transactions, clean_users


def join_tables(transactions: pd.DataFrame, users: pd.DataFrame, key: str) -> pd.DataFrame:
    """Join transactions with users using the provided key."""
    return transactions.merge(users, on=key, how="left")


def build():
    train_tx, test_tx, train_users, test_users = load_raw()

    train_tx = clean_transactions(train_tx)
    test_tx = clean_transactions(test_tx)
    train_users = clean_users(train_users)
    test_users = clean_users(test_users)

    # TODO: replace 'user_id' with the actual join key from requirements
    train = join_tables(train_tx, train_users, key="user_id")
    test = join_tables(test_tx, test_users, key="user_id")

    return train, test


if __name__ == "__main__":
    train, test = build()
    print(train.shape, test.shape)
