from .load_data import load_raw, preview

def build():
    train_tx, test_tx, train_users, test_users = load_raw()

    preview(train_tx, "train_transactions")
    preview(test_tx, "test_transactions")
    preview(train_users, "train_users")
    preview(test_users, "test_users")

    return train_tx, test_tx, train_users, test_users

if __name__ == "__main__":
    build()
