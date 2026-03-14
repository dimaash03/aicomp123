import pandas as pd


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic cleaning rules. Fill in with PDF rules."""
    out = df.copy()
    # TODO: implement filtering/validation from requirements
    out = out.drop_duplicates()
    return out


def clean_users(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic cleaning rules for user table."""
    out = df.copy()
    out = out.drop_duplicates()
    return out
