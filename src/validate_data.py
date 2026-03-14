import pandas as pd


def basic_report(df: pd.DataFrame) -> dict:
    """Return basic profiling metrics for a dataframe."""
    return {
        "rows": len(df),
        "cols": len(df.columns),
        "missing_by_col": df.isna().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "duplicates": int(df.duplicated().sum()),
    }


def validate_keys(df: pd.DataFrame, keys: list[str]) -> dict:
    """Check key uniqueness and nulls."""
    key_df = df[keys]
    return {
        "nulls": key_df.isna().sum().to_dict(),
        "duplicates": int(key_df.duplicated().sum()),
    }
