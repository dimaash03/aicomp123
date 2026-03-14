import pandas as pd


def quick_eda(df: pd.DataFrame) -> dict:
    """Return quick summary stats for a dataframe."""
    return {
        "shape": df.shape,
        "missing": df.isna().sum().sort_values(ascending=False).head(20).to_dict(),
        "numeric_desc": df.describe().to_dict(),
    }
