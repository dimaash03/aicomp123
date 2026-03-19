import pandas as pd


def add_time_features(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Add basic time features from a timestamp column."""
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce")
    out["hour"] = out[ts_col].dt.hour
    out["day_of_week"] = out[ts_col].dt.dayofweek
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    return out
