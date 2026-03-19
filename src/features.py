from pathlib import Path
import json

import numpy as np
import pandas as pd

BASE_DIR = Path("/Users/user/Documents/AI COMP")
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# One-hot encoding is disabled; we use target encoding instead.
ONE_HOT = False

# Rolling features are disabled for now (too slow)
USE_ROLLING = False

# Target encoding for categorical features (train only, applied to test)
USE_TARGET_ENCODING = True
INCLUDE_DOMAIN_TE = True


def _to_datetime(series):
    return pd.to_datetime(series, format="mixed", utc=True, errors="coerce")


def load_users(split: str) -> pd.DataFrame:
    path = RAW_DIR / f"{split}_users.csv"
    df = pd.read_csv(path)
    df["timestamp_reg"] = _to_datetime(df["timestamp_reg"])
    df["domain"] = df["email"].str.split("@").str[-1].str.lower()
    return df


def load_transactions(split: str) -> pd.DataFrame:
    path = RAW_DIR / f"{split}_transactions.csv"
    df = pd.read_csv(path)
    df["timestamp_tr"] = _to_datetime(df["timestamp_tr"])
    return df


def build_rolling_features(tx: pd.DataFrame) -> pd.DataFrame:
    # Disabled for performance; left for future use
    return pd.DataFrame(columns=["id_user"])


def build_user_tx_features(users: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    # Base counts
    tx_count = tx.groupby("id_user").size().rename("tx_count")

    # Error any (non-null error_group) count
    error_any_count = tx["error_group"].notna().groupby(tx["id_user"]).sum().rename("error_any_count")

    # Status fail rate
    fail_status_coef = (
        tx["status"].eq("fail").groupby(tx["id_user"]).mean().rename("fail_status_coefficient")
    )

    # Error group rates
    fraud_error_rate = (
        tx["error_group"].eq("fraud").groupby(tx["id_user"]).mean().rename("fraud_error_rate")
    )
    fraud_error_count = (
        tx["error_group"].eq("fraud").groupby(tx["id_user"]).sum().rename("fraud_error_count")
    )
    error_antifraud_rate = (
        tx["error_group"].eq("antifraud").groupby(tx["id_user"]).mean().rename("error_antifraud_rate")
    )
    error_3ds_error_rate = (
        tx["error_group"].eq("3ds error").groupby(tx["id_user"]).mean().rename("error_3ds_error_rate")
    )
    error_card_problem_rate = (
        tx["error_group"].eq("card problem").groupby(tx["id_user"]).mean().rename("error_card_problem_rate")
    )

    # Country mismatches
    tx_with_reg = tx.merge(users[["id_user", "reg_country"]], on="id_user", how="left")
    country_cp_mismatch = (
        (tx_with_reg["card_country"] != tx_with_reg["payment_country"])  # card vs payment
        .groupby(tx_with_reg["id_user"]).mean()
        .rename("country_CP_missmatch_coef")
    )
    country_creg_mismatch = (
        (tx_with_reg["card_country"] != tx_with_reg["reg_country"])  # card vs reg
        .groupby(tx_with_reg["id_user"]).mean()
        .rename("country_CReg_missmatch")
    )

    # Unique counts
    unique_cards = tx.groupby("id_user")["card_mask_hash"].nunique().rename("unique_cards_per_user")
    unique_errors = tx.groupby("id_user")["error_group"].nunique().rename("unique_error_types")
    unique_card_countries = tx.groupby("id_user")["card_country"].nunique().rename("unique_card_countries")
    unique_card_holders = tx.groupby("id_user")["card_holder"].nunique().rename("unique_card_holders")
    unique_card_brands = tx.groupby("id_user")["card_brand"].nunique().rename("unique_card_brands")
    unique_card_types = tx.groupby("id_user")["card_type"].nunique().rename("unique_card_types")

    # Amount aggregates
    amount_agg = tx.groupby("id_user")["amount"].agg(
        amount_mean="mean",
        amount_std="std",
        amount_max="max",
        amount_min="min",
        amount_median="median",
        amount_sum="sum",
    )
    amount_agg["amount_std"] = amount_agg["amount_std"].fillna(0)
    amount_agg["amount_range"] = amount_agg["amount_max"] - amount_agg["amount_min"]
    # Transaction type: card_init rate
    card_init_rate = (
        tx["transaction_type"].eq("card_init").groupby(tx["id_user"]).mean().rename("tx_type_card_init_rate")
    )

    # Time to first transaction
    first_tr = tx.groupby("id_user")["timestamp_tr"].min().rename("first_timestamp_tr")
    time_to_first = (
        users.set_index("id_user")["timestamp_reg"].to_frame().join(first_tr)
    )
    time_to_first["time_to_first_transaction"] = (
        time_to_first["first_timestamp_tr"] - time_to_first["timestamp_reg"]
    ).dt.total_seconds()
    time_to_first = time_to_first["time_to_first_transaction"]

    # Time between transactions (min)
    tx_sorted_time = tx.sort_values(["id_user", "timestamp_tr"])
    time_diffs = tx_sorted_time.groupby("id_user")["timestamp_tr"].diff().dt.total_seconds()
    min_time_between_tx = time_diffs.groupby(tx_sorted_time["id_user"]).min().rename("min_time_between_tx")

    # Ratios / interactions
    errors_per_tx = (error_any_count / (tx_count + 1e-6)).rename("errors_per_tx")
    countries_per_tx = (unique_card_countries / (tx_count + 1e-6)).rename("countries_per_tx")
    holders_per_tx = (unique_card_holders / (tx_count + 1e-6)).rename("holders_per_tx")
    holders_per_card = (unique_card_holders / (unique_cards + 1e-6)).rename("holders_per_card")
    errors_x_cards = (error_any_count * unique_cards).rename("errors_x_cards")
    holders_x_cards = (unique_card_holders * unique_cards).rename("holders_x_cards")

    # Build user-level feature table
    features = pd.concat(
        [
            tx_count,
            error_any_count,
            fail_status_coef,
            fraud_error_rate,
            fraud_error_count,
            error_antifraud_rate,
            error_3ds_error_rate,
            error_card_problem_rate,
            country_cp_mismatch,
            country_creg_mismatch,
            unique_cards,
            unique_errors,
            unique_card_countries,
            unique_card_holders,
            unique_card_brands,
            unique_card_types,
            amount_agg,
            card_init_rate,
            time_to_first,
            min_time_between_tx,
            errors_per_tx,
            countries_per_tx,
            holders_per_tx,
            holders_per_card,
            errors_x_cards,
            holders_x_cards,
        ],
        axis=1,
    )

    # Fill missing for users with no tx
    features = features.fillna(0)
    return features.reset_index()

def _clean_name(value: str) -> str:
    if pd.isna(value):
        return ""
    value = str(value).lower()
    value = "".join(ch if ch.isalpha() or ch == " " else " " for ch in value)
    parts = [p for p in value.split() if len(p) > 1]
    return " ".join(parts)


def add_name_match_feature(users: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    users = users.copy()
    users["email_name"] = users["email"].apply(
        lambda x: str(x).split("@")[0].lower() if pd.notna(x) else ""
    )
    most_common_holder = tx.groupby("id_user")["card_holder"].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else ""
    )
    users["main_card_holder"] = users["id_user"].map(most_common_holder).fillna("")
    users["name_in_email"] = users.apply(
        lambda row: any(
            part in row["email_name"]
            for part in str(row["main_card_holder"]).split()
            if len(part) > 2
        ),
        axis=1,
    ).astype(int)
    users = users.drop(columns=["email_name", "main_card_holder"])
    return users

def add_user_time_features(users: pd.DataFrame) -> pd.DataFrame:
    users = users.copy()
    users["reg_hour"] = users["timestamp_reg"].dt.hour
    users["reg_dayofweek"] = users["timestamp_reg"].dt.dayofweek
    users["reg_is_night"] = ((users["reg_hour"] >= 0) & (users["reg_hour"] < 6)).astype(int)
    users["reg_is_weekend"] = (users["reg_dayofweek"] >= 5).astype(int)
    return users


def build_dataset(
    split: str,
    encoders: dict | None = None,
    one_hot: bool = False,
    dummy_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict, list[str] | None]:
    users = load_users(split)
    tx = load_transactions(split)

    users = add_user_time_features(users)
    users = add_name_match_feature(users, tx)
    user_tx_features = build_user_tx_features(users, tx)
    rolling_features = build_rolling_features(tx) if USE_ROLLING else None

    df = users.merge(user_tx_features, on="id_user", how="left")
    if rolling_features is not None and not rolling_features.empty:
        df = df.merge(rolling_features, on="id_user", how="left")

    # Handle missing values and types
    categorical_cols = ["gender", "reg_country", "traffic_type", "domain"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype("string")

    # Gender to binary flags
    df["is_male"] = (df["gender"].str.lower() == "male").astype(int)
    df["is_female"] = (df["gender"].str.lower() == "female").astype(int)

    # Target encoding for selected categoricals
    if USE_TARGET_ENCODING:
        if encoders is None:
            encoders = {}
        if split == "train":
            global_mean = df["is_fraud"].mean()
        else:
            global_mean = encoders.get("_global_mean", 0.0)

        def _fit_te(col):
            mapping = df.groupby(col)["is_fraud"].mean()
            encoders[f"{col}_te"] = mapping.to_dict()
            encoders["_global_mean"] = global_mean

        def _apply_te(col):
            mapping = encoders.get(f"{col}_te", {})
            df[f"{col}_te"] = df[col].map(mapping).fillna(global_mean)

        # reg_country, traffic_type
        for col in ["reg_country", "traffic_type"]:
            if split == "train":
                _fit_te(col)
            _apply_te(col)

        # card_country: use most common card_country per user
        card_country_mode = tx.groupby("id_user")["card_country"].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan
        )
        df = df.merge(card_country_mode.rename("card_country_mode"), on="id_user", how="left")
        if split == "train":
            _fit_te("card_country_mode")
        _apply_te("card_country_mode")
        df = df.drop(columns=["card_country_mode"])

        # optional domain
        if INCLUDE_DOMAIN_TE and "domain" in df.columns:
            if split == "train":
                _fit_te("domain")
            _apply_te("domain")

    # Drop raw columns not needed for training
    drop_cols = ["email", "timestamp_reg", "gender", "domain", "traffic_type", "reg_country"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    numeric_cols = df.columns.difference(["reg_country"])
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    # Filter to selected features (✅ + ⚠️)
    keep = [
        "errors_x_cards",
        "holders_x_cards",
        "unique_card_holders",
        "unique_error_types",
        "unique_cards_per_user",
        "tx_count",
        "fail_status_coefficient",
        "countries_per_tx",
        "min_time_between_tx",
        "amount_sum",
        "unique_card_brands",
        "unique_card_types",
        "country_CP_missmatch_coef",
        "error_antifraud_rate",
        "errors_per_tx",
        "country_CReg_missmatch",
        "fraud_error_count",
        "holders_per_card",
        "fraud_error_rate",
        "tx_type_card_init_rate",
        "error_3ds_error_rate",
        "holders_per_tx",
        "amount_range",
        # ⚠️
        "error_card_problem_rate",
        "amount_std",
        "time_to_first_transaction",
        "name_in_email",
    ]
    # target-encoded categoricals
    te_keep = ["card_country_mode_te", "reg_country_te", "traffic_type_te"]
    if INCLUDE_DOMAIN_TE:
        te_keep.append("domain_te")

    id_cols = ["id_user"]
    if split == "train" and "is_fraud" in df.columns:
        id_cols.append("is_fraud")
    # Always keep gender binary flags
    keep_extra = ["is_male", "is_female"]
    final_cols = id_cols + [c for c in keep + keep_extra + te_keep if c in df.columns]
    df = df[final_cols]

    return df, encoders, dummy_columns


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_df, encoders, dummy_columns = build_dataset("train", one_hot=ONE_HOT)
    train_path = PROCESSED_DIR / "train_features.csv"
    train_df.to_csv(train_path, index=False)

    test_df, _, _ = build_dataset(
        "test", encoders=encoders, one_hot=ONE_HOT, dummy_columns=dummy_columns
    )
    test_path = PROCESSED_DIR / "test_features.csv"
    test_df.to_csv(test_path, index=False)

    # Data preprocessing report (pandas DataFrame)
    def dataframe_report(df: pd.DataFrame, name: str) -> pd.DataFrame:
        report = pd.DataFrame({
            "feature": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "missing_pct": (df.isna().mean() * 100).round(2),
            "n_unique": df.nunique(dropna=True),
        })
        report_path = PROCESSED_DIR / f"{name}_report.csv"
        report.to_csv(report_path, index=False)
        print(f"Saved: {report_path}")
        return report

    dataframe_report(train_df, "train_features")
    dataframe_report(test_df, "test_features")

    # Save feature list (exclude id/target/raw columns)
    drop_cols = {"id_user", "is_fraud", "timestamp_reg", "email"}
    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    (PROCESSED_DIR / "features_list.json").write_text(
        json.dumps(feature_cols, ensure_ascii=False, indent=2)
    )
    if USE_TARGET_ENCODING:
        (PROCESSED_DIR / "target_encoding_maps.json").write_text(
            json.dumps(encoders, ensure_ascii=False, indent=2)
        )

    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")


if __name__ == "__main__":
    main()
