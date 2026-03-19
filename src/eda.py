import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency
df_user = pd.read_csv("/Users/user/Documents/AI COMP/data/raw/train_users.csv")
df_transaction = pd.read_csv("/Users/user/Documents/AI COMP/data/raw/train_transactions.csv")
df = df_transaction.merge(df_user, on="id_user", how="left")
 ##Geographic mismatches
df["mismatch_card_reg"] = df["card_country"] != df["reg_country"]
df["mismatch_pay_reg"]  = df["payment_country"] != df["reg_country"]
df["mismatch_card_pay"] = df["card_country"] != df["payment_country"]

geo = (df.groupby(["mismatch_card_reg","mismatch_pay_reg","mismatch_card_pay"])
         .agg(cnt=("id_user","size"),
              fraud_rate=("is_fraud","mean"))
         .sort_values("cnt", ascending=False))
print(geo.head(10))

print("chi2 — сила відхилення від 'незалежності'. Чим більше — тим сильніший зв'язок")
print("p-value < 0.05 — зв'язок не випадковий (відкидаємо H₀ про незалежність)\n")
cat_columns = ["gender", "reg_country", "traffic_type"]
for col in cat_columns:
    contingency = pd.crosstab(df_user[col], df_user['is_fraud'])
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    print(f"{col}: chi2={chi2:.2f}, p-value={p_value:.6f}")

df = df_transaction.merge(df_user, on="id_user", how="left")

# час у datetime (мікросекунди + timezone у даних)
df["timestamp_tr"] = pd.to_datetime(df["timestamp_tr"], format="mixed", utc=True)

# сортування для коректного rolling
df = df.sort_values(["id_user", "timestamp_tr"])

# set index для time-based rolling
df = df.set_index("timestamp_tr")

# rolling по кожному юзеру
g = df.groupby("id_user")["amount"]

# average / count за останні 1/7/30 днів включно
df["avg_1d"] = g.rolling("1D").mean().reset_index(level=0, drop=True)
df["avg_7d"] = g.rolling("7D").mean().reset_index(level=0, drop=True)
df["avg_30d"] = g.rolling("30D").mean().reset_index(level=0, drop=True)

df["cnt_1d"] = g.rolling("1D").count().reset_index(level=0, drop=True)
df["cnt_7d"] = g.rolling("7D").count().reset_index(level=0, drop=True)
df["cnt_30d"] = g.rolling("30D").count().reset_index(level=0, drop=True)

# попередній день/тиждень (shift на 1 день)
df["avg_prev_1d"] = g.rolling("1D").mean().shift(1).reset_index(level=0, drop=True)
df["avg_prev_7d"] = g.rolling("7D").mean().shift(1).reset_index(level=0, drop=True)

# повернути timestamp назад як колонку
df = df.reset_index()
