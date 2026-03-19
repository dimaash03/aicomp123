# AIComp123

End-to-end data preparation for antifraud modeling.

## Structure
- `data/raw/` — raw CSV files (train/test users + transactions)
- `data/processed/` — generated feature tables and reports
- `src/features.py` — feature engineering pipeline
- `notebooks/` — exploration notebooks

## Environment
- Python >= 3.10
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Build features
```bash
python src/features.py
```

Outputs:
- `data/processed/train_features.csv`
- `data/processed/test_features.csv`
- `data/processed/train_features_report.csv`
- `data/processed/test_features_report.csv`
- `data/processed/features_list.json`
- `data/processed/target_encoding_maps.json`

## Training (example)
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

df = pd.read_csv("data/processed/train_features.csv")
X = df.drop(columns=["is_fraud", "id_user"])
y = df["is_fraud"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = lgb.LGBMClassifier(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

pred = model.predict_proba(X_val)[:, 1]
print("AUC:", roc_auc_score(y_val, pred))
```
