# train.py
# ValuBrix AVM — Model Training Pipeline

import pandas as pd
import numpy as np
import pickle
import os
import json
import warnings
import re

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
# import catboost as cb   # optional (commented to avoid error)
# import xgboost as xgb   # optional

CSV_FILE = "bangalore_corpus.csv"
OUTPUT_DIR = "models"
TEST_SIZE = 0.2
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(CSV_FILE)

# Clean column names
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# Rename columns
col_map = {
    "unit_id": "record_id",
    "region": "zone",
    "asset_type": "property_type",
    "format_sub_type": "sub_category",
    "psf_price_inr": "psf",
    "area_sqft": "area_sqft",
    "total_price_inr": "total_price",
    "status": "status",
}
df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

# Clean data
df = df.dropna(subset=["locality", "psf", "area_sqft", "total_price"])
df = df[df["area_sqft"] > 0]
df = df[df["psf"] > 0]
df = df[df["total_price"] > 0]

# Remove outliers
def remove_outliers(group):
    mean = group["psf"].mean()
    std = group["psf"].std()
    if std == 0 or np.isnan(std):
        return group
    return group[(group["psf"] >= mean - 3*std) & (group["psf"] <= mean + 3*std)]

df = df.groupby("property_type", group_keys=False).apply(remove_outliers)

# Encode categorical
le_dict = {}
cat_cols = ["locality", "zone", "property_type", "sub_category", "status"]

for col in cat_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

# Feature engineering
df["area_log"] = np.log1p(df["area_sqft"])

locality_stats = df.groupby("locality")["psf"].agg(
    locality_mean_psf="mean",
    locality_median_psf="median",
    locality_count="count"
).reset_index()

df = df.merge(locality_stats, on="locality", how="left")

BASE_FEATURES = [
    "area_sqft", "area_log",
    "locality_enc", "locality_mean_psf",
    "locality_median_psf", "locality_count",
    "sub_category_enc",
]

TARGET = "psf"

# Train model
def train_model(X, y):
    model = lgb.LGBMRegressor(n_estimators=300)
    model.fit(X, y)
    return model

# Train per property type
for property_type in df["property_type"].unique():
    subset = df[df["property_type"] == property_type]

    if len(subset) < 50:
        continue

    X = subset[BASE_FEATURES].fillna(0)
    y = subset[TARGET]

    model = train_model(X, y)

    # SAFE FILE NAME FIX
    safe_property_type = re.sub(r'[^a-zA-Z0-9_]', '_', str(property_type))

    model_path = os.path.join(OUTPUT_DIR, f"model_{safe_property_type}.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Saved → {model_path}")

print("Training complete")
