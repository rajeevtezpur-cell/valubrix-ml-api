from flask import Flask, request, jsonify
import os
import pickle
import pandas as pd
import numpy as np
import re

app = Flask(__name__)

MODEL_DIR = "models"

# -----------------------
# Load all models
# -----------------------
models = {}

for file in os.listdir(MODEL_DIR):
    if file.startswith("model_") and file.endswith(".pkl"):
        name = file.replace("model_", "").replace(".pkl", "")
        with open(os.path.join(MODEL_DIR, file), "rb") as f:
            models[name] = pickle.load(f)

# -----------------------
# Load encoders
# -----------------------
with open(os.path.join(MODEL_DIR, "encoders.pkl"), "rb") as f:
    encoders = pickle.load(f)

# -----------------------
# Load locality stats
# -----------------------
stats_df = pd.read_csv(os.path.join(MODEL_DIR, "locality_stats.csv"))

# -----------------------
# Helper
# -----------------------
def safe_name(txt):
    return re.sub(r'[^a-zA-Z0-9_]', '_', str(txt))

def encode_value(col, value):
    le = encoders.get(col)
    if le is None:
        return 0

    classes = list(le.classes_)
    if value in classes:
        return int(le.transform([value])[0])

    return 0

# -----------------------
# Routes
# -----------------------
@app.route("/")
def home():
    return "ValuBrix AVM Running"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    property_type = data.get("property_type", "Villa")
    locality = data.get("locality", "Whitefield")
    sub_category = data.get("sub_category", "Standard")
    area_sqft = float(data.get("area_sqft", 1200))

    model_key = safe_name(property_type)

    if model_key not in models:
        return jsonify({"error": "Model not found for property_type"}), 400

    model = models[model_key]

    # locality stats
    row = stats_df[stats_df["locality"] == locality]

    if len(row) > 0:
        mean_psf = float(row.iloc[0]["locality_mean_psf"])
        median_psf = float(row.iloc[0]["locality_median_psf"])
        count = float(row.iloc[0]["locality_count"])
    else:
        mean_psf = 5000
        median_psf = 5000
        count = 1

    locality_enc = encode_value("locality", locality)
    sub_enc = encode_value("sub_category", sub_category)

    area_log = np.log1p(area_sqft)

    X = pd.DataFrame([{
        "area_sqft": area_sqft,
        "area_log": area_log,
        "locality_enc": locality_enc,
        "locality_mean_psf": mean_psf,
        "locality_median_psf": median_psf,
        "locality_count": count,
        "sub_category_enc": sub_enc
    }])

    pred_psf = float(model.predict(X)[0])
    total_price = pred_psf * area_sqft

    return jsonify({
        "property_type": property_type,
        "locality": locality,
        "predicted_psf": round(pred_psf, 2),
        "estimated_price_inr": round(total_price, 2)
    })

# -----------------------
# Start Server
# -----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
