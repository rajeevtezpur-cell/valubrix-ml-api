# ValuBrix FastAPI ML Service — main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import joblib, numpy as np, shap, os

app = FastAPI(title="ValuBrix ML Service", version="1.0.0")

# ── Input Schema ─────────────────────────────────────────────────────────────
class ValuationRequest(BaseModel):
    micro_locality: str
    city: str
    bhk: int
    area_sqft: float
    floor: int
    total_floors: int
    age_years: int
    furnishing: str          # unfurnished | semifurnished | furnished
    parking_spaces: int
    amenity_count: int
    facing: str              # north | south | east | west | north-east
    builder_id: str
    latitude: float
    longitude: float
    is_rera_approved: bool
    property_subtype: str    # standalone | gated | township | luxury_tower

class ValuationResponse(BaseModel):
    fair_value: float
    price_range_min: float
    price_range_max: float
    confidence_score: float
    valuation_tag: str       # undervalued | fair_value | overpriced | premium_luxury
    shap_factors: List[dict]
    one_year_forecast: float
    three_year_forecast: float
    estimated_rent_min: float
    estimated_rent_max: float
    comparable_count: int
    model_used: str

# ── Model Registry ────────────────────────────────────────────────────────────
MODEL_DIR = os.environ.get("MODEL_DIR", "./models")

MODELS = {
    "apartment-standalone": None,
    "apartment-gated": None,
    "apartment-township": None,
    "villa": None,
    "plot": None,
    "commercial": None,
    "rental": None,
}

@app.on_event("startup")
async def load_models():
    for model_key in MODELS:
        path = f"{MODEL_DIR}/lgbm_{model_key.replace('-', '_')}.pkl"
        if os.path.exists(path):
            MODELS[model_key] = joblib.load(path)

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/api/v1/health")
async def health():
    loaded = [k for k, v in MODELS.items() if v is not None]
    return {"status": "ok", "models_loaded": loaded}

@app.post("/api/v1/predict/apartment-standalone", response_model=ValuationResponse)
async def predict_apartment_standalone(req: ValuationRequest):
    return _predict(req, "apartment-standalone")

@app.post("/api/v1/predict/apartment-gated", response_model=ValuationResponse)
async def predict_apartment_gated(req: ValuationRequest):
    return _predict(req, "apartment-gated")

@app.post("/api/v1/predict/apartment-township", response_model=ValuationResponse)
async def predict_apartment_township(req: ValuationRequest):
    return _predict(req, "apartment-township")

@app.post("/api/v1/predict/villa", response_model=ValuationResponse)
async def predict_villa(req: ValuationRequest):
    return _predict(req, "villa")

@app.post("/api/v1/predict/plot", response_model=ValuationResponse)
async def predict_plot(req: ValuationRequest):
    return _predict(req, "plot")

@app.post("/api/v1/predict/commercial", response_model=ValuationResponse)
async def predict_commercial(req: ValuationRequest):
    return _predict(req, "commercial")

@app.post("/api/v1/predict/rental", response_model=ValuationResponse)
async def predict_rental(req: ValuationRequest):
    return _predict(req, "rental")

def _predict(req: ValuationRequest, model_key: str) -> ValuationResponse:
    model = MODELS.get(model_key)
    features = _build_features(req)

    if model is None:
        # Fallback heuristic until model is loaded
        base_psf = 8500.0
        fair_value = base_psf * req.area_sqft
        return ValuationResponse(
            fair_value=fair_value,
            price_range_min=fair_value * 0.92,
            price_range_max=fair_value * 1.08,
            confidence_score=0.45,
            valuation_tag="fair_value",
            shap_factors=[{"factor": "fallback_estimate", "contribution_pct": 100}],
            one_year_forecast=fair_value * 1.07,
            three_year_forecast=fair_value * 1.21,
            estimated_rent_min=fair_value * 0.003,
            estimated_rent_max=fair_value * 0.004,
            comparable_count=0,
            model_used="heuristic_fallback",
        )

    pred = model.predict([features])[0]
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values([features])[0]
    shap_factors = _build_shap_factors(shap_vals, features)

    confidence = float(np.clip(1.0 - (np.std(shap_vals) / (abs(pred) + 1e-6)), 0.5, 0.99))
    fair_value = float(pred)

    return ValuationResponse(
        fair_value=fair_value,
        price_range_min=fair_value * 0.94,
        price_range_max=fair_value * 1.06,
        confidence_score=confidence,
        valuation_tag=_tag(fair_value, req),
        shap_factors=shap_factors,
        one_year_forecast=fair_value * 1.07,
        three_year_forecast=fair_value * 1.21,
        estimated_rent_min=fair_value * 0.0032,
        estimated_rent_max=fair_value * 0.0045,
        comparable_count=50,
        model_used=f"lightgbm_{model_key}",
    )

def _build_features(req: ValuationRequest) -> list:
    furnishing_map = {"unfurnished": 0, "semifurnished": 1, "furnished": 2}
    facing_map = {"north": 3, "north-east": 2, "east": 1, "south": 0, "west": -1}
    return [
        req.area_sqft, req.bhk, req.floor, req.total_floors,
        req.age_years, furnishing_map.get(req.furnishing, 0),
        req.parking_spaces, req.amenity_count,
        facing_map.get(req.facing, 0), int(req.is_rera_approved),
        req.latitude, req.longitude,
    ]

def _build_shap_factors(shap_vals, features) -> list:
    labels = ["Area", "BHK", "Floor", "Total Floors", "Age",
              "Furnishing", "Parking", "Amenities", "Facing", "RERA", "Lat", "Lng"]
    total = sum(abs(v) for v in shap_vals) or 1
    return sorted([
        {"factor": labels[i], "contribution_pct": round(float(v / total) * 100, 1), "direction": "positive" if v > 0 else "negative"}
        for i, v in enumerate(shap_vals)
    ], key=lambda x: abs(x["contribution_pct"]), reverse=True)

def _tag(fair_value: float, req: ValuationRequest) -> str:
    if req.amenity_count > 20: return "premium_luxury"
    if fair_value > 20_000_000: return "premium_luxury"
    return "fair_value"
