# ============================================================
# WATER QUALITY FORECASTING API
# FastAPI Backend — Zambia Water Quality System
# Mike Machayi | ZCAS University
# ============================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import tensorflow as tf
from datetime import datetime
import uvicorn

# ============================================================
# LOAD ALL SAVED MODELS
# ============================================================

print("Loading models...")

rf_model      = joblib.load('rf_model.pkl')
xgb_model     = joblib.load('xgb_model.pkl')
scaler        = joblib.load('scaler.pkl')
lstm_model    = tf.keras.models.load_model('lstm_model.keras')
cnn_lstm_model = tf.keras.models.load_model('cnn_lstm_model.keras')

print("✅ All models loaded successfully")

# ============================================================
# CREATE FASTAPI APP
# ============================================================

app = FastAPI(
    title="Water Quality Forecasting API",
    description="ML-Based Water Quality Early Warning System — Zambia",
    version="1.0.0"
)

# Allow Flutter app to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# DATA MODEL — what the app sends to the API
# ============================================================

class WaterReading(BaseModel):
    pH: float
    turbidity: float
    temperature: float
    do: float           # dissolved oxygen
    bod: float          # biological oxygen demand
    lead: float
    mercury: float
    arsenic: float
    cu: float           # copper
    co: float           # cobalt
    mn: float           # manganese
    ni: float           # nickel
    u: float            # uranium
    hour_sin: float
    hour_cos: float
    day_of_week: int
    month: int
    location_code: int  # 0=Kafue, 1=Zambezi, 2=Luangwa, 3=Chambeshi, 4=Kariba

# ============================================================
# HELPER FUNCTIONS
# ============================================================

# Convert prediction number to risk label
def get_risk_label(prediction: int) -> str:
    labels = {0: 'Low', 1: 'Moderate', 2: 'High'}
    return labels.get(prediction, 'Unknown')

# Convert risk label to color for Flutter app
def get_risk_color(prediction: int) -> str:
    colors = {0: 'green', 1: 'orange', 2: 'red'}
    return colors.get(prediction, 'grey')

# Get recommendation based on risk level
def get_recommendation(prediction: int, location: int) -> str:
    locations = ['Kafue River', 'Zambezi River', 'Luangwa River',
                 'Chambeshi River', 'Lake Kariba']
    loc_name = locations[location] if location < len(locations) else 'Unknown'

    if prediction == 0:
        return f"{loc_name}: Water quality is within safe limits. No action required."
    elif prediction == 1:
        return f"{loc_name}: Moderate contamination detected. Monitor closely and notify field teams."
    else:
        return f"{loc_name}: HIGH RISK detected. Deploy emergency response immediately. Do not use water."

# Prepare input features for model
def prepare_features(reading: WaterReading):
    return np.array([[
        reading.pH,
        reading.turbidity,
        reading.temperature,
        reading.do,
        reading.bod,
        reading.lead,
        reading.mercury,
        reading.arsenic,
        reading.cu,
        reading.co,
        reading.mn,
        reading.ni,
        reading.u,
        reading.hour_sin,
        reading.hour_cos,
        reading.day_of_week,
        reading.month,
        reading.location_code
    ]])

# ============================================================
# API ENDPOINTS
# ============================================================

# --- Root endpoint ---
@app.get("/")
def root():
    return {
        "message": "Water Quality Forecasting API",
        "status": "running",
        "version": "1.0.0",
        "models": ["Random Forest", "XGBoost", "LSTM", "CNN-LSTM"]
    }

# --- Health check ---
@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": str(datetime.now())}

# --- Random Forest prediction ---
@app.post("/predict/rf")
def predict_rf(reading: WaterReading):
    features = prepare_features(reading)
    features_scaled = scaler.transform(features)
    prediction = int(rf_model.predict(features_scaled)[0])
    probability = rf_model.predict_proba(features_scaled)[0].tolist()

    return {
        "model": "Random Forest",
        "accuracy": "99.82%",
        "prediction": prediction,
        "risk_level": get_risk_label(prediction),
        "color": get_risk_color(prediction),
        "confidence": round(max(probability) * 100, 2),
        "recommendation": get_recommendation(prediction, reading.location_code)
    }

# --- XGBoost prediction ---
@app.post("/predict/xgb")
def predict_xgb(reading: WaterReading):
    features = prepare_features(reading)
    features_scaled = scaler.transform(features)
    prediction = int(xgb_model.predict(features_scaled)[0])
    probability = xgb_model.predict_proba(features_scaled)[0].tolist()

    return {
        "model": "XGBoost",
        "accuracy": "99.45%",
        "prediction": prediction,
        "risk_level": get_risk_label(prediction),
        "color": get_risk_color(prediction),
        "confidence": round(max(probability) * 100, 2),
        "recommendation": get_recommendation(prediction, reading.location_code)
    }

# --- Best model prediction (CNN-LSTM for forecasting) ---
@app.post("/predict/best")
def predict_best(reading: WaterReading):
    features = prepare_features(reading)
    features_scaled = scaler.transform(features)

    # Use Random Forest as primary (highest accuracy)
    prediction = int(rf_model.predict(features_scaled)[0])
    probability = rf_model.predict_proba(features_scaled)[0].tolist()

    return {
        "model": "Random Forest (Best)",
        "accuracy": "99.82%",
        "prediction": prediction,
        "risk_level": get_risk_label(prediction),
        "color": get_risk_color(prediction),
        "confidence": round(max(probability) * 100, 2),
        "recommendation": get_recommendation(prediction, reading.location_code)
    }

# --- Get all sites current status ---
@app.get("/sites")
def get_sites():
    # Simulated current readings for each site
    sites = [
        {
            "id": 0,
            "name": "Kafue River",
            "catchment": "Kafue Catchment",
            "ph": 6.2,
            "turbidity": 45.0,
            "copper": 3.2,
            "risk_level": "High",
            "color": "red",
            "safe": False,
            "message": "High copper levels detected. Do not use this water.",
            "last_updated": str(datetime.now())
        },
        {
            "id": 1,
            "name": "Zambezi River",
            "catchment": "Zambezi Catchment",
            "ph": 7.1,
            "turbidity": 12.0,
            "copper": 0.8,
            "risk_level": "Low",
            "color": "green",
            "safe": True,
            "message": "Water quality is within safe limits.",
            "last_updated": str(datetime.now())
        },
        {
            "id": 2,
            "name": "Luangwa River",
            "catchment": "Luangwa Catchment",
            "ph": 6.8,
            "turbidity": 28.0,
            "copper": 1.9,
            "risk_level": "Moderate",
            "color": "orange",
            "safe": False,
            "message": "Use with caution. Slightly elevated turbidity.",
            "last_updated": str(datetime.now())
        },
        {
            "id": 3,
            "name": "Chambeshi River",
            "catchment": "Chambeshi Catchment",
            "ph": 5.8,
            "turbidity": 62.0,
            "copper": 5.1,
            "risk_level": "Critical",
            "color": "red",
            "safe": False,
            "message": "CRITICAL: Severe contamination. Avoid all contact.",
            "last_updated": str(datetime.now())
        },
        {
            "id": 4,
            "name": "Lake Kariba",
            "catchment": "Zambezi Catchment",
            "ph": 7.4,
            "turbidity": 8.0,
            "copper": 0.3,
            "risk_level": "Low",
            "color": "green",
            "safe": True,
            "message": "Water quality is good. Safe for use.",
            "last_updated": str(datetime.now())
        }
    ]
    return {"sites": sites, "total": len(sites)}

# --- Get forecast for a site ---
@app.get("/forecast/{site_id}")
def get_forecast(site_id: int):
    # Simulated forecasts — in production these come from LSTM/CNN-LSTM
    forecasts = {
        0: {  # Kafue River
            "site": "Kafue River",
            "forecasts": [
                {"horizon": "1 Hour",  "risk": "High",     "ph": 6.0, "turbidity": 48.0, "copper": 3.5, "recommendation": "Alert field team immediately"},
                {"horizon": "6 Hours", "risk": "High",     "ph": 5.8, "turbidity": 55.0, "copper": 4.1, "recommendation": "Contamination spreading downstream"},
                {"horizon": "24 Hours","risk": "Critical", "ph": 5.2, "turbidity": 72.0, "copper": 6.8, "recommendation": "Deploy emergency response teams"}
            ]
        },
        1: {  # Zambezi River
            "site": "Zambezi River",
            "forecasts": [
                {"horizon": "1 Hour",  "risk": "Low",      "ph": 7.1, "turbidity": 11.0, "copper": 0.7, "recommendation": "No action required"},
                {"horizon": "6 Hours", "risk": "Low",      "ph": 7.2, "turbidity": 10.0, "copper": 0.6, "recommendation": "Conditions stable"},
                {"horizon": "24 Hours","risk": "Low",      "ph": 7.3, "turbidity": 9.0,  "copper": 0.5, "recommendation": "Water quality improving"}
            ]
        },
        2: {  # Luangwa River
            "site": "Luangwa River",
            "forecasts": [
                {"horizon": "1 Hour",  "risk": "Moderate", "ph": 6.7, "turbidity": 30.0, "copper": 2.0, "recommendation": "Monitor closely"},
                {"horizon": "6 Hours", "risk": "Moderate", "ph": 6.6, "turbidity": 33.0, "copper": 2.2, "recommendation": "Notify field teams"},
                {"horizon": "24 Hours","risk": "High",     "ph": 6.3, "turbidity": 40.0, "copper": 2.8, "recommendation": "Prepare response teams"}
            ]
        },
        3: {  # Chambeshi River
            "site": "Chambeshi River",
            "forecasts": [
                {"horizon": "1 Hour",  "risk": "Critical", "ph": 5.6, "turbidity": 68.0, "copper": 5.8, "recommendation": "IMMEDIATE ACTION REQUIRED"},
                {"horizon": "6 Hours", "risk": "Critical", "ph": 5.3, "turbidity": 75.0, "copper": 6.5, "recommendation": "Emergency response deployed"},
                {"horizon": "24 Hours","risk": "Critical", "ph": 5.0, "turbidity": 85.0, "copper": 7.2, "recommendation": "Sustained emergency response needed"}
            ]
        },
        4: {  # Lake Kariba
            "site": "Lake Kariba",
            "forecasts": [
                {"horizon": "1 Hour",  "risk": "Low", "ph": 7.4, "turbidity": 7.5, "copper": 0.3, "recommendation": "No action required"},
                {"horizon": "6 Hours", "risk": "Low", "ph": 7.5, "turbidity": 7.0, "copper": 0.2, "recommendation": "Conditions stable"},
                {"horizon": "24 Hours","risk": "Low", "ph": 7.5, "turbidity": 6.5, "copper": 0.2, "recommendation": "Water quality excellent"}
            ]
        }
    }
    return forecasts.get(site_id, {"error": "Site not found"})

# --- Get all active alerts ---
@app.get("/alerts")
def get_alerts():
    alerts = [
        {
            "id": 1,
            "title": "CRITICAL: Copper Spike Detected",
            "location": "Chambeshi River",
            "level": "CRITICAL",
            "color": "red",
            "time": "2 minutes ago",
            "detail": "Copper levels at 5.1 mg/L — WHO limit is 2.0 mg/L. Immediate action required.",
            "active": True
        },
        {
            "id": 2,
            "title": "HIGH RISK: Low pH Detected",
            "location": "Kafue River",
            "level": "HIGH",
            "color": "orange",
            "time": "18 minutes ago",
            "detail": "pH dropped to 6.2. Possible acid mine drainage upstream.",
            "active": True
        },
        {
            "id": 3,
            "title": "WARNING: Turbidity Rising",
            "location": "Luangwa River",
            "level": "MODERATE",
            "color": "amber",
            "time": "1 hour ago",
            "detail": "Turbidity at 28 NTU and rising. Monitor closely.",
            "active": True
        },
        {
            "id": 4,
            "title": "INFO: Readings Back to Normal",
            "location": "Zambezi River",
            "level": "NORMAL",
            "color": "green",
            "time": "3 hours ago",
            "detail": "All parameters within WHO safe limits.",
            "active": False
        }
    ]
    return {"alerts": alerts, "active_count": 3}

# ============================================================
# RUN THE API
# ============================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
