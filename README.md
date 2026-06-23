# AP Rainfall Early Warning System

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-92.2%25_Accuracy-green)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A production-level monsoon rainfall early warning system for Andhra Pradesh that predicts village-level rainfall, identifies at-risk infrastructure, and sends automated SMS alerts — built with machine learning, live weather APIs, and geospatial analysis.

---

## Problem Statement

Andhra Pradesh faces severe annual flooding during monsoon season (June–October), causing damage to irrigation canals, embankments, and thousands of villages. Existing systems lack village-level granularity and real-time alerting. This system addresses that gap by predicting daily rainfall for 15,589 villages and identifying which canals and embankments are at flood risk.

---

## Key Features

- **92.2% prediction accuracy** — XGBoost + Random Forest ensemble trained on 9.5M records
- **Village-level granularity** — covers all 15,589 villages across all AP districts
- **Multi-signal confidence system** — ML model + Live weather + ENSO + NWP forecast + Climatology
- **Physical meteorology gate** — blocks physically impossible predictions (e.g. rain at 41°C, 14% humidity)
- **ENSO climate patterns** — El Niño / La Niña year-factor adjustments for better long-range forecasts
- **Real-time weather** — live temperature, humidity, pressure, wind from OpenWeatherMap API
- **Spatial infrastructure risk** — GeoPandas spatial join identifies canals and embankments within 10km
- **Automated SMS alerts** — Twilio SMS fired when rainfall crosses smart auto-threshold
- **Interactive dashboard** — Streamlit web app with Folium maps, Plotly charts, alert history
- **Prediction logging** — SQLite database stores all predictions and alerts with timestamps

---

## System Architecture
┌─────────────────────────────────────────────────────────┐

│                    Streamlit Dashboard                   │

│   Predict & Alert │ Live Map │ Analytics │ Alert History │

└────────────────────────┬────────────────────────────────┘

│

┌──────────────┼──────────────┐

▼              ▼              ▼

┌──────────┐  ┌──────────────┐  ┌──────────────┐

│ ML Model │  │ OpenWeather  │  │  GeoPandas   │

│ XGBoost  │  │   Map API    │  │  Spatial     │

│ RF Clf   │  │ Live Weather │  │  Analysis    │

└──────────┘  └──────────────┘  └──────────────┘

│              │              │

▼              ▼              ▼

┌──────────────────────────────────────────────┐

│           Confidence Analysis Engine          │

│  ML Signal + Humidity + NWP + Climatology    │

│  + ENSO Pattern + Physical Meteorology Gate  │

└──────────────────────┬───────────────────────┘

│

┌───────────┴───────────┐

▼                       ▼

┌─────────────┐        ┌──────────────┐

│   SQLite DB  │        │  Twilio SMS  │

│  Predictions │        │    Alerts    │

│  & Alerts    │        └──────────────┘

└─────────────┘

---

## Model Performance

| Metric | V1 (Baseline) | V2 (+ Weather) | V3 (+ ENSO) |
|--------|--------------|----------------|-------------|
| Accuracy | 79.1% | 83.0% | 92.2% |
| MAE | 4.58 mm | 4.33 mm | 3.72 mm |
| RMSE | 7.97 mm | 7.50 mm | 6.38 mm |
| Features | 9 | 13 | 16 |

---

## ML Pipeline
Raw Data (9.5M rows)

│

▼

Data Cleaning & Monsoon Filter (Jun-Oct)

│

▼

Feature Engineering

├── Temporal: year, month, day, dayofyear

├── Lag features: rain_lag1, rain_lag7, rain_rolling7

├── Weather: humidity, temperature, pressure, wind

├── ENSO: year_factor (La Nina/El Nino adjustment)

└── Monsoon: monsoon_intensity, monsoon_day

│

▼

Model Training (5% sample = 476K rows)

├── XGBoost Regressor → Rainfall amount (mm)

└── Random Forest Classifier → Will it rain? (Yes/No)

│

▼

Multi-Signal Confidence Engine

├── Signal 1: ML Model prediction

├── Signal 2: Live humidity check (>70% = rain likely)

├── Signal 3: NWP 24hr forecast (for near-term dates)

├── Signal 4: Seasonal climatology (IMD normals)

└── Signal 5: ENSO pattern (for long-range dates)

│

▼

Physical Meteorology Gate

└── Blocks: temp > 42°C, humidity < 35%, etc.

---

## 🗺 GIS Data

| File | Description |
|------|-------------|
| Canals.shp | AP irrigation canal network |
| Embankments.shp | AP flood embankment network |

Used for spatial join to identify infrastructure within 10km of predicted high-rainfall zones.

---

## Tech Stack

| Category | Technology |
|----------|------------|
| ML Models | XGBoost, Random Forest (scikit-learn) |
| Data Processing | Pandas, NumPy |
| Geospatial | GeoPandas, Shapely, Folium |
| Web App | Streamlit |
| Charts | Plotly |
| Live Weather | OpenWeatherMap API |
| SMS Alerts | Twilio API |
| Database | SQLite (SQLAlchemy) |
| Language | Python 3.11 |

---

## Project Structure
Rain_prediction/

├── app.py                    # Main Streamlit application
├── reg_model.pkl             # XGBoost rainfall amount model
├── clf_model.pkl             # Random Forest rain/no-rain classifier
├── village_predictions.csv   # Precomputed village predictions
├── features.json             # Feature list for model inference
├── rainfall.db               # SQLite predictions and alerts database
├── Canals.shp                # Canal network shapefile
├── Canals.dbf                # Canal attributes
├── Canals.shx                # Canal index
├── Canals.prj                # Canal projection
├── Embankments.shp           # Embankment network shapefile
├── Embankments.dbf           # Embankment attributes
├── Embankments.shx           # Embankment index
├── Embankments.prj           # Embankment projection
├── requirements.txt          # Python dependencies
├── .gitignore                # Excluded files
└── README.md                 # This file

---

## Setup and Installation

### Prerequisites
- Python 3.11+
- Git

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/AP-Rainfall-Early-Warning-System.git
cd AP-Rainfall-Early-Warning-System
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API keys
Open `app.py` and update these lines with your keys:
```python
OWM_API_KEY  = "your_openweathermap_api_key"
TWILIO_SID   = "your_twilio_account_sid"
TWILIO_TOKEN = "your_twilio_auth_token"
TWILIO_FROM  = "your_twilio_phone_number"
ALERT_PHONE  = "your_mobile_number"
```

### 4. Run the app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## App Screenshots

| Tab | Description |
|-----|-------------|
| 🎯 Predict & Alert | Select village, run prediction, see confidence analysis |
| 🗺 Live Map | Color-coded risk map across all AP villages |
| 📊 Analytics | District rainfall charts, risk distribution, history |
| 🔔 Alert History | Log of all HIGH and EXTREME alerts triggered |

---

## Confidence Analysis System

The system uses multiple independent signals to verify predictions:

| Signal | When Used | Source |
|--------|-----------|--------|
| ML Model | Always | Trained XGBoost model |
| Live Humidity | Today / Tomorrow | OpenWeatherMap API |
| NWP Forecast | Next 48 hours | OpenWeatherMap Forecast |
| Seasonal Climatology | Medium / Long range | IMD historical normals |
| ENSO Pattern | Long range | El Niño / La Niña factors |
| Bay of Bengal Monsoon | Monsoon months | Meteorological knowledge |
| Physical Gate | Always | Temperature + Humidity rules |

**Confidence levels:**
- 🟢 HIGH (67-100%) — 2-3 signals agree — SMS alert fired
- 🟡 MODERATE (34-66%) — majority agree — monitor closely
- 🔴 LOW (0-33%) — signals disagree — human review recommended

---

## Known Limitations

1. **Monsoon-only training** — model trained on Jun-Oct data. Non-monsoon predictions use seasonal norms.
2. **Historical patterns** — cannot predict truly unseasonal events (cyclones, western disturbances).
3. **Lag features** — use historical averages as proxy for actual recent rainfall (Phase 2: real-time IMD gauge data).
4. **Year variability** — ENSO factors are estimates based on climatological patterns, not real-time ENSO indices.

---

## Future Enhancements (Phase 2)

- [ ] Real-time IMD rain gauge API integration for accurate lag features
- [ ] Cyclone track integration for surprise event detection
- [ ] District-level administrative boundary overlays
- [ ] Mobile app with push notifications
- [ ] Scheduled auto-prediction every 6 hours via APScheduler
- [ ] Cloud deployment on AWS/GCP

---

## Author

**Rounak**
- GitHub: [rounak2601](https://github.com/rounak2601)
- LinkedIn: [rounak_tilante](https://www.linkedin.com/in/rounak-tilante-a9719b257/)

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

- India Meteorological Department (IMD) for historical rainfall data
- OpenWeatherMap for live weather and NWP forecast API
- Twilio for SMS alert infrastructure
- Government of Andhra Pradesh for GIS shapefiles

