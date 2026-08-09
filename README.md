# 🌴 SaruPol - AI & IoT Coconut Soil Intelligence Backend (R26-SE-016)

This repository contains the Backend Decision Support System and Machine Learning Engine for **"AI and IoT Based Intelligent Decision Support System for Coconut Plantation Monitoring and Advisory"** (Kaushalya P.L.P.D – IT22220424).

## 📌 System Architecture (2-Stage AI Pipeline)
1. **Stage 1 (Machine Learning Regression):** Maps real-time 7-in-1 IoT Sensor Soil NPK (`Soil N, P, K`) to **14th Frond Leaf NPK** (`Leaf N, P, K`) trained on empirical data from the **Coconut Research Institute (CRI)** Makandura Estate (`All.csv`, 135 palms).
2. **Stage 2 (CRI Rule-Based Expert System):** Evaluates Predicted Leaf NPK against official CRI Differential Fertilizer Recommendation (DFR) thresholds (Advisory Circular A5) to output precise chemical dosages in grams per palm per year:
   - **Urea** (Nitrogen supply)
   - **Eppawala Rock Phosphate - ERP** (Phosphorus supply)
   - **Muriate of Potash - MOP** (Potassium supply - critical for nut yield)
   - **Dolomite** (Soil pH neutralization & Magnesium supply)

---


## 🧪 API Endpoints

### 1. 3-Point Spatial Triangulated Sampling Prediction (Recommended for Research)
* **Endpoint:** `POST /api/v1/predict/triangulated`
* **Description:** Takes 3 sensor readings around the 1.8m manure circle (120° apart), calculates composite average to eliminate spatial heterogeneity, and outputs CRI fertilizer recommendation.
* **Request Payload Example:**
```json
{
  "tree_no": 30,
  "zone_id": "Zone A (Hilltop)",
  "point_a": { "N": 0.0159, "P": 0.3430, "K": 0.0629, "pH": 6.5 },
  "point_b": { "N": 0.0165, "P": 0.3390, "K": 0.0610, "pH": 6.4 },
  "point_c": { "N": 0.0150, "P": 0.3450, "K": 0.0650, "pH": 6.6 }
}
```

### 2. Single-Point Prediction
* **Endpoint:** `POST /api/v1/predict/single`
* **Request Payload Example:**
```json
{
  "tree_no": 30,
  "zone_id": "Zone B (Lowland)",
  "reading": { "N": 0.0159, "P": 0.3430, "K": 0.0629, "pH": 6.5 }
}
```
