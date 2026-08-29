import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

import json
import joblib #Machine Learning models save/load
from fastapi import FastAPI, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware #Frontend → Backend request allow
from typing import Dict, Any

from app.schemas import (
    TriangulatedSoilInput,
    PredictionResponse,
    AnalysisStartRequest,
    AnalysisStartResponse,
    PointReadingInput,
    AnalysisCompleteRequest,
    ImagePredictionResponse,
    LocationRequest,
    LocationResponse
)
from ml.cri_recommender import calculate_cri_fertilizer
from ml.train_models import train_and_evaluate_models
from ml.nutrient_predictor import predict_image
from ml.image_recommender import get_image_based_recommendation
import firebase_admin
from firebase_admin import credentials, firestore
import uuid
import urllib.request
import urllib.parse
from datetime import datetime, timezone

app = FastAPI(
    title="SaruPol - AI & IoT Coconut Soil Intelligence API",
    description="Backend Decision Support System for Coconut Plantation Monitoring and Advisory (R26-SE-016)",
    version="1.0.0"
)

# Enable CORS for Mobile App and Web Dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model holder
active_model_pipeline = None
active_model_name = "Not Loaded"
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, "ml", "saved_models", "best_soil_to_leaf_model.joblib")
report_path = os.path.join(base_dir, "ml", "saved_models", "model_comparison_report.json")

# Initialize Firebase
firebase_cred_path = os.path.join(base_dir, "firebase-credentials.json")
db = None
try:
    if os.path.exists(firebase_cred_path):
        cred = credentials.Certificate(firebase_cred_path)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("[OK] Firebase Initialized Successfully.")
    else:
        print("[WARN] firebase-credentials.json not found. Firebase features will fail.")
except Exception as e:
    print(f"[ERROR] Firebase init failed: {e}")

def load_active_model():
    global active_model_pipeline, active_model_name
    if os.path.exists(model_path):
        try:
            active_model_pipeline = joblib.load(model_path)
            if os.path.exists(report_path):
                with open(report_path, "r", encoding="utf-8") as f:
                    rep = json.load(f)  #model comparison report read
                    active_model_name = rep.get("best_model", "Trained ML Model")
            else:
                active_model_name = "Trained ML Model"
            print(f"[OK] Loaded ML Model: {active_model_name}")
        except Exception as e:
            print(f"[WARN] Failed to load model: {e}")
    else:
        print("[WARN] No saved model found. Use POST /api/v1/models/train to train models.")

@app.on_event("startup")
def on_startup():
    load_active_model()

@app.get("/")
def read_root():
    return {
        "project": "SaruPol - AI & IoT Coconut Soil Intelligence System",
        "research_id": "R26-SE-016",
        "status": "Online",
        "active_model": active_model_name,
        "endpoints": {
            "triangulated_prediction": "POST /api/v1/predict/triangulated",
            "train_models": "POST /api/v1/models/train",
            "model_status": "GET /api/v1/models/status"
        }
    }

@app.get("/health")
@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Coconut Soil Intelligence",
        "active_model": active_model_name,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/api/v1/models/status")
def get_model_status():
    if not os.path.exists(report_path):
        return {"status": "No model trained yet. Please call POST /api/v1/models/train"}
    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    return {
        "active_model_loaded": active_model_name,
        "comparison_report": rep
    }

@app.get("/api/v1/trees")
def get_all_trees():
    """
    Returns all empirical coconut tree samples from Makandura Estate (All.csv)
    along with their evaluated CRI health status.
    """
    data_path = os.path.join(base_dir, "data", "All.csv")
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail="All.csv dataset not found")
    
    import pandas as pd
    df = pd.read_csv(data_path)
    df.columns = [c.strip() for c in df.columns]
    
    trees = []
    healthy_cnt = 0
    deficient_cnt = 0
    excess_cnt = 0
    
    for idx, row in df.iterrows():
        try:
            t_no = int(row['Tree No'])
            s_n = float(row.get('Soil N', 0.015))
            s_p = float(row.get('Soil P', 0.15))
            s_k = float(row.get('Soil K', 0.06))
            l_n = float(row.get('Leaf N', 1.8)) if pd.notna(row.get('Leaf N')) else 1.8
            l_p = float(row.get('Leaf P', 0.12)) if pd.notna(row.get('Leaf P')) else 0.12
            l_k = float(row.get('Leaf K', 1.3)) if pd.notna(row.get('Leaf K')) else 1.3
            
            cri = calculate_cri_fertilizer(l_n, l_p, l_k)
            st = cri["status"]
            if "Optimal" in st:
                healthy_cnt += 1
            elif "Excess" in st:
                excess_cnt += 1
            else:
                deficient_cnt += 1
                
            trees.append({
                "tree_no": t_no,
                "soil_npk": {"N": round(s_n, 4), "P": round(s_p, 4), "K": round(s_k, 4)},
                "leaf_npk": {"N": round(l_n, 4), "P": round(l_p, 4), "K": round(l_k, 4)},
                "health_status": st,
                "fertilizer_recommendation": cri["dosages_grams_per_palm_per_year"]
            })
        except Exception:
            continue
            
    return {
        "estate_name": "CRI Makandura Research Station",
        "total_trees_analyzed": len(trees),
        "summary": {
            "optimal_health": healthy_cnt,
            "nutrient_deficiency": deficient_cnt,
            "excess_fertilizer": excess_cnt
        },
        "trees": trees
    }

@app.post("/api/v1/models/train")
def trigger_model_training():
    """
    Trains and evaluates 3 ML algorithms (Random Forest, Extra Trees, KNN) on All.csv.
    Selects and saves the model with the highest average R² score.
    """
    try:
        best_name, best_r2, results = train_and_evaluate_models()
        load_active_model()
        return {
            "message": "Model training and evaluation completed successfully!",
            "winning_model": best_name,
            "best_average_r2_score": best_r2,
            "detailed_comparison": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

def execute_prediction_pipeline(tree_no: int, sampling_method: str, avg_n: float, avg_p: float, avg_k: float, soil_ph: float = 6.5) -> PredictionResponse:
    global active_model_pipeline, active_model_name
    if active_model_pipeline is None:
        load_active_model()
        if active_model_pipeline is None:
            raise HTTPException(status_code=503, detail="ML Model is not loaded or trained yet. Please call POST /api/v1/models/train first.")
    
    # Stage 1: Predict 14th Leaf NPK using ML Model
    try:
        pred_leaf = active_model_pipeline.predict([[avg_n, avg_p, avg_k]])[0]
        leaf_n = round(float(pred_leaf[0]), 4)
        leaf_p = round(float(pred_leaf[1]), 4)
        leaf_k = round(float(pred_leaf[2]), 4)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction error: {str(e)}")

    # Stage 2: Apply CRI Rule-Based Expert System
    cri_result = calculate_cri_fertilizer(leaf_n, leaf_p, leaf_k, soil_ph)

    return PredictionResponse(
        tree_no=tree_no,
        sampling_method=sampling_method,
        average_soil_npk={"N": round(avg_n, 4), "P": round(avg_p, 4), "K": round(avg_k, 4)},
        predicted_14th_leaf_npk={"N": leaf_n, "P": leaf_p, "K": leaf_k},
        health_status=cri_result["status"],
        fertilizer_recommendation_grams_per_year=cri_result["dosages_grams_per_palm_per_year"],
        nutrient_evaluation=cri_result["nutrient_status"],
        agronomic_advice=cri_result["agronomic_advice"],
        model_used=active_model_name
    )

@app.post("/api/v1/predict/triangulated", response_model=PredictionResponse)
def predict_triangulated(data: TriangulatedSoilInput):
    """
    Recommended Research Protocol: 3-Point Triangulated Spatial Sampling.
    Takes 3 sensor readings around the 1.8m manure circle, calculates average, and predicts fertilizer dosage.
    """
    avg_n = (data.point_a.N + data.point_b.N + data.point_c.N) / 3.0
    avg_p = (data.point_a.P + data.point_b.P + data.point_c.P) / 3.0
    avg_k = (data.point_a.K + data.point_b.K + data.point_c.K) / 3.0
    
    # Use pH from point A or average
    soil_ph = data.point_a.pH if data.point_a.pH is not None else 6.5

    return execute_prediction_pipeline(
        tree_no=data.tree_no,
        sampling_method="3-Point Spatial Triangulated Sampling (Manure Circle Composite)",
        avg_n=avg_n,
        avg_p=avg_p,
        avg_k=avg_k,
        soil_ph=soil_ph
    )



# ---------------------------------------------------------
# IoT Firebase Integration Endpoints
# ---------------------------------------------------------

@app.post("/api/v1/analysis/start", response_model=AnalysisStartResponse)
def start_analysis(data: AnalysisStartRequest):
    if db is None:
        raise HTTPException(status_code=500, detail="Firebase not configured")
    
    analysis_id = f"AN-{data.tree_no}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{str(uuid.uuid4())[:4]}"
    
    doc_ref = db.collection("trees").document(data.tree_no).collection("analyses").document(analysis_id)
    doc_ref.set({
        "status": "in_progress",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "readings": {}
    })
    
    return AnalysisStartResponse(
        analysis_id=analysis_id,
        tree_no=data.tree_no,
        status="in_progress",
        message="Analysis session started successfully. Please capture Point 1."
    )

@app.post("/api/v1/analysis/reading")
def add_reading(data: PointReadingInput):
    if db is None:
        raise HTTPException(status_code=500, detail="Firebase not configured")
        
    if data.point_name not in ["point1", "point2", "point3"]:
        raise HTTPException(status_code=400, detail="point_name must be point1, point2, or point3")
        
    if not data.analysis_id or not data.analysis_id.strip():
        raise HTTPException(status_code=400, detail="analysis_id cannot be empty")
        
    doc_ref = db.collection("trees").document(data.tree_no).collection("analyses").document(data.analysis_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Analysis session not found.")
        
    reading_data = {
        "N": data.reading.N,
        "P": data.reading.P,
        "K": data.reading.K,
        "pH": data.reading.pH,
        "EC": data.reading.EC,
        "moisture": data.reading.moisture,
        "temperature": data.reading.temperature
    }
    
    doc_ref.update({
        f"readings.{data.point_name}": reading_data
    })
    
    return {"message": f"Successfully saved {data.point_name} for analysis {data.analysis_id}"}

@app.post("/api/v1/analysis/complete")
def complete_analysis(data: AnalysisCompleteRequest):
    if db is None:
        raise HTTPException(status_code=500, detail="Firebase not configured")
        
    if not data.analysis_id or not data.analysis_id.strip():
        raise HTTPException(status_code=400, detail=f"analysis_id cannot be empty. Received: '{data.analysis_id}'")
        
    doc_ref = db.collection("trees").document(data.tree_no).collection("analyses").document(data.analysis_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Analysis session not found.")
        
    doc_data = doc.to_dict()
    readings = doc_data.get("readings", {})
    
    if "point1" not in readings or "point2" not in readings or "point3" not in readings:
        keys_found = list(readings.keys())
        raise HTTPException(status_code=400, detail=f"Cannot complete analysis. Exact 3 valid readings required. Found: {keys_found}")
        
    # Calculate averages
    avg_n = sum(r["N"] for r in readings.values()) / 3.0
    avg_p = sum(r["P"] for r in readings.values()) / 3.0
    avg_k = sum(r["K"] for r in readings.values()) / 3.0
    avg_ph = sum(r.get("pH", 6.5) for r in readings.values()) / 3.0
    avg_ec = sum(r.get("EC", 1.0) for r in readings.values()) / 3.0
    avg_moist = sum(r.get("moisture", 50.0) for r in readings.values()) / 3.0
    avg_temp = sum(r.get("temperature", 28.0) for r in readings.values()) / 3.0
    
    # Predict & Calculate CRI using the existing logic!
    try:
        # 1. Use the trained global model pipeline for 14th leaf
        # We need a numeric tree_no for the old execute function, we can try to parse it or just use 0
        try:
            numeric_tree_no = int(''.join(filter(str.isdigit, data.tree_no)))
        except:
            numeric_tree_no = 0
            
        prediction_res = execute_prediction_pipeline(
            tree_no=numeric_tree_no,
            sampling_method="3-Point Spatial Triangulated IoT Sampling",
            avg_n=avg_n,
            avg_p=avg_p,
            avg_k=avg_k,
            soil_ph=avg_ph
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        
    # Build the final firebase document updates
    updates = {
        "status": "completed",
        "average": {
            "N": round(avg_n, 4),
            "P": round(avg_p, 4),
            "K": round(avg_k, 4),
            "pH": round(avg_ph, 2),
            "EC": round(avg_ec, 2),
            "moisture": round(avg_moist, 2),
            "temperature": round(avg_temp, 2)
        },
        "prediction": {
            "leafN": prediction_res.predicted_14th_leaf_npk["N"],
            "leafP": prediction_res.predicted_14th_leaf_npk["P"],
            "leafK": prediction_res.predicted_14th_leaf_npk["K"],
            "modelVersion": prediction_res.model_used
        },
        "recommendation": {
            "urea": prediction_res.fertilizer_recommendation_grams_per_year.get("Urea", 0) / 1000.0,
            "ERP": prediction_res.fertilizer_recommendation_grams_per_year.get("Eppawala_Rock_Phosphate_ERP", 0) / 1000.0,
            "MOP": prediction_res.fertilizer_recommendation_grams_per_year.get("Muriate_of_Potash_MOP", 0) / 1000.0,
            "dolomite": prediction_res.fertilizer_recommendation_grams_per_year.get("Dolomite", 0) / 1000.0,
            "healthStatus": prediction_res.health_status,
            "nutrientEvaluation": prediction_res.nutrient_evaluation,
            "agronomicAdvice": prediction_res.agronomic_advice
        }
    }
    
    doc_ref.update(updates)
    
    # Return everything to the caller
    return updates

@app.post("/api/v1/nutrient-analysis/predict", response_model=ImagePredictionResponse)
async def predict_nutrient_from_image(image: UploadFile = File(...)):
    """
    Analyzes an uploaded coconut leaf image to predict potential nutrient deficiencies
    (Nitrogen, Boron) based on visual features. This provides a preliminary visual assessment.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")
        
    try:
        contents = await image.read()
        result = predict_image(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")
        
    status = result["status"]
    
    # If validation failed (invalid_input or uncertain)
    if status != "success":
        return {
            "success": True,
            "status": status,
            "message": result["message"],
            "prediction": None,
            "recommendation": None,
            "visual_features": result.get("features", {})
        }
        
    predicted_class = result["prediction"]
    confidence = result["confidence"]
    visual_features = result["features"]
    
    # Get image-specific recommendation
    # We use a threshold of 0.60 as specified
    rec_result = get_image_based_recommendation(
        predicted_nutrient=predicted_class,
        confidence=confidence,
        threshold=0.60
    )
    
    # Merge the visual features into the response for transparency
    rec_result["visual_features"] = visual_features
    
    return rec_result

@app.post("/api/v1/location/agro-zone", response_model=LocationResponse)
def get_agro_climatic_zone(req: LocationRequest):
    lat = req.latitude
    lon = req.longitude
    
    # 1. Validate coordinates bounds
    if not (-90 <= lat <= 90):
        return LocationResponse(success=False, message=f"Invalid latitude: {lat}")
    if not (-180 <= lon <= 180):
        return LocationResponse(success=False, message=f"Invalid longitude: {lon}")
        
    # 2. Query NSDI GIS Service
    url = "https://gisapps.nsdi.gov.lk/server/rest/services/Srilanka/All_Layers/MapServer/111/query"
    params = {
        "geometryType": "esriGeometryPoint",
        "geometry": f"{lon},{lat}",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "zone,climatic_zone,agro_eco_zone,agro_eco_r",
        "returnGeometry": "false",
        "f": "json"
    }
    
    query_string = urllib.parse.urlencode(params)
    full_url = f"{url}?{query_string}"
    
    try:
        req_obj = urllib.request.Request(full_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req_obj, timeout=10.0) as response:
            data = json.loads(response.read().decode('utf-8'))
    except Exception as e:
        return LocationResponse(success=False, message=f"NSDI Service Failure: {str(e)}")
        
    # 3. Handle response format errors
    if "error" in data:
        return LocationResponse(success=False, message=f"NSDI API Error: {data['error'].get('message', 'Unknown Error')}", raw_attributes=data)
        
    features = data.get("features", [])
    if not features:
        return LocationResponse(
            success=True, 
            zone=None, 
            agro_ecological_zone=None, 
            message="No agro-ecological zone found for these coordinates. (Might be ocean or outside boundary).",
            raw_attributes=None
        )
        
    # 4. Normalize based on priority (climatic_zone -> agro_eco_zone prefix)
    attrs = features[0].get("attributes", {})
    climatic_zone = attrs.get("climatic_zone", "")
    agro_eco_zone = attrs.get("agro_eco_zone", "")
    
    major_zone = None
    
    if climatic_zone:
        cz_upper = str(climatic_zone).strip().upper()
        if "WET" in cz_upper:
            major_zone = "Wet"
        elif "INTERMEDIATE" in cz_upper:
            major_zone = "Intermediate"
        elif "DRY" in cz_upper:
            major_zone = "Dry"
            
    # Fallback to prefix if climatic_zone doesn't contain expected keywords
    if not major_zone and agro_eco_zone:
        a_upper = str(agro_eco_zone).strip().upper()
        if a_upper.startswith("W"):
            major_zone = "Wet"
        elif a_upper.startswith("I"):
            major_zone = "Intermediate"
        elif a_upper.startswith("D"):
            major_zone = "Dry"
            
    return LocationResponse(
        success=True,
        zone=major_zone,
        agro_ecological_zone=agro_eco_zone,
        message="Zone successfully detected." if major_zone else "Zone detected but major zone format is unrecognized.",
        raw_attributes=attrs
    )

