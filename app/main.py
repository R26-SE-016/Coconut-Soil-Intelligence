import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

import json
from fastapi import FastAPI, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

from app.schemas import (
    ImagePredictionResponse,
    LocationRequest,
    LocationResponse,
    SaveNutrientScanRequest,
    LabRecommendationRequest,
    LabRecommendationResponse
)
from ml.nutrient_predictor import predict_image
from ml.image_recommender import get_image_based_recommendation
import firebase_admin
from firebase_admin import credentials, firestore
import urllib.request
import urllib.parse
from datetime import datetime, timezone

app = FastAPI(
    title="SaruPol - AI Coconut Nutrient & Lab Intelligence API",
    description="Backend Decision Support System for Coconut Leaf Nutrient Analysis and Laboratory Advisory",
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

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Initialize Firebase
# Supports: (1) local firebase-credentials.json file, or (2) FIREBASE_CREDENTIALS_JSON env var (for Render)
firebase_cred_path = os.path.join(base_dir, "firebase-credentials.json")
db = None
try:
    cred = None
    if os.path.exists(firebase_cred_path):
        # Local development: use credentials file
        cred = credentials.Certificate(firebase_cred_path)
        print("[INFO] Firebase: Loading credentials from file.")
    elif os.environ.get("FIREBASE_CREDENTIALS_JSON"):
        # Render / Cloud deployment: use environment variable
        import json
        cred_dict = json.loads(os.environ["FIREBASE_CREDENTIALS_JSON"])
        cred = credentials.Certificate(cred_dict)
        print("[INFO] Firebase: Loading credentials from environment variable.")
    
    if cred:
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("[OK] Firebase Initialized Successfully.")
    else:
        print("[WARN] Firebase credentials not found (no file or env var). Firebase features will fail.")
except Exception as e:
    print(f"[ERROR] Firebase init failed: {e}")

def seed_deficiencies_if_empty():
    if db is None:
        print("[WARN] Firebase DB is not initialized. Skipping deficiencies seeding.")
        return
    try:
        deficiencies_ref = db.collection("deficiencies")
        docs = deficiencies_ref.limit(1).get()
        if len(docs) == 0:
            print("[INFO] Seeding deficiencies to Firestore...")
            default_deficiencies = [
                {
                    "id": "nitrogen",
                    "nameEn": "Nitrogen Deficiency",
                    "chemicalSymbol": "N",
                    "criticalRange": "1.80% - 2.00%",
                    "overview": "Nitrogen is a primary macronutrient essential for vegetative growth, leaf production, and chlorophyll synthesis. When deficient, the palm cannot photosynthesize efficiently, leading to reduced vigor and stunted growth.",
                    "symptoms": [
                        "General yellowing (chlorosis) of the older leaves first, which slowly spreads to the younger ones.",
                        "Leaflets turn pale green to golden yellow.",
                        "The growth rate of the palm slows down, and fronds become shorter.",
                        "Thin crowns and slender trunks develop over time.",
                        "Nut size and yield drop significantly."
                    ],
                    "causes": [
                        "Acidic soil conditions which limit nitrogen availability.",
                        "Heavy leaching in sandy/gravelly soils during monsoon seasons.",
                        "Low soil organic matter and poor biological activity."
                    ],
                    "correctiveMeasures": [
                        "Apply an additional 100-200g of Urea per palm depending on the growth stage (under CRI A7 Guidelines).",
                        "Incorporate organic manure or compost around the manure circle to naturally raise soil organic matter.",
                        "Grow cover crops (like Mucuna bracteata) in the interspaces to fix atmospheric nitrogen.",
                        "Practice proper mulching with coconut husks in the 1.8m manure circle to retain soil moisture and reduce nitrogen volatilization."
                    ],
                    "themeColor": "#4CAF50",
                    "description": "Nitrogen deficiency causes general yellowing (chlorosis) of older leaves first, progressing to the younger leaves. The growth rate slows down, fronds become shorter, and the crown becomes thin, significantly dropping nut size and yield.",
                    "advice": "Apply an additional 100-200g of Urea per palm depending on the growth stage. Incorporate organic manure, compost, or cover crops (like Mucuna) to naturally raise soil organic matter and mulch the base."
                },
                {
                    "id": "potassium",
                    "nameEn": "Potassium Deficiency",
                    "chemicalSymbol": "K",
                    "criticalRange": "1.20% - 1.50%",
                    "overview": "Potassium is the most heavily extracted nutrient by coconut palms. It regulates stomatal opening, water relations, carbohydrate translocation, and directly influences nut size, weight, and copra quality.",
                    "symptoms": [
                        "Orange-yellow chlorotic spots appear on older leaves first.",
                        "Leaflet margins and tips show necrosis (scorching or burning) that moves inwards.",
                        "Midribs and petioles become weak, causing older fronds to hang down or break prematurely.",
                        "Yield decreases rapidly with smaller nut sizes and thin, fiberless husks.",
                        "Increased susceptibility to droughts and pest attacks."
                    ],
                    "causes": [
                        "Highly leached sandy or gravelly soils where potassium is easily washed away.",
                        "Acidic soils or soils with low cation exchange capacity.",
                        "Harvesting nuts repeatedly without replacing the extracted potassium."
                    ],
                    "correctiveMeasures": [
                        "Apply an additional 500g of Muriate of Potash (MOP) per adult palm per year (CRI A7 Guidance).",
                        "Bury coconut husks and fronds in trenches between rows (husk burial). Coconut husks are rich in potassium and store moisture.",
                        "Ensure balanced fertilizer application since excess calcium/magnesium can inhibit potassium uptake."
                    ],
                    "themeColor": "#FF9800",
                    "description": "Potassium deficiency is common in sandy soils. It shows as orange-yellow chlorotic spots on older leaves, with leaflet margins and tips exhibiting burning/necrosis. Midribs weaken and fronds hang down or break prematurely.",
                    "advice": "Apply an additional 500g of Muriate of Potash (MOP) per adult palm per year. Bury coconut husks and fronds in trenches between rows to recycle potassium and preserve moisture."
                },
                {
                    "id": "magnesium",
                    "nameEn": "Magnesium Deficiency",
                    "chemicalSymbol": "Mg",
                    "criticalRange": "0.20% - 0.35%",
                    "overview": "Magnesium is the central component of the chlorophyll molecule, making it essential for photosynthesis. Deficiency leads to direct yellowing of mature leaves and significantly reduces starch synthesis.",
                    "symptoms": [
                        "Classic \"V-shaped\" yellowing on older leaves; leaflet margins turn bright orange-yellow while the area near the midrib remains green.",
                        "Translucent yellow spotting on leaflets exposed to direct sunlight.",
                        "Leaf tips become necrotic and die back in severe stages.",
                        "Healthy young green leaves are only found at the center of the crown."
                    ],
                    "causes": [
                        "Highly acidic, sandy soils prone to leaching.",
                        "Excessive application of Potassium or Ammonium fertilizers which competitively inhibits Magnesium uptake."
                    ],
                    "correctiveMeasures": [
                        "For severe cases: Apply 1 kg of Kieserite (Magnesium Sulphate) per adult palm half-yearly. Apply Kieserite to one half of the manure circle and NPK to the other half.",
                        "For young palms showing symptoms: Apply 0.5 kg of Kieserite half-yearly.",
                        "For long-term prevention: Apply 1 kg of Dolomite per palm per year. Apply dolomite at least 2 weeks before or after applying chemical fertilizers."
                    ],
                    "themeColor": "#009688",
                    "description": "Magnesium deficiency is characterized by a V-shaped yellowing on older leaves, where leaflet margins turn bright orange-yellow while the midrib area remains green. Photosynthesis is severely reduced, affecting root and nut growth.",
                    "advice": "For severe cases: Apply 1 kg Kieserite (Magnesium Sulphate) per adult palm half-yearly (apply NPK to one half of the circle and Kieserite to the other). For long-term prevention, apply 1 kg Dolomite per palm per year."
                },
                {
                    "id": "boron",
                    "nameEn": "Boron Deficiency",
                    "chemicalSymbol": "B",
                    "criticalRange": "8 - 10 ppm",
                    "overview": "Boron is a vital micronutrient required for cell division, cell wall development, pollen germination, and sugar transport. Deficiency causes severe malformations in growing tissues.",
                    "symptoms": [
                        "\"Hook Leaf\": Young emerging fronds show leaflets with bent, rigid tips that cannot be straightened.",
                        "Spear leaves fail to open properly or appear crinkled (\"crown choke\").",
                        "The crown may exhibit a zigzag or serrated silhouette.",
                        "Deformed, flat-sided, or undersized nuts (barren nuts).",
                        "Severe button shedding and necrotic inflorescence."
                    ],
                    "causes": [
                        "Leached sandy soils or highly alkaline soils.",
                        "Extended drought periods which restrict water movement and boron transport in the soil.",
                        "Imbalanced soil chemistry."
                    ],
                    "correctiveMeasures": [
                        "Apply 20g of Sodium Tetraborate (Borax) per mature or young palm at 6-month intervals until symptoms disappear (CRI A7 Guidance).",
                        "For seedlings: Apply 10g of Borax at 6-month intervals.",
                        "Ensure the soil is moist during application to facilitate uptake.",
                        "Caution: Apply strictly according to recommended rates, as boron has a narrow range between deficiency and toxicity."
                    ],
                    "themeColor": "#E91E63",
                    "description": "Boron deficiency manifests as 'Hook Leaf' on emerging fronds where leaflet tips are bent and rigid. Spear leaves fail to open, inflorescences become necrotic, and the palm produces flat-sided, barren nuts due to poor cell division.",
                    "advice": "Apply 20g of Sodium Tetraborate (Borax) per mature or young palm at 6-month intervals until symptoms disappear. Seedlings should receive 10g Borax."
                }
            ]
            for def_data in default_deficiencies:
                deficiencies_ref.document(def_data["id"]).set(def_data)
            print("[INFO] Successfully seeded deficiencies.")
        else:
            print("[INFO] Deficiencies already seeded. Skipping.")
    except Exception as e:
         print(f"[ERROR] Seeding deficiencies failed: {e}")

@app.on_event("startup")
def on_startup():
    seed_deficiencies_if_empty()

@app.get("/")
def read_root():
    return {
        "project": "SaruPol - AI Coconut Nutrient & Lab Intelligence System",
        "status": "Online",
        "endpoints": {
            "image_prediction": "POST /api/v1/nutrient-analysis/predict",
            "lab_recommendation": "POST /api/v1/nutrient-analysis/lab-recommendation",
            "agro_zone": "POST /api/v1/location/agro-zone",
            "deficiencies": "GET /api/v1/nutrient-analysis/deficiencies"
        }
    }

@app.get("/health")
@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Coconut Nutrient & Lab Intelligence",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

# ---------------------------------------------------------
# Image Scan Part: Leaf Nutrient Deficiency Detection
# ---------------------------------------------------------

@app.post("/api/v1/nutrient-analysis/predict", response_model=ImagePredictionResponse)
@app.post("/api/nutrient/nutrient-analysis/predict", response_model=ImagePredictionResponse)
@app.post("/api/soil/nutrient-analysis/predict", response_model=ImagePredictionResponse)
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
        
    status_str = result["status"]
    
    # If validation failed (invalid_input or uncertain)
    if status_str != "success":
        return {
            "success": True,
            "status": status_str,
            "message": result["message"],
            "prediction": None,
            "recommendation": None,
            "visual_features": result.get("features", {})
        }
        
    predicted_class = result["prediction"]
    confidence = result["confidence"]
    visual_features = result["features"]
    
    # Get image-specific recommendation
    rec_result = get_image_based_recommendation(
        predicted_nutrient=predicted_class,
        confidence=confidence,
        threshold=0.60
    )
    
    # Merge the visual features into the response for transparency
    rec_result["visual_features"] = visual_features
    if "cnn_comparison" in result:
        rec_result["cnn_comparison"] = result["cnn_comparison"]
    
    return rec_result

# ---------------------------------------------------------
# Location & Agro-Climatic Zone
# ---------------------------------------------------------

@app.post("/api/v1/location/agro-zone", response_model=LocationResponse)
@app.post("/api/nutrient/location/agro-zone", response_model=LocationResponse)
@app.post("/api/soil/location/agro-zone", response_model=LocationResponse)
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
        with urllib.request.urlopen(req_obj, timeout=3.5) as response:
            data = json.loads(response.read().decode('utf-8'))
            
        if "error" in data:
            raise ValueError(f"NSDI API Error: {data['error'].get('message', 'Unknown Error')}")
            
        features = data.get("features", [])
        if not features:
            return LocationResponse(
                success=True,
                zone="Intermediate",
                agro_ecological_zone="IL1a",
                message="No features returned by NSDI. Used local Intermediate fallback.",
                raw_attributes=None
            )
            
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
                
        if not major_zone and agro_eco_zone:
            a_upper = str(agro_eco_zone).strip().upper()
            if a_upper.startswith("W"):
                major_zone = "Wet"
            elif a_upper.startswith("I"):
                major_zone = "Intermediate"
            elif a_upper.startswith("D"):
                major_zone = "Dry"
                
        if not major_zone:
            major_zone = "Intermediate"
            
        return LocationResponse(
            success=True,
            zone=major_zone,
            agro_ecological_zone=agro_eco_zone,
            message="Zone successfully detected via NSDI.",
            raw_attributes=attrs
        )
        
    except Exception as e:
        if 5.9 <= lat <= 9.9 and 79.6 <= lon <= 81.9:
            if lat > 7.8 or lon > 80.9:
                major_zone = "Dry"
                agro_eco_zone = "DL1"
            elif lat < 7.3 and lon < 80.5:
                major_zone = "Wet"
                agro_eco_zone = "WL1"
            else:
                major_zone = "Intermediate"
                agro_eco_zone = "IL1a"
        else:
            major_zone = "Intermediate"
            agro_eco_zone = "IL1a"
            
        return LocationResponse(
            success=True,
            zone=major_zone,
            agro_ecological_zone=agro_eco_zone,
            message=f"Zone estimated via local geographic heuristic (NSDI Service unavailable: {str(e)}).",
            raw_attributes={"error_fallback": str(e)}
        )

# ---------------------------------------------------------
# Nutrient Scan History & Deficiencies Library
# ---------------------------------------------------------

@app.get("/api/v1/nutrient-analysis/deficiencies")
@app.get("/api/nutrient/nutrient-analysis/deficiencies")
@app.get("/api/soil/nutrient-analysis/deficiencies")
def get_deficiencies():
    if db is None:
        raise HTTPException(status_code=500, detail="Firebase not configured")
    try:
        deficiencies_ref = db.collection("deficiencies")
        docs = deficiencies_ref.stream()
        results = []
        for doc in docs:
            results.append(doc.to_dict())
        results.sort(key=lambda x: x.get("id", ""))
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch deficiencies: {str(e)}")

@app.post("/api/v1/nutrient-analysis/scans")
@app.post("/api/nutrient/nutrient-analysis/scans")
@app.post("/api/soil/nutrient-analysis/scans")
def save_nutrient_scan(data: SaveNutrientScanRequest):
    if db is None:
        raise HTTPException(status_code=500, detail="Firebase not configured")
    try:
        import time
        scan_id = f"scan_{int(time.time() * 1000)}"
        doc_ref = db.collection("users").document(data.user_id).collection("nutrient_scans").document(scan_id)
        scan_doc = {
            "id": scan_id,
            "user_id": data.user_id,
            "palm_age": data.palm_age,
            "palm_stage": data.palm_stage,
            "zone": data.zone,
            "image_uri": data.image_uri,
            "prediction": data.prediction.dict() if data.prediction else None,
            "recommendation": data.recommendation.dict() if data.recommendation else None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        doc_ref.set(scan_doc)
        return {"success": True, "message": "Scan result saved successfully.", "id": scan_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save scan result: {str(e)}")

@app.get("/api/v1/nutrient-analysis/scans")
@app.get("/api/nutrient/nutrient-analysis/scans")
@app.get("/api/soil/nutrient-analysis/scans")
def get_nutrient_scans(user_id: str):
    if db is None:
        raise HTTPException(status_code=500, detail="Firebase not configured")
    try:
        scans_ref = db.collection("users").document(user_id).collection("nutrient_scans")
        docs = scans_ref.order_by("timestamp", direction="DESCENDING").stream()
        results = []
        for doc in docs:
            results.append(doc.to_dict())
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch nutrient scans: {str(e)}")

# ---------------------------------------------------------
# Lab Analysis Part: Differential Fertilizer Advisory
# ---------------------------------------------------------

@app.post("/api/v1/nutrient-analysis/lab-recommendation", response_model=LabRecommendationResponse)
@app.post("/api/nutrient/nutrient-analysis/lab-recommendation", response_model=LabRecommendationResponse)
@app.post("/api/soil/nutrient-analysis/lab-recommendation", response_model=LabRecommendationResponse)
def get_lab_recommendation(data: LabRecommendationRequest):
    try:
        n_val = data.nitrogen
        p_val = data.phosphorus
        k_val = data.potassium
        mg_val = data.magnesium
        age_val = data.palm_age
        zone = data.zone

        is_adult = age_val >= 4
        is_dry_zone = "dry" in zone.lower()

        # 1. Nitrogen (N) Evaluation & Urea Calculation
        evalN = 'Optimal'
        if n_val < 1.90:
            evalN = 'Deficient'
        elif n_val > 2.10:
            evalN = 'Excess'

        if 1.90 <= n_val <= 2.10:
            urea = 800
        elif 1.70 <= n_val < 1.90:
            urea = 900
        elif 1.60 <= n_val < 1.70:
            urea = 1000
        elif n_val < 1.60:
            urea = 1100
        else:
            urea = 500

        # 2. Potassium (K) Evaluation & MOP Calculation
        evalK = 'Optimal'
        if k_val < 1.20:
            evalK = 'Deficient'
        elif k_val > 1.50:
            evalK = 'Excess'

        if 1.20 <= k_val <= 1.50:
            mop = 1600
        elif 1.00 <= k_val < 1.20:
            mop = 1700
        elif 0.80 <= k_val < 1.00:
            mop = 1800
        elif 0.70 <= k_val < 0.80:
            mop = 1900
        elif 0.60 <= k_val < 0.70:
            mop = 2000
        else:
            if k_val > 1.50:
                mop = 1000
            else:
                mop = 2200

        # 3. Phosphorus (P) Evaluation & ERP/TSP Calculation
        evalP = 'Optimal'
        if p_val < 0.11:
            evalP = 'Deficient'
        elif p_val > 0.13:
            evalP = 'Excess'

        phosphate_type = 'TSP' if is_dry_zone else 'ERP'
        erp_or_tsp = 900
        p_special_advice = None

        if 0.11 <= p_val <= 0.13:
            erp_or_tsp = 400 if is_dry_zone else 900
        elif 0.09 <= p_val < 0.11:
            erp_or_tsp = 400 if is_dry_zone else 800
        elif 0.08 <= p_val < 0.09:
            erp_or_tsp = 600
            p_special_advice = "Phosphorus (P) is moderately deficient (0.08 - 0.09%). Apply 0.3 kg of Imported Rock Phosphate (IRP) + 0.3 kg of Triple Super Phosphate (TSP) to enhance absorption."
        elif p_val < 0.08:
            erp_or_tsp = 600
            phosphate_type = 'TSP'
            p_special_advice = "Phosphorus (P) is severely deficient (< 0.08%). Apply 0.6 kg of Triple Super Phosphate (TSP) directly to the root zone."
        else:
            erp_or_tsp = 200 if is_dry_zone else 400

        # 4. Magnesium (Mg) Evaluation & Dolomite/Kieserite Calculation
        evalMg = 'N/A'
        dolomite = 1000
        kieserite = 0
        mg_special_advice = None

        if mg_val is not None:
            evalMg = 'Optimal'
            if mg_val < 0.25:
                evalMg = 'Deficient'
            elif mg_val > 0.35:
                evalMg = 'Excess'

            if 0.25 <= mg_val <= 0.35:
                dolomite = 1000
                kieserite = 0
            elif 0.21 <= mg_val < 0.25:
                dolomite = 2000
                kieserite = 0
            elif 0.15 <= mg_val < 0.21:
                dolomite = 2000
                kieserite = 1000
                mg_special_advice = "Magnesium (Mg) is moderately deficient (0.15 - 0.20%). Apply 2.0 kg of Dolomite and 1.0 kg of Kieserite per palm."
            elif 0.10 <= mg_val < 0.15:
                dolomite = 2000
                kieserite = 1500
                mg_special_advice = "Magnesium (Mg) is significantly deficient (0.10 - 0.14%). Apply 2.0 kg of Dolomite and 1.5 kg of Kieserite per palm."
            elif mg_val < 0.10:
                dolomite = 2000
                kieserite = 2000
                mg_special_advice = "Magnesium (Mg) is severely deficient (< 0.10%). Apply 2.0 kg of Dolomite and 2.0 kg of Kieserite per palm as an emergency corrective measure."
            else:
                dolomite = 500
                kieserite = 0

        # 5. Age-Specific Baseline Lookup (Young Palms vs Adult Palms)
        if not is_adult:
            if age_val < 1.0:
                urea = 190
                mop = 190
                dolomite = 500
                erp_or_tsp = 160 if is_dry_zone else 420
            elif 1.0 <= age_val < 2.0:
                urea = 235
                mop = 235
                dolomite = 500
                erp_or_tsp = 200 if is_dry_zone else 530
            elif 2.0 <= age_val < 3.0:
                urea = 305
                mop = 305
                dolomite = 500
                erp_or_tsp = 300 if is_dry_zone else 690
            else:
                urea = 375
                mop = 375
                dolomite = 500
                erp_or_tsp = 360 if is_dry_zone else 850
            
            kieserite = 0
            mg_special_advice = None
            p_special_advice = None

        # 6. Agronomic Advice & Response Formulating
        is_healthy = (evalN == 'Optimal' and evalP == 'Optimal' and 
                      evalK == 'Optimal' and (evalMg == 'N/A' or evalMg == 'Optimal'))
        health_status = 'Healthy Palm' if is_healthy else 'Fertilizer Required'

        if is_adult:
            advice_list = [
                'Apply fertilizer in a circular trench 1.8m away from the base of the palm.',
                'Divide the annual dosage into two equal applications (Yala and Maha seasons).'
            ]

            if evalN == 'Deficient':
                advice_list.append(f"Apply {urea}g of Urea per year to correct Nitrogen deficiency.")
            if evalK == 'Deficient':
                advice_list.append(f"Apply {mop}g of Muriate of Potash (MOP) per year. Bury coconut husks in trenches between rows to recycle Potassium and conserve moisture.")
            
            if p_special_advice:
                advice_list.append(p_special_advice)
            elif evalP == 'Deficient':
                advice_list.append(f"Apply {erp_or_tsp}g of {phosphate_type} per year to correct Phosphorus deficiency.")
                
            if mg_special_advice:
                advice_list.append(mg_special_advice)
            elif evalMg == 'Deficient':
                advice_list.append(f"Apply {dolomite}g of Dolomite to buffer soil acidity and supply Magnesium.")
        else:
            advice_list = [
                f"For young palms (age {age_val} yrs), apply standard CRI straight fertilizer dosage every 6 months.",
                'Apply fertilizer in a circle starting 30cm to 90cm away from the base depending on growth.',
                f"Apply {urea}g of Urea, {erp_or_tsp}g of {phosphate_type}, {mop}g of MOP, and {dolomite}g of Dolomite per application.",
                "Leaf analysis-based adjustments are not required for young vegetative palms under 4 years."
            ]

        return LabRecommendationResponse(
            urea=urea,
            erp_or_tsp=erp_or_tsp,
            mop=mop,
            dolomite=dolomite,
            phosphate_type=phosphate_type,
            evalN=f"{evalN} (N)",
            evalP=f"{evalP} (P)",
            evalK=f"{evalK} (K)",
            evalMg=evalMg if evalMg == 'N/A' else f"{evalMg} (Mg)",
            health_status=health_status,
            agronomic_advice=advice_list
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate lab recommendation: {str(e)}")

# ---------------------------------------------------------
# Pathology Hotspots Fallback / Mock Endpoints
# ---------------------------------------------------------
@app.get("/api/pathology/aerial/hotspots")
def get_pathology_aerial_hotspots(estate_id: str = "estate_001", limit: int = 10, status: str = None):
    return {
        "estate_id": estate_id,
        "count": 2,
        "hotspots": [
            {
                "id": "hotspot_sector_b_01",
                "estate_id": estate_id,
                "location": {"lat": 7.2914, "lng": 80.6342},
                "severity": "critical",
                "mean_index_value": 0.28,
                "radius_meters": 6.4,
                "recommended_action": "Sector B: Immediate on-site mobile leaf scan required for Bud Rot.",
                "status": "pending"
            },
            {
                "id": "hotspot_sector_c_04",
                "estate_id": estate_id,
                "location": {"lat": 7.2928, "lng": 80.6325},
                "severity": "high",
                "mean_index_value": 0.42,
                "radius_meters": 4.8,
                "recommended_action": "Sector C: Crown chlorosis detected. Inspect leaf margins for Gray Leaf Spot.",
                "status": "pending"
            }
        ]
    }

@app.patch("/api/pathology/aerial/hotspots/{hotspot_id}")
def update_pathology_aerial_hotspot(hotspot_id: str, payload: dict = None):
    return {
        "id": hotspot_id,
        "status": payload.get("resolution_status", "investigating") if payload else "investigating",
        "notes": payload.get("notes", "") if payload else ""
    }
