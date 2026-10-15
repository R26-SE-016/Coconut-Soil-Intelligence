import numpy as np
from typing import Tuple
import os
import requests
import cv2

from dotenv import load_dotenv

# Ensure .env is loaded regardless of current working directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(base_dir, ".env"))

# =====================================================================
# CONFIGURATION CONSTANTS
# Engineering safeguards, not biological nutrient thresholds.
# =====================================================================
MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100
MIN_FOREGROUND_PIXELS = 500
MIN_LEAF_AREA_RATIO = 0.05
MIN_COLOR_DENSITY = 0.01

SPOT_COUNT_THRESHOLD = 20
DISCOLORATION_THRESHOLD = 0.30
POST_VALIDATION_CONFIDENCE_THRESHOLD = 0.60

class ValidationResult:
    def __init__(self, is_valid: bool, status: str, message: str = ""):
        self.is_valid = is_valid
        self.status = status
        self.message = message

def validate_basic_leaf_input(image: np.ndarray, leaf_mask: np.ndarray, feats: dict) -> ValidationResult:
    """
    Basic Leaf Input Validation
    
    PURPOSE:
    To reject obviously invalid images and unsuitable inputs.
    
    LIMITATION:
    This rule-based pre-validation cannot reliably prove that an image is specifically a coconut leaf.
    It only establishes that the image contains a reasonably detectable leaf-like foreground.
    A dedicated Coconut-vs-Non-Coconut image classifier is recommended for production-level OOD detection.
    """
    
    # 1. Resolution Check
    h, w = image.shape[:2]
    if h < MIN_IMAGE_HEIGHT or w < MIN_IMAGE_WIDTH:
        return ValidationResult(False, "invalid_input", "Please upload a clear image of a coconut leaf. (Resolution too low)")
        
    # 2. Minimum Foreground Pixels Check
    import cv2
    foreground_pixels = cv2.countNonZero(leaf_mask)
    if foreground_pixels < MIN_FOREGROUND_PIXELS:
        return ValidationResult(False, "invalid_input", "Please upload a clear image of a coconut leaf. (No leaf detected)")
        
    # 3. Leaf Area Ratio Check
    leaf_area_ratio = feats.get('total_leaf_area', 0) / (h * w)
    if leaf_area_ratio < MIN_LEAF_AREA_RATIO:
        return ValidationResult(False, "invalid_input", "Please upload a clear image of a coconut leaf. (Leaf area too small)")
        
    # 4. Color Density Check
    # Ensures the extracted mask actually contains standard plant colors, mitigating noise false positives
    color_density = (
        feats.get('green_ratio', 0) + 
        feats.get('yellow_ratio', 0) + 
        feats.get('brown_ratio', 0) + 
        feats.get('rust_ratio', 0)
    )
    if color_density < MIN_COLOR_DENSITY:
        return ValidationResult(False, "invalid_input", "Please upload a clear image of a coconut leaf. (Unrelated colors)")
        
    # 5. Plant Vegetation / Out-of-Distribution (OOD) Safety Check
    # We analyze the RGB color relation within the detected leaf area.
    # Coconut leaves (even deficient ones) have a minimum green presence and plant color signature.
    # Skin tones, wood, and concrete typically have strongly negative ExG (< -10) and R significantly greater than G.
    mean_r = feats.get('mean_r', 0)
    mean_g = feats.get('mean_g', 0)
    mean_b = feats.get('mean_b', 0)
    green_ratio = feats.get('green_ratio', 0)
    
    # Excess Green Index (ExG)
    exg = 2 * mean_g - mean_r - mean_b
    
    # Catch indoor scenes, wood, human skin, food plates where green is absent (< 1.5%) and Red dominates Green
    if (exg < 0.0 and green_ratio < 0.02 and mean_r > 1.25 * mean_g) or exg < -15.0:
        return ValidationResult(
            False, 
            "invalid_input", 
            "The uploaded image does not appear to be a valid coconut leaf. Please capture a clear photo of a coconut leaf."
        )
        
    # 6. Optional PlantNet API Verification (if API key is configured in env)
    # This precisely identifies whether the leaf is a coconut leaf (Cocos nucifera) or another plant species.
    is_coconut, err_msg = verify_coconut_leaf_via_plantnet(image)
    if not is_coconut:
        return ValidationResult(False, "invalid_input", err_msg)
        
    return ValidationResult(True, "success")


def verify_coconut_leaf_via_plantnet(image: np.ndarray) -> Tuple[bool, str]:
    """
    Verifies whether the uploaded image is a coconut leaf using the PlantNet API.
    
    Strategy:
    - Scans ALL returned results (not just top-1) for Arecaceae (palm family) or Cocos genus.
    - If ANY result matches palm/coconut family → PASS (it's likely a coconut/palm leaf).
    - If PlantNet returns 404 (species not found) → REJECT (image is not a recognizable plant).
    - If PlantNet returns results but NONE are palms → REJECT with detected plant name.
    - On API error or missing key → gracefully PASS (let other validators handle it).
    """
    api_key = os.getenv("PLANTNET_API_KEY")
    if not api_key:
        return True, ""
        
    url = f"https://my-api.plantnet.org/v2/identify/all?api-key={api_key}"
    
    try:
        # Encode image to JPEG
        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = buffer.tobytes()
        
        files = {
            'images': ('image.jpg', image_bytes, 'image/jpeg')
        }
        data = {
            'organs': ['leaf']
        }
        
        response = requests.post(url, files=files, data=data, timeout=10.0)
        
        # 404 = PlantNet could not identify ANY plant species in the image
        if response.status_code == 404:
            print("[INFO] PlantNet: No plant species found in image (404)")
            return False, "The uploaded image does not appear to contain a recognizable plant leaf. Please upload a clear photo of a coconut leaf."
        
        if response.status_code == 200:
            res_json = response.json()
            results = res_json.get('results', [])
            
            if not results:
                # API returned 200 but empty results — unlikely but handle gracefully
                return True, ""
            
            # --- Scan ALL results for any palm/coconut match ---
            PALM_FAMILIES = {"Arecaceae"}
            COCONUT_GENERA = {"Cocos"}
            COCONUT_SPECIES = {"Cocos nucifera"}
            
            for result in results:
                species = result.get('species', {})
                scientific_name = species.get('scientificNameWithoutAuthor', '')
                
                genus_info = species.get('genus', {})
                genus_name = genus_info.get('scientificNameWithoutAuthor', '') if isinstance(genus_info, dict) else str(genus_info)
                
                family_info = species.get('family', {})
                family_name = family_info.get('scientificNameWithoutAuthor', '') if isinstance(family_info, dict) else str(family_info)
                
                # Check if this result is a coconut or palm
                if (scientific_name in COCONUT_SPECIES or 
                    genus_name in COCONUT_GENERA or 
                    family_name in PALM_FAMILIES):
                    print(f"[INFO] PlantNet: Coconut/palm match found — {scientific_name} (family: {family_name})")
                    return True, ""
            
            # No palm/coconut found in ANY result — this is likely a different plant
            top_species = results[0].get('species', {})
            top_common_names = top_species.get('commonNames', [])
            top_scientific = top_species.get('scientificNameWithoutAuthor', 'Unknown')
            plant_name = top_common_names[0] if top_common_names else top_scientific
            
            print(f"[INFO] PlantNet: Non-coconut plant detected — {plant_name} ({top_scientific})")
            return False, f"Detected plant: {plant_name}. This system only analyzes coconut leaves. Please upload a clear coconut leaf image."
        
        # Other HTTP errors (rate limit, server error, etc.) — pass gracefully
        print(f"[WARN] PlantNet API returned status {response.status_code}")
        return True, ""
        
    except requests.exceptions.Timeout:
        print("[WARN] PlantNet API request timed out")
        return True, ""
    except Exception as e:
        print(f"[WARN] PlantNet API verification failed: {e}")
        return True, ""


def apply_post_prediction_safety_rule(feats: dict, predicted_class: str, model_confidence: float) -> ValidationResult:
    """
    Post-Prediction Safety Rule (Pest/Disease Confounding Check)
    
    PURPOSE:
    A conservative uncertainty safeguard. It does NOT classify disease.
    It simply means that the visual evidence is not sufficiently reliable for a confident nutrient-only diagnosis.
    """
    spot_count = feats.get('spot_count', 0)
    discolored_ratio = feats.get('discolored_ratio', 0.0)
    
    if (spot_count > SPOT_COUNT_THRESHOLD or discolored_ratio > DISCOLORATION_THRESHOLD):
        # If the model predicts "Healthy" but there are significant spots/discoloration, 
        # it is a major contradiction (likely OOD image). 
        # Otherwise, check the general confidence threshold.
        if predicted_class == 'Healthy' or model_confidence < POST_VALIDATION_CONFIDENCE_THRESHOLD:
            return ValidationResult(
                False, 
                "uncertain", 
                "Unable to confidently determine whether the observed symptoms are nutrient-related. Laboratory or pest/disease analysis is recommended."
            )
            
    return ValidationResult(True, "success")
