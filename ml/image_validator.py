import numpy as np
from typing import Tuple

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
        
    return ValidationResult(True, "success")


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
