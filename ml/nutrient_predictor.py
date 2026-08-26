import os
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, Any, Tuple, Optional

from ml.image_validator import validate_basic_leaf_input, apply_post_prediction_safety_rule
from ml.leaf_analyzer import (
    create_leaf_mask,
    extract_color_features,
    extract_spot_features,
    extract_spatial_features
)

class HybridNutrientPredictor:
    def __init__(self):
        self.model = None
        self._load_model()
        
    def _load_model(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # We now use the newly trained YOLO model (best.pt)
        model_path = os.path.join(base_dir, 'ml', 'models', 'best.pt')
        
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
            print("[OK] Loaded YOLOv8 Deep Learning Model (Hybrid Approach)")
        else:
            print("[WARN] YOLOv8 Model not found at", model_path)
            
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        # 1. Image preprocessing and manual feature extraction (OpenCV - For Explainability)
        leaf_mask = create_leaf_mask(image)
        c_feats, disc_mask = extract_color_features(image, leaf_mask)
        
        if c_feats is None:
            # Mask generation entirely failed
            c_feats = {'total_leaf_area': 0}
            disc_mask = np.zeros_like(leaf_mask)
            
        s_feats = extract_spot_features(disc_mask, c_feats['total_leaf_area'])
        sp_feats = extract_spatial_features(leaf_mask, disc_mask)
        
        # Combine all manual features
        all_feats = {}
        all_feats.update(c_feats)
        all_feats.update(s_feats)
        all_feats.update(sp_feats)
            
        # 2. Pre-Prediction Validation (Checking if it's actually a leaf)
        pre_val = validate_basic_leaf_input(image, leaf_mask, all_feats)
        if not pre_val.is_valid:
            return {
                "status": pre_val.status,
                "prediction": None,
                "confidence": 0.0,
                "features": all_feats,
                "message": pre_val.message
            }
            
        # 3. Inference using YOLO (Deep Learning for high accuracy)
        if self.model is None:
            return {
                "status": "error",
                "prediction": None,
                "confidence": 0.0,
                "features": all_feats,
                "message": "YOLO AI Model is not loaded. Check model path."
            }
            
        try:
            results = self.model(image, verbose=False)
            probs = results[0].probs
            top_class_index = probs.top1
            predicted_class = results[0].names[top_class_index]
            confidence = float(probs.top1conf.item())
        except Exception as e:
            return {
                "status": "error",
                "prediction": None,
                "confidence": 0.0,
                "features": all_feats,
                "message": f"YOLO prediction failed: {str(e)}"
            }
        
        # 4. Post-Prediction Validation (Safety Rule)
        post_val = apply_post_prediction_safety_rule(all_feats, predicted_class, confidence)
        if not post_val.is_valid:
            return {
                "status": post_val.status,
                "prediction": predicted_class,
                "confidence": confidence,
                "features": all_feats,
                "message": post_val.message
            }
        
        return {
            "status": "success",
            "prediction": predicted_class,
            "confidence": confidence,
            "features": all_feats,
            "message": None
        }

# Singleton instance
predictor_instance = HybridNutrientPredictor()

def predict_image(image_bytes: bytes) -> Dict[str, Any]:
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Invalid image format.")
        
    return predictor_instance.predict(image)
