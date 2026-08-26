from typing import Dict, Any, Optional

def get_image_based_recommendation(predicted_nutrient: str, confidence: float, threshold: float = 0.60) -> Dict[str, Any]:
    """
    Provides a preliminary recommendation based ONLY on visual leaf image classification.
    Does NOT simulate numeric laboratory values.
    """
    
    if confidence < threshold:
        return {
            "success": True,
            "status": "uncertain",
            "prediction": None,
            "message": "The leaf image could not be classified confidently. Laboratory analysis is recommended for confirmation."
        }
        
    if predicted_nutrient == "Healthy":
        return {
            "success": True,
            "status": "success",
            "prediction": {
                "nutrient": "Healthy",
                "class": "Healthy",
                "confidence": round(confidence, 4)
            },
            "recommendation": None
        }
        
    if predicted_nutrient == "Nitrogen":
        return {
            "success": True,
            "status": "success",
            "prediction": {
                "nutrient": "Nitrogen",
                "class": "Nitrogen",
                "confidence": round(confidence, 4)
            },
            "recommendation": {
                "source": "leaf_image",
                "assessment_type": "preliminary_visual_assessment",
                "advice": "Possible Nitrogen deficiency detected. Image-based nutrient assessment is preliminary. Laboratory/leaf analysis is recommended for confirmation before applying a fertilizer treatment."
            }
        }
        
    if predicted_nutrient == "Boron":
        return {
            "success": True,
            "status": "success",
            "prediction": {
                "nutrient": "Boron",
                "class": "Boron",
                "confidence": round(confidence, 4)
            },
            "recommendation": {
                "source": "leaf_image",
                "assessment_type": "preliminary_visual_assessment",
                "advice": "Possible Boron deficiency detected. Image-based nutrient assessment is preliminary. Laboratory/leaf analysis is recommended for confirmation before applying a fertilizer treatment."
            }
        }
        
    if predicted_nutrient == "Magnesium":
        return {
            "success": True,
            "status": "success",
            "prediction": {
                "nutrient": "Magnesium",
                "class": "Magnesium",
                "confidence": round(confidence, 4)
            },
            "recommendation": {
                "source": "leaf_image",
                "assessment_type": "preliminary_visual_assessment",
                "advice": "Possible Magnesium deficiency detected. Common symptoms include yellowing at the leaf margins. Laboratory/leaf analysis is recommended for confirmation before applying Kieserite or Dolomite."
            }
        }
        
    # Fallback
    return {
        "success": False,
        "status": "error",
        "message": f"Unknown predicted class: {predicted_nutrient}"
    }
