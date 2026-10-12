from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class ImagePredictionDetails(BaseModel):
    nutrient: str
    class_name: str = Field(..., alias="class")
    confidence: float

class ImageRecommendationAdvice(BaseModel):
    source: str
    assessment_type: str
    advice: str

class CNNComparisonDetails(BaseModel):
    model_name: str = "Custom CNN (Baseline)"
    prediction: str
    confidence: float
    accuracy: float = 64.2

class ImagePredictionResponse(BaseModel):
    success: bool
    status: str
    message: Optional[str] = None
    prediction: Optional[ImagePredictionDetails] = None
    recommendation: Optional[ImageRecommendationAdvice] = None
    visual_features: Optional[Dict[str, float]] = None
    cnn_comparison: Optional[CNNComparisonDetails] = None

class LocationRequest(BaseModel):
    latitude: float = Field(..., description="Latitude of the location", example=7.29)
    longitude: float = Field(..., description="Longitude of the location", example=80.63)

class LocationResponse(BaseModel):
    success: bool
    zone: Optional[str] = Field(None, description="Normalized major zone (Wet, Intermediate, Dry)", example="Intermediate")
    agro_ecological_zone: Optional[str] = Field(None, description="Detailed agro-ecological zone from GIS", example="WM3b")
    message: Optional[str] = None
    raw_attributes: Optional[Dict[str, Any]] = Field(None, description="Raw attributes returned from NSDI GIS API")

class SaveNutrientScanRequest(BaseModel):
    user_id: str
    palm_age: str
    palm_stage: str
    zone: str
    image_uri: Optional[str] = None
    prediction: Optional[ImagePredictionDetails] = None
    recommendation: Optional[ImageRecommendationAdvice] = None

class LabRecommendationRequest(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    magnesium: Optional[float] = None
    palm_age: float
    zone: str

class LabRecommendationResponse(BaseModel):
    urea: int
    erp_or_tsp: int
    mop: int
    dolomite: int
    phosphate_type: str
    evalN: str
    evalP: str
    evalK: str
    evalMg: str
    health_status: str
    agronomic_advice: List[str]
