from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class SoilReading(BaseModel):
    N: float = Field(..., description="Soil Nitrogen (N) percentage or mg/kg", example=0.0159)
    P: float = Field(..., description="Soil Phosphorus (P) percentage or mg/kg", example=0.3430)
    K: float = Field(..., description="Soil Potassium (K) percentage or mg/kg", example=0.0629)
    pH: Optional[float] = Field(6.5, description="Soil pH value (0 - 14)", example=6.5)
    moisture: Optional[float] = Field(45.0, description="Soil moisture percentage", example=45.0)
    temperature: Optional[float] = Field(28.5, description="Soil temperature in Celsius", example=28.5)
    EC: Optional[float] = Field(1.2, description="Electrical Conductivity in mS/cm", example=1.2)

class TriangulatedSoilInput(BaseModel):
    tree_no: int = Field(..., description="Tree ID or sample number", example=30)
    point_a: SoilReading
    point_b: SoilReading
    point_c: SoilReading

class PredictionResponse(BaseModel):
    tree_no: int
    sampling_method: str
    average_soil_npk: Dict[str, float]
    predicted_14th_leaf_npk: Dict[str, float]
    health_status: str
    fertilizer_recommendation_grams_per_year: Dict[str, int]
    nutrient_evaluation: Dict[str, str]
    agronomic_advice: List[str]
    model_used: str

class AnalysisStartRequest(BaseModel):
    tree_no: str = Field(..., description="Tree ID as string", example="MK-101")

class AnalysisStartResponse(BaseModel):
    analysis_id: str
    tree_no: str
    status: str
    message: str

class PointReadingInput(BaseModel):
    analysis_id: str = Field(..., description="The unique session ID for the analysis", example="AN-MK101-20260805-001")
    tree_no: str = Field(..., description="Tree ID", example="MK-101")
    point_name: str = Field(..., description="Must be 'point1', 'point2', or 'point3'", example="point1")
    reading: SoilReading

class AnalysisCompleteRequest(BaseModel):
    analysis_id: str = Field(..., description="The unique session ID for the analysis", example="AN-MK101-20260805-001")
    tree_no: str = Field(..., description="Tree ID", example="MK-101")

class ImagePredictionDetails(BaseModel):
    nutrient: str
    class_name: str = Field(..., alias="class")
    confidence: float

class ImageRecommendationAdvice(BaseModel):
    source: str
    assessment_type: str
    advice: str

class ImagePredictionResponse(BaseModel):
    success: bool
    status: str
    message: Optional[str] = None
    prediction: Optional[ImagePredictionDetails] = None
    recommendation: Optional[ImageRecommendationAdvice] = None
    visual_features: Optional[Dict[str, float]] = None

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
    agronomic_advice: list[str]


