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

class SinglePointSoilInput(BaseModel):
    tree_no: int = Field(..., description="Tree ID or sample number", example=30)
    zone_id: Optional[str] = Field("Zone A", description="Plantation zone name", example="Zone A (Hilltop)")
    reading: SoilReading

class TriangulatedSoilInput(BaseModel):
    tree_no: int = Field(..., description="Tree ID or sample number", example=30)
    zone_id: Optional[str] = Field("Zone A", description="Plantation zone name", example="Zone A (Hilltop)")
    point_a: SoilReading
    point_b: SoilReading
    point_c: SoilReading

class PredictionResponse(BaseModel):
    tree_no: int
    zone_id: str
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
    zone_id: Optional[str] = Field("Zone A", description="Plantation zone name", example="Zone A (Hilltop)")

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
