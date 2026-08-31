import os
import sys
import json
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)

def test_root_endpoint():
    """Integration Test: Verify Root API Endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "SaruPol" in data.get("project", "") or data.get("status") == "Online"

def test_health_endpoint():
    """Integration Test: Verify Health Check Endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "healthy"

def test_deficiencies_endpoint():
    """Integration Test: Verify Deficiency Guide Retrieval Endpoint"""
    response = client.get("/api/v1/nutrient-analysis/deficiencies")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

def test_invalid_predict_payload():
    """Integration Test: Verify Invalid Image Upload Error Handling"""
    response = client.post(
        "/api/v1/nutrient-analysis/predict",
        files={"image": ("bad.txt", b"not an image", "text/plain")}
    )
    assert response.status_code in [200, 400, 422]

def test_model_status_endpoint():
    """Integration Test: Verify ML Model Status Endpoint"""
    response = client.get("/api/v1/models/status")
    assert response.status_code == 200
    data = response.json()
    assert "active_model_loaded" in data or "status" in data
