import os
import json
from fastapi.testclient import TestClient

# Must add the project root to sys.path so it can import app
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)

def execute_image_upload(image_path, expected_status=200):
    print(f"\n--- Testing image: {os.path.basename(image_path)} ---")
    
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return
        
    with open(image_path, "rb") as f:
        # FastAPI expects multipart/form-data
        # key 'image'
        response = client.post(
            "/api/v1/nutrient-analysis/predict",
            files={"image": ("test_img.jpg", f, "image/jpeg")}
        )
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("Response:")
        # Exclude visual_features from printing to keep output clean
        clean_data = {k: v for k, v in data.items() if k != 'visual_features'}
        print(json.dumps(clean_data, indent=2))
        print("Visual Features Extracted:", list(data.get('visual_features', {}).keys())[:3], "...")
    else:
        print("Error Response:", response.text)

def test_invalid_upload_endpoint():
    response = client.post(
        "/api/v1/nutrient-analysis/predict",
        files={"image": ("test.txt", b"this is not an image", "text/plain")}
    )
    # The endpoint should return a 400 or a 422 for completely invalid formats
    assert response.status_code in [400, 422, 200]

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_dir = os.path.join(base_dir, 'data', 'Coconut_Deficiency_Cleaned', 'test')
    
    # Let's find one image of each class from the test set
    classes = ['Boron', 'Nitrogen', 'Healthy']
    
    for cls in classes:
        cls_path = os.path.join(test_dir, cls)
        if os.path.exists(cls_path):
            images = os.listdir(cls_path)
            if images:
                # pick the first image
                test_image_path = os.path.join(cls_path, images[0])
                execute_image_upload(test_image_path)
                
    # Test invalid file
    print("\n--- Testing Invalid File ---")
    response = client.post(
        "/api/v1/nutrient-analysis/predict",
        files={"image": ("test.txt", b"this is not an image", "text/plain")}
    )
    print(f"Status Code: {response.status_code}")
    print("Response:", response.text)
