import time
import requests
import json

BASE_URL = "http://127.0.0.1:8000/api/v1/analysis"
TREE_ID = "MK-101"

def run_simulation():
    print(f"--- Starting ESP32 IoT Simulation for Tree {TREE_ID} ---")
    
    # 1. Start Analysis Session
    print("\n[1] Starting new analysis session...")
    start_resp = requests.post(f"{BASE_URL}/start", json={
        "tree_no": TREE_ID,
        "zone_id": "Zone A"
    })
    
    if start_resp.status_code != 200:
        print(f"Error starting analysis: {start_resp.text}")
        return
        
    session_data = start_resp.json()
    analysis_id = session_data["analysis_id"]
    print(f" -> Session created: {analysis_id}")
    
    time.sleep(1)
    
    # 2. Capture Point 1
    print("\n[2] Capturing Point 1...")
    p1_resp = requests.post(f"{BASE_URL}/reading", json={
        "analysis_id": analysis_id,
        "tree_no": TREE_ID,
        "point_name": "point1",
        "reading": {
            "N": 0.015, "P": 0.34, "K": 0.06, 
            "pH": 6.4, "EC": 1.2, "moisture": 48.0, "temperature": 24.2
        }
    })
    print(f" -> Point 1 Status: {p1_resp.status_code} | {p1_resp.json()['message']}")
    
    time.sleep(1)
    
    # 3. Capture Point 2
    print("\n[3] Capturing Point 2...")
    p2_resp = requests.post(f"{BASE_URL}/reading", json={
        "analysis_id": analysis_id,
        "tree_no": TREE_ID,
        "point_name": "point2",
        "reading": {
            "N": 0.033, "P": 0.15, "K": 0.06, 
            "pH": 6.5, "EC": 1.3, "moisture": 50.0, "temperature": 24.6
        }
    })
    print(f" -> Point 2 Status: {p2_resp.status_code} | {p2_resp.json()['message']}")
    
    time.sleep(1)
    
    # 4. Capture Point 3
    print("\n[4] Capturing Point 3...")
    p3_resp = requests.post(f"{BASE_URL}/reading", json={
        "analysis_id": analysis_id,
        "tree_no": TREE_ID,
        "point_name": "point3",
        "reading": {
            "N": 0.021, "P": 0.11, "K": 0.11, 
            "pH": 6.3, "EC": 1.1, "moisture": 46.0, "temperature": 24.0
        }
    })
    print(f" -> Point 3 Status: {p3_resp.status_code} | {p3_resp.json()['message']}")
    
    time.sleep(1)
    
    # 5. Complete Analysis & Trigger ML Prediction
    print("\n[5] 3 Points Captured. Completing Analysis & Running ML Prediction...")
    complete_resp = requests.post(f"{BASE_URL}/complete", json={
        "analysis_id": analysis_id,
        "tree_no": TREE_ID
    })
    
    if complete_resp.status_code == 200:
        result = complete_resp.json()
        print("\n=== FINAL RESULT ===")
        print(f"Status: {result['status']}")
        print(f"\nAverages Calculated:")
        print(json.dumps(result['average'], indent=2))
        
        print(f"\nML Predicted 14th Leaf NPK (Model: {result['prediction']['modelVersion']}):")
        print(f"  Leaf N: {result['prediction']['leafN']}%")
        print(f"  Leaf P: {result['prediction']['leafP']}%")
        print(f"  Leaf K: {result['prediction']['leafK']}%")
        
        print(f"\nCRI Official Fertilizer Recommendation:")
        print(f"  Health:   {result['recommendation']['healthStatus']}")
        print(f"  Urea:     {result['recommendation']['urea']} kg")
        print(f"  ERP:      {result['recommendation']['ERP']} kg")
        print(f"  MOP:      {result['recommendation']['MOP']} kg")
        print(f"  Dolomite: {result['recommendation']['dolomite']} kg")
        
        print("\nAdvice:")
        for advice in result['recommendation']['agronomicAdvice']:
            print(f" - {advice}")
            
        print("\n✅ Simulation completed successfully! Check your Firebase Console to see the document.")
    else:
        print(f"❌ Error completing analysis: {complete_resp.status_code} - {complete_resp.text}")

if __name__ == "__main__":
    try:
        run_simulation()
    except requests.exceptions.ConnectionError:
        print("\n❌ Connection Error: Ensure your FastAPI server is running (uvicorn app.main:app --reload)")
