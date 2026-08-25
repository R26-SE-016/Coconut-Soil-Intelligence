import os
import sys
import cv2
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.nutrient_predictor import predict_image

def run_test(name: str, image: np.ndarray):
    print(f"\n--- Test: {name} ---")
    try:
        # Encode as bytes to simulate upload
        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = buffer.tobytes()
        
        result = predict_image(image_bytes)
        status = result['status']
        print(f"Status: {status}")
        
        if status == 'success':
            print(f"Prediction: {result['prediction']} ({result['confidence']*100:.1f}%)")
        else:
            print(f"Reason: {result['message']}")
            
    except Exception as e:
        print(f"Error: {e}")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 1. Non-coconut broadleaf image (Artifact)
    broadleaf_path = r'C:\Users\upeksha\.gemini\antigravity-ide\brain\63461ccb-079d-400a-8eec-5b0b6657f20a\.user_uploaded\media_1787157606233.png'
    if os.path.exists(broadleaf_path):
        run_test("1. Non-coconut broadleaf", cv2.imread(broadleaf_path))
    else:
        print(f"File not found: {broadleaf_path}")
        
    # 2. Clear Healthy coconut leaf (from dataset)
    healthy_path = os.path.join(base_dir, 'data', 'Coconut_Deficiency_Cleaned', 'test', 'Healthy', 'Healthy_leaf_110_jpg.rf.4d6784286bdb74ae57324f26a7acf1bb.jpg')
    if os.path.exists(healthy_path):
        run_test("2. Clear Healthy coconut leaf", cv2.imread(healthy_path))
        
    # 3. Nitrogen dataset image
    nitrogen_path = os.path.join(base_dir, 'data', 'Coconut_Deficiency_Cleaned', 'test', 'Nitrogen', 'f_aug_0_1023_jpeg.rf.2dbd551b756f5371eacc68050087a0ec.jpg')
    if os.path.exists(nitrogen_path):
        run_test("3. Nitrogen dataset image", cv2.imread(nitrogen_path))
        
    # 4. Boron dataset image
    boron_path = os.path.join(base_dir, 'data', 'Coconut_Deficiency_Cleaned', 'test', 'Boron', 'DSC_0156_JPG.rf.57935e60400d8237f7b944f0f20b31ce.jpg')
    if os.path.exists(boron_path):
        run_test("4. Boron dataset image", cv2.imread(boron_path))
        
    # 5. Blurry image (Blur the healthy image)
    if os.path.exists(healthy_path):
        healthy_img = cv2.imread(healthy_path)
        blurry = cv2.GaussianBlur(healthy_img, (99, 99), 0)
        run_test("5. Blurry image", blurry)
        
    # 6. Non-leaf image (Random noise)
    noise = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    run_test("6. Non-leaf image (Noise)", noise)
    
    # 7. Very small image
    small = np.zeros((50, 50, 3), dtype=np.uint8)
    run_test("7. Very small image", small)
    
    # 8. High-spot/discoloration image (Simulated by drawing many brown/red spots on a green background)
    high_spot = np.zeros((400, 400, 3), dtype=np.uint8)
    high_spot[:] = (40, 150, 40)  # Green background
    for _ in range(50):
        x = np.random.randint(0, 400)
        y = np.random.randint(0, 400)
        r = np.random.randint(5, 20)
        cv2.circle(high_spot, (x, y), r, (20, 40, 150), -1) # Brown/rust spots
    run_test("8. High-spot/discoloration image", high_spot)

if __name__ == "__main__":
    main()
