import cv2
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.leaf_analyzer import (
    create_leaf_mask,
    extract_color_features,
    extract_spot_features,
    extract_spatial_features
)
from ml.image_validator import validate_basic_leaf_input

img_path = r"C:\Users\MSI\.gemini\antigravity\brain\1d9c4c1e-d9fe-4e32-981f-1af82c3e6941\media__1788016046652.jpg"
image = cv2.imread(img_path)

if image is None:
    print("Failed to load image!")
else:
    print("Image loaded successfully. Shape:", image.shape)
    
    # Run analyzer steps
    leaf_mask = create_leaf_mask(image)
    c_feats, disc_mask = extract_color_features(image, leaf_mask)
    
    if c_feats is None:
        c_feats = {'total_leaf_area': 0}
        disc_mask = np.zeros_like(leaf_mask)
        
    s_feats = extract_spot_features(disc_mask, c_feats['total_leaf_area'])
    sp_feats = extract_spatial_features(leaf_mask, disc_mask)
    
    all_feats = {}
    all_feats.update(c_feats)
    all_feats.update(s_feats)
    all_feats.update(sp_feats)
    
    print("\n--- Features ---")
    for k, v in all_feats.items():
        if k != 'contour_coords':
            print(f"{k}: {v}")
            
    # Calculate ExG
    mean_r = all_feats.get('mean_r', 0)
    mean_g = all_feats.get('mean_g', 0)
    mean_b = all_feats.get('mean_b', 0)
    exg = 2 * mean_g - mean_r - mean_b
    print("\nExG:", exg)
    
    # Run validation
    res = validate_basic_leaf_input(image, leaf_mask, all_feats)
    print("\nValidation Result:")
    print("is_valid:", res.is_valid)
    print("status:", res.status)
    print("message:", res.message)
