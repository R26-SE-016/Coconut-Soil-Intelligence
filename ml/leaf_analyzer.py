import os
import cv2
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# where the data hv and where the  data save define
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'Coconut_Deficiency_Cleaned')
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'leaf_visual_features.csv')
SUMMARY_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'visual_feature_summary.csv')
REPORT_JSON = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'feature_extraction_report.json')
DEBUG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'debug_vis')
PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'feature_distributions')

# These are strictly visual feature thresholds, not diagnostic rules.
COLOR_BOUNDS = {
    'green':  [(35, 40, 40), (85, 255, 255)],
    'yellow': [(15, 50, 50), (34, 255, 255)],
    'brown':  [(0, 40, 20),  (14, 255, 150)], # Darker, low H
    'rust':   [(0, 50, 151), (14, 255, 255)], # Brighter/Reddish
}
# Wrap around for red/brown hues
COLOR_BOUNDS_WRAP = {
    'brown_wrap': [(170, 40, 20), (179, 255, 150)],
    'rust_wrap':  [(170, 50, 151), (179, 255, 255)]
}

for d in [DEBUG_DIR, PLOT_DIR, os.path.dirname(OUTPUT_CSV)]:
    os.makedirs(d, exist_ok=True)

# =====================================================================
# 1. IMAGE PREPROCESSING & SEGMENTATION
# =====================================================================
def create_leaf_mask(image):
    """
    Creates a binary mask isolating the leaf from the background.
    Uses broad HSV ranges to capture all potential leaf colors (green, yellow, brown)
    while excluding typical backgrounds.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Broad mask for leaf-like colors
    lower_leaf = np.array([0, 30, 20])
    upper_leaf = np.array([100, 255, 255])
    mask1 = cv2.inRange(hsv, lower_leaf, upper_leaf)
    
    # Add high-hue reds (170-179)
    lower_red = np.array([170, 30, 20])
    upper_red = np.array([179, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Morphological clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Keep only the largest contour to remove background noise
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(mask)
        cv2.drawContours(clean_mask, [largest_contour], -1, 255, -1)
        return clean_mask
    return mask

# =====================================================================
# 2 & 3. COLOR SPACE ANALYSIS & FEATURES
# =====================================================================
def get_color_mask(hsv_img, bounds, wrap_bounds=None):
    mask = cv2.inRange(hsv_img, np.array(bounds[0]), np.array(bounds[1]))
    if wrap_bounds:
        mask_wrap = cv2.inRange(hsv_img, np.array(wrap_bounds[0]), np.array(wrap_bounds[1]))
        mask = cv2.bitwise_or(mask, mask_wrap)
    return mask

def extract_color_features(img, leaf_mask):
    total_leaf_pixels = cv2.countNonZero(leaf_mask)
    if total_leaf_pixels == 0:
        return None, None
        
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Means
    mean_bgr = cv2.mean(img, mask=leaf_mask)[:3]
    mean_hsv = cv2.mean(hsv, mask=leaf_mask)[:3]
    mean_lab = cv2.mean(lab, mask=leaf_mask)[:3]
    
    # Color Masks
    green_mask = get_color_mask(hsv, COLOR_BOUNDS['green'])
    yellow_mask = get_color_mask(hsv, COLOR_BOUNDS['yellow'])
    brown_mask = get_color_mask(hsv, COLOR_BOUNDS['brown'], COLOR_BOUNDS_WRAP['brown_wrap'])
    rust_mask = get_color_mask(hsv, COLOR_BOUNDS['rust'], COLOR_BOUNDS_WRAP['rust_wrap'])
    
    # Intersection with leaf mask
    green_mask = cv2.bitwise_and(green_mask, leaf_mask)
    yellow_mask = cv2.bitwise_and(yellow_mask, leaf_mask)
    brown_mask = cv2.bitwise_and(brown_mask, leaf_mask)
    rust_mask = cv2.bitwise_and(rust_mask, leaf_mask)
    
    discolored_mask = cv2.bitwise_or(yellow_mask, cv2.bitwise_or(brown_mask, rust_mask))
    
    feats = {
        'green_ratio': cv2.countNonZero(green_mask) / total_leaf_pixels,
        'yellow_ratio': cv2.countNonZero(yellow_mask) / total_leaf_pixels,
        'brown_ratio': cv2.countNonZero(brown_mask) / total_leaf_pixels,
        'rust_ratio': cv2.countNonZero(rust_mask) / total_leaf_pixels,
        'discolored_ratio': cv2.countNonZero(discolored_mask) / total_leaf_pixels,
        'mean_r': mean_bgr[2], 'mean_g': mean_bgr[1], 'mean_b': mean_bgr[0],
        'mean_h': mean_hsv[0], 'mean_s': mean_hsv[1], 'mean_v': mean_hsv[2],
        'mean_l': mean_lab[0], 'mean_a': mean_lab[1], 'mean_b': mean_lab[2],
        'total_leaf_area': total_leaf_pixels
    }
    return feats, discolored_mask

# =====================================================================
# 4 & 5. AFFECTED AREA & SPOT FEATURES
# =====================================================================
def extract_spot_features(discolored_mask, total_leaf_area):
    contours, _ = cv2.findContours(discolored_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    spot_count = 0
    total_spot_area = 0
    largest_spot_area = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 10: # Filter noise
            spot_count += 1
            total_spot_area += area
            if area > largest_spot_area:
                largest_spot_area = area
                
    avg_spot_area = total_spot_area / spot_count if spot_count > 0 else 0
    
    return {
        'spot_count': spot_count,
        'average_spot_area': avg_spot_area,
        'largest_spot_area': largest_spot_area,
        'total_spot_area': total_spot_area,
        'affected_area_ratio': total_spot_area / total_leaf_area if total_leaf_area > 0 else 0
    }

# =====================================================================
# 6. SPATIAL FEATURES
# =====================================================================
def extract_spatial_features(leaf_mask, discolored_mask):
    contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {'edge_affected_ratio':0, 'center_affected_ratio':0, 'upper_affected_ratio':0, 'lower_affected_ratio':0}
    
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Regions
    mid_y = y + h // 2
    upper_leaf = leaf_mask[y:mid_y, x:x+w]
    lower_leaf = leaf_mask[mid_y:y+h, x:x+w]
    upper_disc = discolored_mask[y:mid_y, x:x+w]
    lower_disc = discolored_mask[mid_y:y+h, x:x+w]
    
    # Edge/Center
    cx, cy, cw, ch = x + w//4, y + h//4, w//2, h//2
    center_leaf_mask = np.zeros_like(leaf_mask)
    center_leaf_mask[cy:cy+ch, cx:cx+cw] = 255
    edge_leaf_mask = cv2.bitwise_not(center_leaf_mask)
    
    center_leaf = cv2.bitwise_and(leaf_mask, center_leaf_mask)
    edge_leaf = cv2.bitwise_and(leaf_mask, edge_leaf_mask)
    center_disc = cv2.bitwise_and(discolored_mask, center_leaf_mask)
    edge_disc = cv2.bitwise_and(discolored_mask, edge_leaf_mask)
    
    def safe_ratio(num, denom): return cv2.countNonZero(num) / cv2.countNonZero(denom) if cv2.countNonZero(denom) > 0 else 0.0

    return {
        'upper_affected_ratio': safe_ratio(upper_disc, upper_leaf),
        'lower_affected_ratio': safe_ratio(lower_disc, lower_leaf),
        'center_affected_ratio': safe_ratio(center_disc, center_leaf),
        'edge_affected_ratio': safe_ratio(edge_disc, edge_leaf)
    }

# =====================================================================
# MAIN PIPELINE
# =====================================================================
def main():
    print("Starting Visual Feature Extraction...")
    all_features = []
    report = {
        'processed': 0, 'failed': 0, 'segmentation_failures': 0,
        'class_distribution': {}, 'warnings': []
    }
    
    debug_images_saved = {'Nitrogen': 0, 'Boron': 0, 'Healthy': 0}
    
    splits = ['train', 'valid', 'test']
    for split in splits:
        split_dir = os.path.join(DATA_DIR, split)
        if not os.path.exists(split_dir): continue
            
        for cls in os.listdir(split_dir):
            if cls not in report['class_distribution']:
                report['class_distribution'][cls] = 0
                
            cls_dir = os.path.join(split_dir, cls)
            for img_name in tqdm(os.listdir(cls_dir), desc=f"Processing {split}/{cls}"):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue
                img_path = os.path.join(cls_dir, img_name)
                
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        report['failed'] += 1
                        continue
                        
                    leaf_mask = create_leaf_mask(img)
                    if cv2.countNonZero(leaf_mask) < 500:
                        report['segmentation_failures'] += 1
                        report['warnings'].append(f"Seg failed on {img_path}")
                        continue
                        
                    c_feats, disc_mask = extract_color_features(img, leaf_mask)
                    if c_feats is None: continue
                    
                    s_feats = extract_spot_features(disc_mask, c_feats['total_leaf_area'])
                    sp_feats = extract_spatial_features(leaf_mask, disc_mask)
                    
                    row = {'image_path': img_path, 'class': cls, 'split': split}
                    row.update(c_feats)
                    row.update(s_feats)
                    row.update(sp_feats)
                    all_features.append(row)
                    
                    report['processed'] += 1
                    report['class_distribution'][cls] += 1
                    
                    # Save 3 debug images per class
                    if debug_images_saved.get(cls, 0) < 3:
                        save_debug_vis(img, leaf_mask, disc_mask, row, cls, debug_images_saved[cls])
                        debug_images_saved[cls] += 1
                        
                except Exception as e:
                    report['failed'] += 1
                    report['warnings'].append(f"Error on {img_path}: {e}")
                    
    # Save Output
    df = pd.DataFrame(all_features)
    df.to_csv(OUTPUT_CSV, index=False)
    
    if not df.empty:
        generate_stats_and_plots(df)
        
    with open(REPORT_JSON, 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"\\nExtraction complete. Processed: {report['processed']}, Failed: {report['failed']}")

def save_debug_vis(img, leaf_mask, disc_mask, feats, cls, idx):
    leaf_only = cv2.bitwise_and(img, img, mask=leaf_mask)
    affected_vis = img.copy()
    affected_vis[disc_mask > 0] = [0, 0, 255]
    mask_3c = cv2.cvtColor(leaf_mask, cv2.COLOR_GRAY2BGR)
    
    top = np.hstack([cv2.resize(img, (224,224)), cv2.resize(mask_3c, (224,224))])
    bottom = np.hstack([cv2.resize(leaf_only, (224,224)), cv2.resize(affected_vis, (224,224))])
    grid = np.vstack([top, bottom])
    
    text = f"Y: {feats['yellow_ratio']:.2f} B: {feats['brown_ratio']:.2f} Spots: {feats['spot_count']}"
    cv2.putText(grid, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    out_path = os.path.join(DEBUG_DIR, f"{cls}_debug_{idx}.jpg")
    cv2.imwrite(out_path, grid)

def generate_stats_and_plots(df):
    numeric_cols = [
        'green_ratio', 'yellow_ratio', 'brown_ratio', 'rust_ratio', 
        'affected_area_ratio', 'spot_count', 'average_spot_area',
        'edge_affected_ratio', 'center_affected_ratio'
    ]
    
    summary = df.groupby('class')[numeric_cols].agg(['mean', 'std']).reset_index()
    summary.columns = ['_'.join(col).strip() if col[1] else col[0] for col in summary.columns.values]
    summary.to_csv(SUMMARY_CSV, index=False)
    
    for col in numeric_cols:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='class', y=col, data=df)
        plt.title(f'Distribution of {col}')
        plt.savefig(os.path.join(PLOT_DIR, f'boxplot_{col}.png'))
        plt.close()

if __name__ == "__main__":
    main()
