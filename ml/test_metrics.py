import os
import sys

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.metrics import (
    precision_at_k,
    recall_at_k,
    mean_precision_at_k,
    mean_recall_at_k
)

def run_metric_demos():
    print("="*60)
    # Scenario: A diagnostic system recommending nutrient deficiency actions
    print("DEMONSTRATING PRECISION@K & RECALL@K FOR NUTRIENT DIAGNOSTIC MODELS")
    print("="*60)
    
    # 1. Single sample demonstration
    print("\n--- 1. Single Palm Assessment Demo ---")
    
    # The actual deficiencies present in the palm
    actual = ["Nitrogen", "Potassium"]
    # Model's ranked predicted list (sorted by highest probability/confidence first)
    predicted = ["Nitrogen", "Magnesium", "Potassium", "Boron"]
    
    print(f"Actual Deficiencies: {actual}")
    print(f"Model Predictions (Ranked): {predicted}\n")
    
    for k in [1, 2, 3]:
        p = precision_at_k(actual, predicted, k)
        r = recall_at_k(actual, predicted, k)
        print(f"At k={k}:")
        print(f"  - Precision@{k} : {p:.4f} (Out of top {k} predictions, {p*100:.1f}% are correct)")
        print(f"  - Recall@{k}    : {r:.4f} (Retrieved {r*100:.1f}% of all actual deficiencies)")
        
    # 2. Batch demonstration (evaluating model performance on a test set)
    print("\n--- 2. Batch/Dataset Evaluation Demo ---")
    
    # A dataset of 3 test palms
    batch_actual = [
        ["Nitrogen"],                          # Palm 1: Only Nitrogen
        ["Nitrogen", "Boron"],                 # Palm 2: Nitrogen and Boron
        ["Potassium", "Magnesium", "Boron"]    # Palm 3: Potassium, Magnesium, Boron
    ]
    
    batch_predicted = [
        ["Nitrogen", "Potassium", "Boron"],    # Model ranked prediction for Palm 1
        ["Boron", "Magnesium", "Nitrogen"],    # Model ranked prediction for Palm 2
        ["Potassium", "Boron", "Nitrogen"]     # Model ranked prediction for Palm 3
    ]
    
    print(f"Test Samples: {len(batch_actual)}")
    
    for k in [1, 2]:
        m_pk = mean_precision_at_k(batch_actual, batch_predicted, k)
        m_rk = mean_recall_at_k(batch_actual, batch_predicted, k)
        print(f"Dataset Mean Metrics at k={k}:")
        print(f"  - Mean Precision@{k} : {m_pk:.4f}")
        print(f"  - Mean Recall@{k}    : {m_rk:.4f}")
    print("="*60)

if __name__ == "__main__":
    run_metric_demos()
