import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def train_and_evaluate_models():
    print("="*70)
    print("COCONUT SOIL TO LEAF NPK - MULTI-MODEL TRAINING & EVALUATION")
    print("="*70)

    # 1. Locate Dataset
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "All.csv")
    if not os.path.exists(data_path):
        # Fallback to parent directory if running from somewhere else
        data_path = os.path.join(os.path.dirname(base_dir), "All.csv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find All.csv at {data_path}")

    print(f"[*] Loading empirical dataset from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]
    print(f"[*] Dataset Shape: {df.shape[0]} palms, Columns: {list(df.columns)}")

    # Ensure required columns exist
    required_cols = ['Soil N', 'Soil P', 'Soil K', 'Leaf N', 'Leaf P', 'Leaf K']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Convert to numeric, coercing errors to NaN
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows where target Leaf NPK has any missing value (supervised learning requires valid targets)
    df = df.dropna(subset=['Leaf N', 'Leaf P', 'Leaf K'], how='any')
    # For any missing soil feature values, fill with the median of that feature
    for col in ['Soil N', 'Soil P', 'Soil K']:
        df[col] = df[col].fillna(df[col].median())
    print(f"[*] Valid samples after cleaning target labels: {len(df)}")

    # Feature Matrix X (Soil NPK) and Target Matrix Y (14th Leaf NPK)
    X = df[['Soil N', 'Soil P', 'Soil K']]
    Y = df[['Leaf N', 'Leaf P', 'Leaf K']]

    # 2. Train-Test Split (80% Train, 20% Test for Research Validation)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.20, random_state=42
    )
    print(f"[*] Data Split: {len(X_train)} Training samples | {len(X_test)} Testing samples\n")

    # 3. Define 3 Candidate Machine Learning Pipelines
    # Model 1: Random Forest Regressor (Ensemble Trees)
    rf_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42))
    ])

    # Model 2: Extra Trees Regressor (Extremely Randomized Trees - great for small/noisy datasets)
    et_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', ExtraTreesRegressor(n_estimators=200, max_depth=12, random_state=42))
    ])

    # Model 3: K-Nearest Neighbors Regressor (Spatial distance-based regression)
    knn_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', MultiOutputRegressor(KNeighborsRegressor(n_neighbors=5, weights='distance')))
    ])

    # Model 4: Gradient Boosting Regressor (HistGradientBoosting)
    gb_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', MultiOutputRegressor(HistGradientBoostingRegressor(random_state=42, max_iter=200)))
    ])

    # Model 5: Polynomial Ridge Regression (Captures non-linear NPK interaction curves)
    poly_ridge_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('model', MultiOutputRegressor(Ridge(alpha=1.0, random_state=42)))
    ])

    models = {
        "Random Forest (Ensemble)": rf_pipeline,
        "Extra Trees Regressor": et_pipeline,
        "Gradient Boosting (HistGB)": gb_pipeline,
        "Polynomial Ridge Regression": poly_ridge_pipeline,
        "K-Nearest Neighbors (KNN)": knn_pipeline
    }

    results = {}
    best_model_name = None
    best_model_pipeline = None
    best_avg_r2 = -float('inf')

    print("--- MODEL EVALUATION RESULTS ---")
    
    # 4. Train & Evaluate Each Model
    for name, pipeline in models.items():
        print(f"\nTraining [{name}]...")
        pipeline.fit(X_train, Y_train)
        
        # Predictions on Test Set
        Y_pred = pipeline.predict(X_test)
        
        # Calculate Metrics per Nutrient
        r2_n = r2_score(Y_test['Leaf N'], Y_pred[:, 0])
        r2_p = r2_score(Y_test['Leaf P'], Y_pred[:, 1])
        r2_k = r2_score(Y_test['Leaf K'], Y_pred[:, 2])
        avg_r2 = (r2_n + r2_p + r2_k) / 3.0
        
        rmse_n = np.sqrt(mean_squared_error(Y_test['Leaf N'], Y_pred[:, 0]))
        rmse_p = np.sqrt(mean_squared_error(Y_test['Leaf P'], Y_pred[:, 1]))
        rmse_k = np.sqrt(mean_squared_error(Y_test['Leaf K'], Y_pred[:, 2]))
        avg_rmse = (rmse_n + rmse_p + rmse_k) / 3.0

        mae_n = mean_absolute_error(Y_test['Leaf N'], Y_pred[:, 0])
        mae_p = mean_absolute_error(Y_test['Leaf P'], Y_pred[:, 1])
        mae_k = mean_absolute_error(Y_test['Leaf K'], Y_pred[:, 2])
        avg_mae = (mae_n + mae_p + mae_k) / 3.0

        results[name] = {
            "R2_Score": {
                "Nitrogen (N)": round(r2_n, 4),
                "Phosphorus (P)": round(r2_p, 4),
                "Potassium (K)": round(r2_k, 4),
                "Average_R2": round(avg_r2, 4)
            },
            "RMSE": {
                "Nitrogen (N)": round(rmse_n, 4),
                "Phosphorus (P)": round(rmse_p, 4),
                "Potassium (K)": round(rmse_k, 4),
                "Average_RMSE": round(avg_rmse, 4)
            },
            "MAE": {
                "Nitrogen (N)": round(mae_n, 4),
                "Phosphorus (P)": round(mae_p, 4),
                "Potassium (K)": round(mae_k, 4),
                "Average_MAE": round(avg_mae, 4)
            }
        }

        print(f"   -> Average R² Score : {avg_r2:.4f} (N: {r2_n:.2f}, P: {r2_p:.2f}, K: {r2_k:.2f})")
        print(f"   -> Average RMSE     : {avg_rmse:.4f}")
        print(f"   -> Average MAE      : {avg_mae:.4f}")

        if avg_r2 > best_avg_r2:
            best_avg_r2 = avg_r2
            best_model_name = name
            best_model_pipeline = pipeline

    print("\n" + "="*70)
    print(f"[WINNER] WINNING MODEL SELECTED: {best_model_name} (Best Average R2: {best_avg_r2:.4f})")
    print("="*70)

    # 5. Save Winning Model & Report
    saved_models_dir = os.path.join(base_dir, "ml", "saved_models")
    os.makedirs(saved_models_dir, exist_ok=True)

    model_save_path = os.path.join(saved_models_dir, "best_soil_to_leaf_model.joblib")
    joblib.dump(best_model_pipeline, model_save_path)
    print(f"[OK] Best Model successfully saved to: {model_save_path}")

    report_save_path = os.path.join(saved_models_dir, "model_comparison_report.json")
    with open(report_save_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_model": best_model_name,
            "best_average_r2": round(best_avg_r2, 4),
            "all_models_comparison": results,
            "dataset_info": {
                "total_samples": len(df),
                "train_samples": len(X_train),
                "test_samples": len(X_test)
            }
        }, f, indent=4)
    print(f"[OK] Comparison Report saved to: {report_save_path}\n")
    
    return best_model_name, best_avg_r2, results

if __name__ == "__main__":
    train_and_evaluate_models()
