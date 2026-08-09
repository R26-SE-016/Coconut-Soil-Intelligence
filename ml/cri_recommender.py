from typing import Dict, Any

def calculate_cri_fertilizer(leaf_n: float, leaf_p: float, leaf_k: float, soil_ph: float = 6.5) -> Dict[str, Any]:
    """
    Stage 2 of the AI Pipeline:
    Applies Coconut Research Institute (CRI) Differential Fertilizer Recommendation (DFR) rules
    to calculate exact fertilizer dosages in grams per palm per year.
    Based on Advisory Circular No. A5 and DFR tables.
    """
    
    recommendation = {
        "status": "Optimal",
        "dosages_grams_per_palm_per_year": {
            "Urea": 800,
            "Eppawala_Rock_Phosphate_ERP": 600,
            "Muriate_of_Potash_MOP": 1600,
            "Dolomite": 1000
        },
        "nutrient_status": {
            "Nitrogen_N": "Optimal",
            "Phosphorus_P": "Optimal",
            "Potassium_K": "Optimal",
            "Soil_pH": "Normal"
        },
        "agronomic_advice": []
    }

    deficiency_count = 0
    excess_count = 0

    # --- 1. NITROGEN (N) EVALUATION (Optimal: 1.9% - 2.1%) ---
    if leaf_n < 1.60:
        recommendation["nutrient_status"]["Nitrogen_N"] = "Severe Deficiency"
        recommendation["dosages_grams_per_palm_per_year"]["Urea"] = 1100
        recommendation["agronomic_advice"].append("Severe Nitrogen deficiency detected. Increase Urea application and ensure proper mulching to prevent ammonia volatilization.")
        deficiency_count += 1
    elif leaf_n < 1.90:
        recommendation["nutrient_status"]["Nitrogen_N"] = "Moderate Deficiency"
        recommendation["dosages_grams_per_palm_per_year"]["Urea"] = 900
        recommendation["agronomic_advice"].append("Slight Nitrogen deficiency. Apply 900g Urea per palm in split doses during rains.")
        deficiency_count += 1
    elif leaf_n > 2.10:
        recommendation["nutrient_status"]["Nitrogen_N"] = "Excess"
        recommendation["dosages_grams_per_palm_per_year"]["Urea"] = 400
        recommendation["agronomic_advice"].append("Nitrogen levels are above optimal (Excess). Reduce Urea dosage to avoid chemical waste and groundwater leaching.")
        excess_count += 1
    else:
        recommendation["nutrient_status"]["Nitrogen_N"] = "Optimal"
        recommendation["dosages_grams_per_palm_per_year"]["Urea"] = 800

    # --- 2. PHOSPHORUS (P) EVALUATION (Optimal: 0.11% - 0.13%) ---
    if leaf_p < 0.08:
        recommendation["nutrient_status"]["Phosphorus_P"] = "Severe Deficiency"
        recommendation["dosages_grams_per_palm_per_year"]["Eppawala_Rock_Phosphate_ERP"] = 900
        recommendation["agronomic_advice"].append("Severe Phosphorus deficiency. Apply 900g of Eppawala Rock Phosphate (ERP) directly to the topsoil manure circle.")
        deficiency_count += 1
    elif leaf_p < 0.11:
        recommendation["nutrient_status"]["Phosphorus_P"] = "Moderate Deficiency"
        recommendation["dosages_grams_per_palm_per_year"]["Eppawala_Rock_Phosphate_ERP"] = 750
        deficiency_count += 1
    elif leaf_p > 0.13:
        recommendation["nutrient_status"]["Phosphorus_P"] = "Excess"
        recommendation["dosages_grams_per_palm_per_year"]["Eppawala_Rock_Phosphate_ERP"] = 300
        recommendation["agronomic_advice"].append("Phosphorus levels exceed CRI threshold. Reduced ERP dosage recommended.")
        excess_count += 1
    else:
        recommendation["nutrient_status"]["Phosphorus_P"] = "Optimal"
        recommendation["dosages_grams_per_palm_per_year"]["Eppawala_Rock_Phosphate_ERP"] = 600

    # --- 3. POTASSIUM (K) EVALUATION (Optimal: 1.2% - 1.5%) ---
    # Note: Potassium is vital for nut yield and water retention in coconut palms!
    if leaf_k < 0.80:
        recommendation["nutrient_status"]["Potassium_K"] = "Severe Deficiency"
        recommendation["dosages_grams_per_palm_per_year"]["Muriate_of_Potash_MOP"] = 2200
        recommendation["agronomic_advice"].append("Critical Potassium deficiency! This directly causes button shedding and small nuts. Apply 2.2kg MOP per palm.")
        deficiency_count += 1
    elif leaf_k < 1.20:
        recommendation["nutrient_status"]["Potassium_K"] = "Moderate Deficiency"
        recommendation["dosages_grams_per_palm_per_year"]["Muriate_of_Potash_MOP"] = 1800
        deficiency_count += 1
    elif leaf_k > 1.50:
        recommendation["nutrient_status"]["Potassium_K"] = "Excess"
        recommendation["dosages_grams_per_palm_per_year"]["Muriate_of_Potash_MOP"] = 800
        excess_count += 1
    else:
        recommendation["nutrient_status"]["Potassium_K"] = "Optimal"
        recommendation["dosages_grams_per_palm_per_year"]["Muriate_of_Potash_MOP"] = 1600

    # --- 4. SOIL pH & MAGNESIUM (Dolomite Recommendation) ---
    if soil_ph < 5.5:
        recommendation["nutrient_status"]["Soil_pH"] = "Acidic (< 5.5)"
        recommendation["dosages_grams_per_palm_per_year"]["Dolomite"] = 1500
        recommendation["agronomic_advice"].append("Soil is acidic. Apply 1.5kg Dolomite to neutralize acidity and supply Magnesium. IMPORTANT: Do not mix Dolomite with Urea; apply separately.")
    elif soil_ph > 8.0:
        recommendation["nutrient_status"]["Soil_pH"] = "Alkaline (> 8.0)"
        recommendation["dosages_grams_per_palm_per_year"]["Dolomite"] = 500
    else:
        recommendation["nutrient_status"]["Soil_pH"] = "Optimal (5.5 - 7.5)"
        recommendation["dosages_grams_per_palm_per_year"]["Dolomite"] = 1000

    # --- Overall Palm Health Status Classification ---
    if deficiency_count >= 2:
        recommendation["status"] = "Severe Nutrient Deficiency"
    elif deficiency_count == 1:
        recommendation["status"] = "Moderate Deficiency"
    elif excess_count >= 2:
        recommendation["status"] = "Excess Fertilizer (Risk of Waste)"
    else:
        recommendation["status"] = "Optimal Health"

    if not recommendation["agronomic_advice"]:
        recommendation["agronomic_advice"].append("Palm is in optimal nutrient condition. Maintain standard basal application rates and mulch the manure circle.")

    return recommendation
