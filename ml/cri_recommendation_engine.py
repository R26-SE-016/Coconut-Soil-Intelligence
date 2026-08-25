from typing import Dict, Any, Optional

def get_visual_recommendation(
    nutrient_prediction: str,
    confidence: float,
    palm_age_months: int,
    palm_stage: str,  # "seedling", "young", "adult"
    agro_climatic_zone: str,  # "Wet", "Intermediate", "Dry"
    is_high_yielding: bool = False
) -> Dict[str, Any]:
    """
    PATH 1 - IMAGE BASED RECOMMENDATION
    Generates preliminary visual assessment guidance based on CRI Advisory Circulars A5 and A7.
    """
    
    is_dry = agro_climatic_zone.strip().lower() == "dry"
    
    # 1. A5 Basal Fertilizer Logic
    base_fertilizer_program = ""
    fertilizer_type = ""
    application_rate = ""
    application_frequency = ""
    
    # High yield multiplier
    multiplier = 1.5 if is_high_yielding else 1.0

    if palm_stage.lower() == "adult":
        application_frequency = "Annually (or split half-yearly)"
        if is_dry:
            fertilizer_type = "APM-D (Adult Palm Mixture - Dry) + Dolomite"
            apm = 2.8 * multiplier
            dolomite = 1.0 * multiplier
            application_rate = f"APM-D: {apm:.2f} kg/palm/year, Dolomite: {dolomite:.2f} kg/palm/year"
            base_fertilizer_program = "A5: Apply APM-D and Dolomite for Adult Palms in Dry Zone."
        else:
            fertilizer_type = "APM-W (Adult Palm Mixture - Wet) + Dolomite"
            apm = 3.3 * multiplier
            dolomite = 1.0 * multiplier
            application_rate = f"APM-W: {apm:.2f} kg/palm/year, Dolomite: {dolomite:.2f} kg/palm/year"
            base_fertilizer_program = "A5: Apply APM-W and Dolomite for Adult Palms in Wet/Intermediate Zone."
            
    else:  # Young palm
        application_frequency = "Half-yearly (every 6 months)"
        
        # A5 Young Palm Rates Lookup
        # Age brackets in months: 6, 12, 18, 24, 30, 36, 42, 48
        # We find the closest or bracket.
        if palm_age_months <= 6:
            ypm_w, ypm_d = 800, 540
        elif palm_age_months <= 12:
            ypm_w, ypm_d = 1000, 670
        elif palm_age_months <= 18:
            ypm_w, ypm_d = 1000, 670
        elif palm_age_months <= 24:
            ypm_w, ypm_d = 1300, 905
        elif palm_age_months <= 30:
            ypm_w, ypm_d = 1300, 905
        elif palm_age_months <= 36:
            ypm_w, ypm_d = 1600, 1110
        elif palm_age_months <= 42:
            ypm_w, ypm_d = 1600, 1110
        else:
            ypm_w, ypm_d = 2000, 1340
            
        dolomite_young = 500 * multiplier  # grams
        
        if is_dry:
            fertilizer_type = "YPM-D (Young Palm Mixture - Dry) + Dolomite"
            ypm = ypm_d * multiplier
            application_rate = f"YPM-D: {ypm:.0f} g/palm/6-months, Dolomite: {dolomite_young:.0f} g/palm/6-months"
            base_fertilizer_program = "A5: Apply YPM-D and Dolomite for Young Palms in Dry Zone."
        else:
            fertilizer_type = "YPM-W (Young Palm Mixture - Wet) + Dolomite"
            ypm = ypm_w * multiplier
            application_rate = f"YPM-W: {ypm:.0f} g/palm/6-months, Dolomite: {dolomite_young:.0f} g/palm/6-months"
            base_fertilizer_program = "A5: Apply YPM-W and Dolomite for Young Palms in Wet/Intermediate Zone."

    # 2. A7 Corrective Measures Guidance
    deficiency_guidance = "None detected."
    
    # Supported predictions based on dataset: Healthy, Nitrogen, Boron
    prediction_norm = nutrient_prediction.strip().lower()
    
    if "nitrogen" in prediction_norm:
        if palm_stage.lower() == "adult":
            deficiency_guidance = "A7 Nitrogen Deficiency Corrective Measure: Apply an additional 200 g of Urea per palm."
        else:
            deficiency_guidance = "A7 Nitrogen Deficiency Corrective Measure: Apply an additional 100 g of Urea per palm."
            
    elif "boron" in prediction_norm:
        if palm_stage.lower() == "seedling":
            deficiency_guidance = "A7 Boron Deficiency Corrective Measure: Apply 10 g sodium tetraborate at 6-month intervals until symptoms disappear (Seedling)."
        else:
            deficiency_guidance = "A7 Boron Deficiency Corrective Measure: Apply 20 g sodium tetraborate at 6-month intervals until symptoms disappear (Mature/Young palm)."
            
    # For Potasium or Magnesium (if forced into the system via manual override, provide A7 guidance)
    elif "potassium" in prediction_norm:
        if palm_stage.lower() == "adult":
            deficiency_guidance = "A7 Potassium Deficiency Corrective Measure: Apply an additional 500 g Muriate of Potash per adult palm."
        else:
            deficiency_guidance = "A7 Potassium Deficiency Corrective Measure: No specific young palm corrective dose in A7, ensure standard YPM is applied and verified."
            
    elif "magnesium" in prediction_norm:
        if palm_stage.lower() == "adult":
            deficiency_guidance = (
                "A7 Magnesium Deficiency Corrective Measure:\n"
                "- Old Recommendation: Apply 1 kg Kieserite per palm half-yearly.\n"
                "- New Recommendation: Apply NPK to half the manure circle and 1 kg Kieserite to the other half. For long-term prevention, apply 1 kg Dolomite per palm per year."
            )
        else:
            deficiency_guidance = "A7 Magnesium Deficiency Corrective Measure: Apply 0.5 kg Kieserite per palm half-yearly."
            
    elif "healthy" in prediction_norm:
        deficiency_guidance = "Palm appears visually healthy. Continue standard A5 basal program."
        
    else:
        deficiency_guidance = "Uncertain visual diagnosis. Follow standard A5 basal program and conduct soil/leaf tests."

    return {
        "nutrient_prediction": nutrient_prediction,
        "confidence": round(confidence, 4),
        "palm_age": palm_age_months,
        "palm_stage": palm_stage,
        "agro_climatic_zone": agro_climatic_zone,
        "base_fertilizer_program": base_fertilizer_program,
        "deficiency_guidance": deficiency_guidance,
        "fertilizer_type": fertilizer_type,
        "application_rate": application_rate,
        "application_frequency": application_frequency,
        "source": "CRI A5/A7",
        "assessment_type": "preliminary_visual_assessment",
        "disclaimer": (
            "CRI baseline fertilizer recommendation is based on A5. "
            "CRI deficiency corrective guidance is based on A7. "
            "Visual symptoms may be caused by multiple nutrients or temporary conditions (drought, waterlogging). "
            "This is a preliminary visual assessment. A laboratory leaf/soil analysis is highly recommended for confirmation before treatment."
        )
    }
