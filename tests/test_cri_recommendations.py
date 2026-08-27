
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.cri_recommendation_engine import get_visual_recommendation

def test_1_6_month_wet_palm():
    res = get_visual_recommendation("Healthy", 0.9, 6, "young", "Wet")
    assert "YPM-W" in res["fertilizer_type"]
    assert "800 g/palm/6-months" in res["application_rate"]
    assert "Dolomite: 500 g/palm/6-months" in res["application_rate"]
    assert "healthy" in res["deficiency_guidance"]

def test_2_2_year_intermediate_palm():
    res = get_visual_recommendation("Healthy", 0.9, 24, "young", "Intermediate")
    assert "YPM-W" in res["fertilizer_type"]
    assert "1300 g/palm/6-months" in res["application_rate"]

def test_3_4_year_dry_palm():
    res = get_visual_recommendation("Healthy", 0.9, 48, "young", "Dry")
    assert "YPM-D" in res["fertilizer_type"]
    assert "1340 g/palm/6-months" in res["application_rate"]

def test_4_adult_wet_palm():
    res = get_visual_recommendation("Healthy", 0.9, 60, "adult", "Wet")
    assert "APM-W" in res["fertilizer_type"]
    assert "3.30 kg/palm/year" in res["application_rate"]
    assert "Dolomite: 1.00 kg/palm/year" in res["application_rate"]

def test_5_adult_dry_palm():
    res = get_visual_recommendation("Healthy", 0.9, 60, "adult", "Dry")
    assert "APM-D" in res["fertilizer_type"]
    assert "2.80 kg/palm/year" in res["application_rate"]

def test_6_nitrogen_young_palm():
    res = get_visual_recommendation("Nitrogen", 0.9, 12, "young", "Wet")
    assert "Nitrogen Deficiency Corrective Measure" in res["deficiency_guidance"]
    assert "100 g" in res["deficiency_guidance"]

def test_7_nitrogen_adult_palm():
    res = get_visual_recommendation("Nitrogen", 0.9, 60, "adult", "Dry")
    assert "200 g of Urea" in res["deficiency_guidance"]

def test_8_potassium_adult_palm():
    res = get_visual_recommendation("Potassium", 0.9, 60, "adult", "Wet")
    assert "500 g Muriate of Potash" in res["deficiency_guidance"]

def test_9_magnesium_young_palm():
    res = get_visual_recommendation("Magnesium", 0.9, 12, "young", "Dry")
    assert "0.5 kg Kieserite" in res["deficiency_guidance"]

def test_10_magnesium_adult_palm():
    res = get_visual_recommendation("Magnesium", 0.9, 60, "adult", "Wet")
    assert "Old Recommendation" in res["deficiency_guidance"]
    assert "New Recommendation" in res["deficiency_guidance"]

def test_11_boron_seedling_palm():
    res = get_visual_recommendation("Boron", 0.9, 6, "seedling", "Intermediate")
    assert "10 g sodium tetraborate" in res["deficiency_guidance"]

def test_12_boron_mature_palm():
    res = get_visual_recommendation("Boron", 0.9, 60, "adult", "Dry")
    assert "20 g sodium tetraborate" in res["deficiency_guidance"]

def test_13_improved_cultivar():
    res = get_visual_recommendation("Healthy", 0.9, 60, "adult", "Wet", is_high_yielding=True)
    assert "4.95 kg/palm/year" in res["application_rate"]  # 3.3 * 1.5 = 4.95
    assert "Dolomite: 1.50 kg/palm/year" in res["application_rate"]  # 1.0 * 1.5 = 1.5

def test_14_healthy_image():
    res = get_visual_recommendation("Healthy", 0.9, 60, "adult", "Wet")
    assert "visually healthy" in res["deficiency_guidance"]

def test_15_uncertain_image():
    res = get_visual_recommendation("Uncertain", 0.5, 60, "adult", "Wet")
    assert "Uncertain visual diagnosis" in res["deficiency_guidance"]
    assert "preliminary_visual_assessment" in res["assessment_type"]
    assert "CRI baseline fertilizer recommendation is based on A5" in res["disclaimer"]
