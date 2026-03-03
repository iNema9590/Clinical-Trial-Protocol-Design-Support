import pandas as pd
import numpy as np

np.random.seed(42)

# Create eligible patients (40 patients that meet inclusion criteria)
eligible_data = pd.DataFrame({
    "PATIENT_ID": range(1, 41),
    "AGE": np.random.randint(18, 70, 40),  # >= 18 years
    "BMI": np.random.uniform(18.5, 30, 40),  # Normal BMI
    "ALT": np.random.uniform(10, 35, 40),  # Normal ALT (< 40 ULN)
    "ULN": 40,
    "PCR_RESULT": ["Negative"] * 40,  # Negative PCR required
    "SEVERE_ALLERGY_HISTORY": [False] * 40,  # No severe allergies
    "IMMUNOSUPPRESSIVE_THERAPY_6M": [False] * 40,  # No immunosuppressive therapy
    "PREGNANT": [False] * 40,  # Not pregnant
    "BODY_TEMPERATURE": np.random.uniform(36.2, 37.5, 40),  # Normal temp
    "RECENT_VACCINE_30D": [False] * 40,  # No recent vaccine
    "RECENT_BLOOD_DONATION_30D": [False] * 40,  # No recent blood donation
    "SARS_COV2_RISK_LEVEL": ["Low"] * 40,  # Low risk
    "MEDICAL_STABILITY_STATUS": ["Stable"] * 40,  # Medically stable
    "COGNITIVE_COMPLIANCE_CAPABLE": [True] * 40,  # Can comply
    "USING_CONTRACEPTION": [True] * 30 + [False] * 10,  # Some using contraception
    "CONSENTED": [True] * 40,  # All consented
    "GUILLAIN_BARRE_HISTORY": [False] * 40,
    "IMMUNODEFICIENCY_CONDITION": [False] * 40,
    "MALIGNANCY_HISTORY": [False] * 40,
    "BLEEDING_DISORDER_HISTORY": [False] * 40,
    "SEVERE_COMORBIDITY": [False] * 40,
    "INVESTIGATIONAL_SARS_COV2_DRUG": [False] * 40,
    "RECENT_IMMUNOGLOBULIN_3M": [False] * 40,
    "STUDY_STAFF_INVOLVEMENT": [False] * 40,
})

# Create ineligible patients (10 patients with various violations)
ineligible_data = pd.DataFrame({
    "PATIENT_ID": range(41, 51),
    "AGE": [15, 16, 17, 25, 30, 45, 55, 60, 72, 50],  # One under 18
    "BMI": [15, 18, 32, 35.5, 28, 19, 22, 26, 24, 29],  # Some too high
    "ALT": [45, 50, 80, 65, 35, 40, 150, 30, 25, 95],  # Some elevated
    "ULN": 40,
    "PCR_RESULT": ["Positive", "Negative", "Positive", "Negative", "Negative", 
                   "Positive", "Negative", "Negative", "Positive", "Negative"],  # Some positive
    "SEVERE_ALLERGY_HISTORY": [True, False, False, False, False, 
                               False, True, False, False, True],  # Some with allergies
    "IMMUNOSUPPRESSIVE_THERAPY_6M": [False, True, False, False, False, 
                                      True, False, False, True, False],  # Some on therapy
    "PREGNANT": [False, False, True, False, False, False, False, False, False, False],  # One pregnant
    "BODY_TEMPERATURE": [36.8, 37.2, 38.1, 37.5, 36.5, 38.5, 37.0, 36.9, 37.3, 38.2],  # Some elevated
    "RECENT_VACCINE_30D": [False, True, False, False, False, False, True, False, False, False],  # Some with vaccine
    "RECENT_BLOOD_DONATION_30D": [False, False, True, False, False, False, False, True, False, False],  # Some donated
    "SARS_COV2_RISK_LEVEL": ["High", "Low", "High", "Low", "Low", "High", "Low", "Low", "High", "Low"],
    "MEDICAL_STABILITY_STATUS": ["Unstable", "Stable", "Unstable", "Stable", "Stable", 
                                  "Unstable", "Stable", "Stable", "Unstable", "Stable"],
    "COGNITIVE_COMPLIANCE_CAPABLE": [False, True, True, False, True, True, True, True, True, False],
    "USING_CONTRACEPTION": [False, False, False, True, True, False, True, True, True, True],
    "CONSENTED": [True, True, False, True, True, True, True, True, True, False],
    "GUILLAIN_BARRE_HISTORY": [False, True, False, False, False, False, False, True, False, False],
    "IMMUNODEFICIENCY_CONDITION": [False, False, True, False, False, False, True, False, False, False],
    "MALIGNANCY_HISTORY": [False, False, False, True, False, True, False, False, False, False],
    "BLEEDING_DISORDER_HISTORY": [False, False, False, False, True, False, False, False, True, False],
    "SEVERE_COMORBIDITY": [True, False, True, False, False, True, False, True, False, False],
    "INVESTIGATIONAL_SARS_COV2_DRUG": [False, False, False, False, False, True, False, False, False, False],
    "RECENT_IMMUNOGLOBULIN_3M": [False, False, False, True, False, False, True, False, False, False],
    "STUDY_STAFF_INVOLVEMENT": [False, False, False, False, False, False, False, False, True, False],
})

# Combine eligible and ineligible data
data = pd.concat([eligible_data, ineligible_data], ignore_index=False).reset_index(drop=True)

# Save to CSV
data.to_csv("synthetic_patient_data.csv", index=False)
print(f"Generated {len(data)} patient records ({len(eligible_data)} eligible, {len(ineligible_data)} ineligible)")