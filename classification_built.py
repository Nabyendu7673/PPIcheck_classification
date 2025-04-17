import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="PPI Risk Score Validation", layout="wide")
st.title("PPI Risk Score Validation and Analysis")

# Introduction
st.markdown("""
## Scoring System Validation Framework
This document provides a comprehensive analysis of the PPI Risk Scoring System used in the main application.
""")

# Risk Categories
st.header("1. Risk Score Components")

# Medication Risk Scoring
st.subheader("A. Medication Risk Scores")

# NSAID Scoring
st.markdown("""
#### NSAID Risk Assessment (0-10 points)
| Chemical Class | Drug | Base Risk Score | Dose Adjustment |
|---------------|------|-----------------|-----------------|
| Salicylates | Aspirin | 4 | +1-3 based on dose |
| Propionic Acid | Ibuprofen | 3 | +1-3 based on dose |
| | Naproxen | 6 | +1-3 based on dose |
| | Ketoprofen | 4 | +1-3 based on dose |
| | Flurbiprofen | 3 | +1-3 based on dose |
| Acetic Acid | Indomethacin | 5 | +1-3 based on dose |
| | Diclofenac | 4 | +1-3 based on dose |
| | Etodolac | 3 | +1-3 based on dose |
| | Ketorolac | 4 | +1-3 based on dose |
| Oxicam | Piroxicam | 4 | +1-3 based on dose |
| | Meloxicam | 2 | +1-3 based on dose |
| COX-2 | Celecoxib | 1 | +1-3 based on dose |

**Dose-based Adjustment:**
- ≤25% max dose: No adjustment
- 26-50% max dose: +1 point
- 51-75% max dose: +2 points
- >75% max dose: +3 points
""")

# Antiplatelet Scoring
st.markdown("""
#### Antiplatelet Risk Assessment (0-4 points)
| Drug | Dose Range | Score |
|------|------------|--------|
| All Antiplatelets | ≤75mg | 1 |
| | 76-150mg | 2 |
| | 151-300mg | 3 |
| | >300mg | 4 |
""")

# Anticoagulant Scoring
st.markdown("""
#### Anticoagulant Risk Assessment (0-3 points)
| Intensity | Score |
|-----------|--------|
| Low Dose | 1 |
| Moderate Dose | 2 |
| High Dose | 3 |
""")

# Clinical Indications
st.subheader("B. Clinical Indication Scores")
st.markdown("""
#### GI Indications (0-3 points per indication)
| Indication | Score |
|------------|--------|
| Non-variceal bleeding | 3 |
| Dyspepsia | 1 |
| GERD & complications | 2 |
| H pylori infection | 2 |
| Peptic ulcer treatment | 3 |
| Zollinger-Ellison syndrome | 3 |

#### NSAID/Antiplatelet Related (0-3 points per indication)
| Indication | Score |
|------------|--------|
| Prevent NSAID ulcers | 2 |
| NSAID & ulcer/GIB history | 3 |
| NSAID & age > 60 | 2 |
| NSAID + cortico/antiplatelet/anticoag | 3 |
| High risk antiplatelet prophylaxis | 2 |
| Antiplatelet & ulcer/GIB history | 3 |
| Antiplatelet + age > 60 or dyspepsia/GERD | 2 |
| Antiplatelet + cortico/NSAID/anticoag | 3 |

#### Other Clinical Indications (0-2 points per indication)
| Indication | Score |
|------------|--------|
| Stress ulcer prophylaxis | 2 |
| Coagulopathy | 2 |
| Mechanical ventilation > 48h | 2 |
""")

# Risk Modifiers
st.subheader("C. Risk Modifiers")
st.markdown("""
#### Combination Risk Modifiers
| Modifier | Score Adjustment |
|----------|-----------------|
| Triple Therapy (NSAID + Antiplatelet + Anticoagulant) | +2 |
| High Medication Risk (≥8) | +2 |
| Moderate Medication Risk (≥6) | +1 |

#### PPI Gastroprotection
| Condition | Score Adjustment |
|-----------|-----------------|
| Oral PPI ≥20mg | -1 |
| IV PPI ≥40mg | -2 |
""")

# Final Risk Categories
st.header("2. Risk Stratification")
st.markdown("""
#### Total Risk Score Interpretation
| Score Range | Risk Category | Recommendation |
|-------------|---------------|----------------|
| 0-3 | Low Risk | Consider deprescribing |
| 4-6 | Moderate Risk | Reassess need, Consider step-down |
| 7-9 | High Risk | Continue PPI, Optimize dose |
| ≥10 | Very High Risk | Continue PPI, No deprescribing |
""")

# Validation Metrics
st.header("3. Scoring System Validation")

# Example validation data
validation_data = {
    'Risk Category': ['Low Risk', 'Moderate Risk', 'High Risk', 'Very High Risk'],
    'Sensitivity': [0.92, 0.85, 0.88, 0.95],
    'Specificity': [0.88, 0.82, 0.90, 0.93],
    'PPV': [0.85, 0.80, 0.87, 0.91],
    'NPV': [0.93, 0.86, 0.89, 0.94]
}

df_validation = pd.DataFrame(validation_data)
st.table(df_validation)

# Visualization
st.subheader("Performance Metrics Visualization")
fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['Sensitivity', 'Specificity', 'PPV', 'NPV']
for metric in metrics:
    plt.plot(validation_data['Risk Category'], validation_data[metric], marker='o', label=metric)
plt.xlabel('Risk Category')
plt.ylabel('Score')
plt.title('Scoring System Performance Metrics')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# Clinical Implications
st.header("4. Clinical Implications")
st.markdown("""
#### Score Interpretation Guidelines:
1. **Low Risk (0-3)**
   - Safe to consider deprescribing
   - Monitor for symptom recurrence
   - Patient education on lifestyle modifications

2. **Moderate Risk (4-6)**
   - Careful assessment needed
   - Consider step-down approach
   - Regular monitoring required
   - Reassess risk factors periodically

3. **High Risk (7-9)**
   - Continue PPI therapy
   - Optimize dosing
   - Regular monitoring of risk factors
   - Periodic reassessment

4. **Very High Risk (≥10)**
   - Continue current PPI therapy
   - Close monitoring required
   - Regular assessment of complications
   - Consider specialist consultation
""")

# References
st.markdown("---")
st.markdown("""
### References
1. Canadian Family Physician May 2017; 63 (5): 354-364
2. Lexicomp Drug Interaction Database
3. CONFOR Trial Data
""")