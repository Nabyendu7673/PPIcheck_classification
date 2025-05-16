import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

st.set_page_config(page_title="PPI Risk Score Validation", layout="wide")
st.title("PPI Risk Score Validation and Analysis")

# Introduction
st.markdown("""
## Scoring System Validation Framework
This document provides a comprehensive analysis and validation framework for the PPI Risk Scoring System.
""")

# Risk Score Components
st.header("1. Risk Score Components")

# Medication Risk Scoring
st.subheader("A. Medication Risk Scores")

# NSAID Scoring
st.markdown("""
#### NSAID Risk Assessment (Base Risk + Dose Adjustment)
Based on PDF Source
| Parameter     | Sub-Parameter                     | Condition / Range   | Score      | Notes           |
|---------------|-----------------------------------|---------------------|------------|-----------------|
| NSAIDS        | Drug Class Dose % of Max          | Propionic Acid (e.g., Ibuprofen) | 3          | Base risk       |
|               |                                   | Acetic Acid (e.g., Diclofenac)   | +4 to +5   | Higher GI risk  |
|               |                                   | Oxicams (e.g., Piroxicam)      | 4          | Higher GI risk  |
|               |                                   | COX-2 inhibitors (e.g., Celecoxib) | 1          | Lower GI risk   |
|               | Adjusted to base risk             | ≤ 25%               | 0          | Adjusted to base risk |
|               |                                   | 26-50%              | 1          |                 |
|               |                                   | 51-75%              | 2          |                 |
|               |                                   | > 75%               | 3          |                 |
""")

# Antiplatelet Scoring
st.markdown("""
#### Antiplatelet Risk Assessment
Based on PDF Source
| Parameter                      | Sub-Parameter                     | Condition / Range   | Score | Notes               |
|--------------------------------|-----------------------------------|---------------------|-------|---------------------|
| Antiplatelets                  | Aspirin (Oral)                    | ≤ 75 mg             | 2     | Base score          |
|                                |                                   | 76-150 mg           | 3     |                     |
|                                |                                   | > 150 mg            | 4     | Loading dose        |
|                                | Clopidogrel, Ticagrelor, Prasugrel (Oral) | ≤ 74 mg             | 2     | Lower GI risk       |
|                                |                                   | 75-299 mg           | 3     | Moderate GI risk    |
|                                |                                   | ≥ 300 mg            | 4     |                     |
|                                | Dipyridamole                      | Any dose            | 1     |                     |
|                                | Ticlopidine                       | Any dose            | 2     |                     |
|                                | Abciximab, Eptifibatide, Tirofiban (IV) | Any IV dose         | 3     | Base + route adjustment |
""")

# Anticoagulant Scoring
st.markdown("""
#### Anticoagulant Risk Assessment
Based on PDF Source
| Parameter           | Sub-Parameter   | Condition / Range         | Score |
|---------------------|-----------------|---------------------------|-------|
| Anticoagulants      | UFH (IV/SC)     | SC 5000 units (Prophylactic) | 2     |
|                     |                 | IV infusion (Therapeutic) | 3     |
|                     | LMWH (Enoxaparin SC) | 20-40 mg (Prophylactic)   | 2     |
|                     |                 | BID 1 mg/kg (Therapeutic) | 3     |
|                     | Dalteparin (SC) | 2500-5000 IU (Prophylactic)| 2     |
|                     |                 | ≥200 IU/kg (Therapeutic)  | 3     |
|                     | Fondaparinux (SC)| 2.5 mg (Low dose)         | 1     |
|                     |                 | ≥ 5 mg (Higher dose)      | 2     |
|                     | Argatroban (IV) | <5 mcg/kg/min (Low dose)  | 2     |
|                     |                 | ≥5 mcg/kg/min (High dose) | 3     |
|                     | Bivalirudin (IV)| <1.5 mg/kg/hr             | 2     |
|                     |                 | ≥1.5 mg/kg/hr             | 3     |
""")

# Clinical Indications
st.subheader("B. Clinical Indication Scores")
st.markdown("""
#### Clinical Indications
Based on PDF Source
| Clinical Indications                       | Condition / Range                  | Score | Notes            |
|--------------------------------------------|------------------------------------|-------|------------------|
| Non-variceal bleeding, PUD, Any ZES        | Any                                | 3     | High-risk indications |
| GERD, H. pylori, NSAID+age>60, Any AP+age>60 | Any                                | 2     | Moderate-risk    |
| Stress ulcer prophylaxis, MV > 48h, coagulopathy | Any                                | 2     |                  |
""")

# Risk Modifiers
st.subheader("C. Risk Modifiers")
st.markdown("""
#### Risk Modifiers
Based on PDF Source
| Parameter             | Sub-Parameter                               | Condition / Range                    | Score Adjustment | Notes                       |
|-----------------------|---------------------------------------------|--------------------------------------|------------------|-----------------------------|
| High-Risk Flag        | Medication Score ≥ 6 OR Indication Score ≥ 6 | Triggered                            | 1                | Auto-calculated based on thresholds |
| Triple Therapy Flag   | NSAID + Antiplatelet + Anticoagulant        | Yes                                  | 2                | Significant GI bleed risk   |
| PPI Protection (-ve)  | Oral PPI ≥20 mg, with NSAID/AP/AC present   | Yes                                  | -1               | Gastroprotection            |
|                       | IV PPI ≥40 mg, with NSAID/AP/AC present     | Yes                                  | -2               | More potent protection      |
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

# Validation Metrics (Keeping the original example data)
st.header("3. Scoring System Validation")

st.markdown("The table below shows example validation metrics for the scoring system across different risk categories.")

validation_data = {
    'Risk Category': ['Low Risk', 'Moderate Risk', 'High Risk', 'Very High Risk'],
    'Sensitivity': [0.92, 0.85, 0.88, 0.95],
    'Specificity': [0.88, 0.82, 0.90, 0.93],
    'PPV': [0.85, 0.80, 0.87, 0.91],
    'NPV': [0.93, 0.86, 0.89, 0.94]
}

df_validation = pd.DataFrame(validation_data)
st.table(df_validation)

# Visualization of Performance Metrics (Keeping the original plot)
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


# Classification Model Evaluation Visualizations
st.header("4. Classification Model Evaluation Visualizations")

st.markdown("""
To further evaluate the performance of the risk scoring system as a classifier (e.g., classifying patients into 'High Risk' vs 'Not High Risk'), we can use visualizations like Confusion Matrix heatmaps and ROC curves.

**Note:** The data used in the following visualizations is sample data for demonstration purposes only and does not represent actual validation results.
""")

# --- Sample Data for Demonstration ---
# Let's simulate a binary classification scenario:
# Class 0: Low/Moderate Risk (Score 0-6)
# Class 1: High/Very High Risk (Score >= 7)

# Assume we have a test set with actual outcomes (e.g., did the patient experience a GI event?)
# and the risk score predicted by the system. We convert scores to predicted classes.

# Sample Actual Labels (0 for No Event, 1 for Event)
# Sample Predicted Labels (0 for Low/Moderate Risk, 1 for High/Very High Risk)
# This is a simplified example. In a real scenario, you'd compare the predicted risk category
# to a defined outcome (e.g., GI bleed).

# For demonstration, let's create sample data that would result in a confusion matrix
# True Positives (TP): Actual Event (1), Predicted High/Very High Risk (1)
# True Negatives (TN): Actual No Event (0), Predicted Low/Moderate Risk (0)
# False Positives (FP): Actual No Event (0), Predicted High/Very High Risk (1)
# False Negatives (FN): Actual Event (1), Predicted Low/Moderate Risk (0)

# Sample Confusion Matrix Counts (Illustrative)
tp = 80
tn = 150
fp = 20
fn = 30

# Create a dummy list of actual and predicted labels to generate the confusion matrix
# Total samples = tp + tn + fp + fn = 80 + 150 + 20 + 30 = 280
actual_labels = [1] * tp + [0] * tn + [0] * fp + [1] * fn
predicted_labels = [1] * tp + [0] * tn + [1] * fp + [0] * fn

# Calculate Confusion Matrix
cm = confusion_matrix(actual_labels, predicted_labels)
cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])

# --- Confusion Matrix Heatmap ---
st.subheader("Confusion Matrix Heatmap")
st.markdown("""
A confusion matrix shows the counts of correct and incorrect predictions made by the classification system compared to the actual outcomes.
- **True Positives (TP):** Correctly predicted positive cases.
- **True Negatives (TN):** Correctly predicted negative cases.
- **False Positives (FP):** Incorrectly predicted positive cases (Type I error).
- **False Negatives (FN):** Incorrectly predicted negative cases (Type II error).
""")

fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
ax_cm.set_xlabel('Predicted Label')
ax_cm.set_ylabel('Actual Label')
ax_cm.set_title('Confusion Matrix')
st.pyplot(fig_cm)

# --- Metrics from Confusion Matrix ---
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0 # Also known as Recall
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

st.subheader("Calculated Metrics (from Sample Data)")
st.write(f"**Accuracy:** {accuracy:.4f}")
st.write(f"**Precision:** {precision:.4f}")
st.write(f"**Sensitivity (Recall):** {sensitivity:.4f}")
st.write(f"**Specificity:** {specificity:.4f}")


# --- ROC Curve ---
st.subheader("ROC Curve")
st.markdown("""
The Receiver Operating Characteristic (ROC) curve illustrates the performance of a binary classifier system as its discrimination threshold is varied. It plots the True Positive Rate (Sensitivity) against the False Positive Rate (1 - Specificity). The Area Under the Curve (AUC) provides an aggregate measure of performance across all possible thresholds. A higher AUC indicates better discriminatory power.
""")

# To generate an ROC curve, we need predicted probabilities and actual labels.
# Since we don't have a model outputting probabilities, we'll simulate some.
# In a real scenario, the 'predicted_probabilities' would come from your model's output
# before applying a hard threshold to get predicted labels.

# Simulate predicted probabilities (e.g., risk scores could be scaled or used directly)
# Let's create probabilities that roughly align with the sample confusion matrix
# Assume higher scores correspond to higher probabilities of being in the positive class (Event)
np.random.seed(42) # for reproducibility

# Simulate probabilities:
# For actual positive cases (Event), probabilities should generally be higher
# For actual negative cases (No Event), probabilities should generally be lower

# Probabilities for Actual Positives (tp + fn)
probs_positive = np.random.beta(a=5, b=2, size=(tp + fn)) # Skewed towards higher values
# Probabilities for Actual Negatives (tn + fp)
probs_negative = np.random.beta(a=2, b=5, size=(tn + fp)) # Skewed towards lower values

predicted_probabilities = np.concatenate((probs_positive, probs_negative))
# Ensure the order matches the actual_labels list created earlier
# Actual labels: [1]*tp + [0]*tn + [0]*fp + [1]*fn
# Probabilities should align: [probs for TP] + [probs for TN] + [probs for FP] + [probs for FN]
# Let's regenerate based on this structure for clarity
predicted_probabilities = np.concatenate([
    np.random.beta(a=5, b=2, size=tp),  # TP probabilities (high)
    np.random.beta(a=2, b=5, size=tn),  # TN probabilities (low)
    np.random.beta(a=5, b=2, size=fp),  # FP probabilities (should be low, but predicted high) - simulate some overlap
    np.random.beta(a=2, b=5, size=fn)   # FN probabilities (should be high, but predicted low) - simulate some overlap
])

# Shuffle the data to mix positives and negatives
combined_data = list(zip(actual_labels, predicted_probabilities))
np.random.shuffle(combined_data)
shuffled_actual_labels, shuffled_predicted_probabilities = zip(*combined_data)


# Calculate ROC curve points
fpr, tpr, thresholds = roc_curve(shuffled_actual_labels, shuffled_predicted_probabilities)
roc_auc = auc(fpr, tpr)

fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess (AUC = 0.50)')
ax_roc.set_xlabel('False Positive Rate (1 - Specificity)')
ax_roc.set_ylabel('True Positive Rate (Sensitivity)')
ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
ax_roc.legend(loc="lower right")
ax_roc.grid(True)
st.pyplot(fig_roc)


# Clinical Implications
st.header("5. Clinical Implications")
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
1. Scoring System PDF (Uploaded)
2. Canadian Family Physician May 2017; 63 (5): 354-364
3. Lexicomp Drug Interaction Database
4. CONFOR Trial Data
""")
