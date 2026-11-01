"""
Healthcare Risk Stratification - Training Script (FIXED)

Key fixes vs the original:
1. BUG (main cause of "always High Risk"): features were on very different
   scales (Age ~30-90, TreatmentCost ~2000-10000) but fed into
   LogisticRegression's L2-regularized fit without scaling. Regularization
   penalizes all raw coefficients equally, so features that need larger
   coefficients (Age, AbnormalLabCount) got shrunk almost to nothing, while
   the intercept absorbed the class imbalance (~68% of training outcomes are
   Complicated/Deceased) and pushed nearly every prediction above 0.5.
   FIX: wrap the model in a Pipeline with StandardScaler, and use
   class_weight='balanced' so the intercept doesn't just encode the base rate.
2. BUG: 'Cholestrol' (typo) never matched the real lab test name
   'Cholesterol' in labs.csv, so that lab silently never counted as abnormal.
   FIX: corrected spelling.
3. NOTE (not fixed automatically, flagged for you): LengthOfstay and
   TreatmentCost are only known AFTER a patient's stay ends, so using them to
   predict risk at intake is a form of data leakage / an unrealistic feature
   set for a real deployed tool. Kept them since the app is a portfolio demo,
   but call this out if you present the project - a more realistic version
   would use admission-time features only (age, admission diagnosis, labs at
   admission).
4. Dropped the unused 'HighRisk' column - it was computed but never used as
   the actual target (the model trains on OutcomeEncoded), so it was dead,
   confusing code that didn't match what the app's "High Risk" label means.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import joblib

# --- Load & merge ---
patients = pd.read_csv('patients.csv')
diagnoses = pd.read_csv('diagnoses.csv')
outcomes = pd.read_csv('outcomes.csv')
labs = pd.read_csv('labs.csv')

patients = patients.merge(diagnoses, on='DiagnosisID')
patients = patients.merge(outcomes, on='OutcomeID')

patients['AdmissionDate'] = pd.to_datetime(patients['AdmissionDate'])
patients['DischargeDate'] = pd.to_datetime(patients['DischargeDate'])
patients['LengthOfstay'] = (patients['DischargeDate'] - patients['AdmissionDate']).dt.days

# Target: 0 = Recovered, 1 = Complicated/Deceased
patients['OutcomeEncoded'] = patients['OutcomeName'].map(
    {'Recovered': 0, 'Complicated': 1, 'Deceased': 1}
)

# --- Abnormal lab counts (typo fixed: 'Cholesterol') ---
abnormal_conditions = {
    'Blood Sugar': lambda x: x > 120,
    'Cholesterol': lambda x: x > 200,
    'Hemoglobin': lambda x: x < 13,
}

def count_abnormal_labs(patient_id):
    patient_labs = labs[labs['PatientID'] == patient_id]
    count = 0
    for test_name, condition in abnormal_conditions.items():
        test_results = patient_labs[patient_labs['TestName'] == test_name]
        count += test_results['Result'].apply(condition).sum()
    return count

patients['AbnormalLabCount'] = patients['PatientID'].apply(count_abnormal_labs)

# --- Features / target ---
feature_cols = ['Age', 'LengthOfstay', 'TreatmentCost', 'AbnormalLabCount']
features = patients[feature_cols]
target = patients['OutcomeEncoded']

x_train, x_test, y_train, y_test = train_test_split(
    features, target, test_size=0.3, random_state=42, stratify=target
)

# --- Model: scaling + balanced classes fixes the "always High Risk" bug ---
model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=500, class_weight='balanced')),
])
model.fit(x_train, y_train)

# --- Evaluate ---
y_pred = model.predict(x_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Test set prediction counts:")
print(pd.Series(y_pred).value_counts())

y_prob = model.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.title('Receiver Operating Characteristics')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png', bbox_inches='tight')
plt.close()

# --- Save (note: .pkl, not .ipynb - the original saved a model as a notebook file) ---
joblib.dump(model, 'risk_model1.pkl')
print("Saved model to risk_model1.pkl")

# Sanity check: a clearly "normal" patient should now predict Low Risk
sanity_normal = pd.DataFrame([[30, 3, 3000, 0]], columns=feature_cols)
print("Sanity check - normal patient prediction:",
      model.predict(sanity_normal), model.predict_proba(sanity_normal))