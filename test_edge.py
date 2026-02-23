import joblib
import pandas as pd

model = joblib.load('model.pkl')

# Edge case: CIBIL Score = 300, Income = 0
edge_case_1 = pd.DataFrame({
    'no_of_dependents': [0],
    'education': ['Not Graduate'],
    'self_employed': ['No'],
    'income_annum': [0],
    'loan_amount': [5000000],
    'loan_term': [20],
    'cibil_score': [300],
    'residential_assets_value': [0],
    'commercial_assets_value': [0],
    'luxury_assets_value': [0],
    'bank_asset_value': [0]
})

pred_edge_1 = model.predict(edge_case_1)[0]
proba_edge_1 = max(model.predict_proba(edge_case_1)[0])

print("Edge Case 1 (CIBIL=300, Income=0):")
print(f"Prediction: {'Approved' if pred_edge_1 == 1 else 'Rejected'}")
print(f"Confidence: {proba_edge_1*100:.1f}%")

# Edge Case 2: CIBIL Score = 900, High Income
edge_case_2 = pd.DataFrame({
    'no_of_dependents': [2],
    'education': ['Graduate'],
    'self_employed': ['Yes'],
    'income_annum': [9000000],
    'loan_amount': [15000000],
    'loan_term': [10],
    'cibil_score': [850],
    'residential_assets_value': [10000000],
    'commercial_assets_value': [5000000],
    'luxury_assets_value': [20000000],
    'bank_asset_value': [5000000]
})

pred_edge_2 = model.predict(edge_case_2)[0]
proba_edge_2 = max(model.predict_proba(edge_case_2)[0])

print("\nEdge Case 2 (CIBIL=850, High Income/Assets):")
print(f"Prediction: {'Approved' if pred_edge_2 == 1 else 'Rejected'}")
print(f"Confidence: {proba_edge_2*100:.1f}%")
