"""
Company Risk Analysis Module
This script allows financial institutions to predict the probability of loan default for applicants using a supervised ML model.
"""
import joblib
import pandas as pd
import numpy as np
import os
from credit_score_predictor import CreditScorePredictor

# Get the directory where this script is located and construct the model path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'credit_score_model.pkl')

# Threshold for loan approval (customize as needed)
DEFAULT_PROB_THRESHOLD = 0.5

def load_model(path=MODEL_PATH):
    """Load the CreditScorePredictor model."""
    predictor = CreditScorePredictor()
    predictor.load_model(path)
    return predictor

def map_company_input_to_model_features(applicant_data):
    """
    Map company/institution input to the features expected by the credit score model.
    This creates ALL features in the exact order expected by the model.
    """
    
    # Extract basic info
    age = applicant_data.get('age', 30)
    annual_income = applicant_data.get('income', 50000)
    monthly_income = annual_income / 12
    loan_amount = applicant_data.get('loan_amount', 10000)
    employment_length = applicant_data.get('employment_length', 5)
    credit_history_length = applicant_data.get('credit_history_length', 5)
    num_of_loans = applicant_data.get('num_of_loans', 2)
    num_of_defaults = applicant_data.get('num_of_defaults', 0)
    
    # Calculate derived values
    payment_history = max(0.1, 1.0 - (num_of_defaults * 0.3))
    credit_utilization = min(100, max(10, (loan_amount / max(annual_income * 0.3, 1000)) * 100))
    debt_to_income_ratio = min(1.0, loan_amount / max(annual_income, 1000))
    savings_ratio = max(0.05, 0.2 - (num_of_defaults * 0.05))
    
    # Create the feature dictionary in the EXACT order expected by the model
    features = {
        # Original features (first 22)
        'payment_history': payment_history,
        'credit_utilization': credit_utilization,
        'length_credit_history': credit_history_length,
        'credit_mix': 0.7,  # Default assumption
        'new_credit_enquiries': min(10, num_of_loans),
        'income_stability_index': max(0.3, 0.8 - (num_of_defaults * 0.1)),
        'digital_transaction_ratio': 0.8,
        'savings_ratio': savings_ratio,
        'platform_loyalty_months': max(12, employment_length * 12),
        'micro_loan_repayment': max(0.1, 1.0 - (num_of_defaults * 0.2)),
        'age': age,
        'work_experience': employment_length,
        'monthly_income': monthly_income,
        'debt_to_income_ratio': debt_to_income_ratio,
        'number_of_platforms': 3,
        'avg_weekly_hours': 40,
        'emergency_fund_months': max(0, 6 - num_of_defaults * 2),
        'bank_account_age_months': max(12, credit_history_length * 12),
        'credit_card_count': max(1, num_of_loans),
        'missed_payments_last_year': num_of_defaults,
        'education_level': 3,  # Default to bachelor's equivalent
        'financial_literacy_score': 0.6,
        
        # Engineered features (must match the model's feature engineering)
        'payment_income_ratio': payment_history * monthly_income / 1000,  # Scaled ratio
        'savings_income_ratio': savings_ratio * monthly_income / 1000,    # Scaled ratio
        'utilization_squared': (credit_utilization / 100) ** 2,          # Squared utilization
        'age_group': 1 if age < 30 else (2 if age < 50 else 3),         # Age categories
        'experience_category': 1 if employment_length < 3 else (2 if employment_length < 7 else 3),  # Experience categories
    }
    
    # Calculate financial risk score (composite risk indicator)
    financial_risk_score = (
        (1 - payment_history) * 0.3 +           # Payment risk
        (credit_utilization / 100) * 0.3 +      # Utilization risk  
        (debt_to_income_ratio) * 0.2 +          # Debt risk
        (1 - savings_ratio) * 0.1 +             # Savings risk
        (num_of_defaults / 10) * 0.1            # Default history risk
    )
    features['financial_risk_score'] = min(1.0, financial_risk_score)
    
    return features

def predict_default_probability(applicant_data, model=None, threshold=DEFAULT_PROB_THRESHOLD):
    """
    Predict the probability of loan default for an applicant.
    
    Args:
        applicant_data: Dict with keys like 'age', 'income', 'loan_amount', etc.
        model: Pre-loaded CreditScorePredictor model (optional)
        threshold: Probability threshold for approval/rejection decision
    
    Returns:
        Dict with 'probability_of_default' and 'decision'
    """
    if model is None:
        model = load_model()
    
    # Map company input to model features
    mapped_features = map_company_input_to_model_features(applicant_data)
    
    # Create DataFrame with features in the exact order expected by the model
    feature_order = [
        'payment_history', 'credit_utilization', 'length_credit_history', 'credit_mix', 
        'new_credit_enquiries', 'income_stability_index', 'digital_transaction_ratio', 
        'savings_ratio', 'platform_loyalty_months', 'micro_loan_repayment', 'age', 
        'work_experience', 'monthly_income', 'debt_to_income_ratio', 'number_of_platforms', 
        'avg_weekly_hours', 'emergency_fund_months', 'bank_account_age_months', 
        'credit_card_count', 'missed_payments_last_year', 'education_level', 
        'financial_literacy_score', 'payment_income_ratio', 'savings_income_ratio', 
        'utilization_squared', 'age_group', 'experience_category', 'financial_risk_score'
    ]
    
    # Create ordered feature list
    feature_values = [mapped_features[feature] for feature in feature_order]
    
    # Create DataFrame with proper feature names and order
    user_df = pd.DataFrame([feature_values], columns=feature_order)
    
    # Get credit score prediction using the model's predict_credit_score method
    # But we need to bypass the feature engineering since we already have all features
    # So we'll use the model's internal prediction directly
    
    # Scale the features using the model's scaler
    X_scaled = model.scaler.transform(user_df)
    
    # Get the credit score prediction directly from the model
    credit_score = model.model.predict(X_scaled)[0]
    
    # Convert credit score to default probability
    # Lower credit score = higher default probability
    # Scale credit score (300-850) to probability (0-1)
    normalized_score = (credit_score - 300) / (850 - 300)  # Normalize to 0-1
    probability_of_default = 1 - normalized_score  # Invert: lower score = higher risk
    
    # Apply some adjustment to make it more realistic for loan default prediction
    if credit_score < 600:
        probability_of_default = min(0.9, probability_of_default * 1.5)
    elif credit_score < 650:
        probability_of_default = min(0.7, probability_of_default * 1.2)
    
    # Make decision based on threshold
    decision = 'Reject' if probability_of_default >= threshold else 'Approve'
    
    return {
        'probability_of_default': probability_of_default,
        'credit_score': credit_score,
        'decision': decision
    }

def main():
    print("--- Company Risk Analysis ---")
    # Example input; replace with actual input method as needed
    applicant_data = {}
    print("Enter applicant data (leave blank to use example):")
    for field in ['age', 'income', 'loan_amount', 'employment_length', 'credit_history_length', 'num_of_loans', 'num_of_defaults']:
        val = input(f"{field}: ")
        if val:
            applicant_data[field] = float(val)
    if not applicant_data:
        # Example data
        applicant_data = {
            'age': 30,
            'income': 50000,
            'loan_amount': 10000,
            'employment_length': 5,
            'credit_history_length': 7,
            'num_of_loans': 2,
            'num_of_defaults': 0
        }
    model = load_model()
    result = predict_default_probability(applicant_data, model)
    print(f"\nProbability of Default: {result['probability_of_default']:.2%}")
    print(f"Decision: {result['decision']}")

if __name__ == "__main__":
    main()
