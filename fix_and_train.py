"""
Fix the dataset features (inconsistent scaling), regenerate consistent scores, 
train the model, and verify with test cases.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

def clean_features(df):
    """
    Standardize features to ensure consistent scaling across all records.
    """
    df_clean = df.copy()
    
    print("Cleaning features...")
    
    # 1. Payment History: Should be 0-1
    # Fix values that are likely percentages (e.g. 98.5 -> 0.985)
    df_clean['payment_history'] = df_clean['payment_history'].apply(lambda x: x/100 if x > 1 else x)
    df_clean['payment_history'] = df_clean['payment_history'].clip(0, 1)
    
    # 2. Credit Utilization: Should be 0-100
    # If small values < 1 exist where they should be large (e.g. 0.3 -> 30%), it's hard to tell.
    # But usually utilization is 0-100. Let's assume values > 1 are % and values <= 1 are ratio?
    # No, utilization can be 0%. Let's look at the data.
    # Most are > 1 in the sample (60.0, 6.1, 74.5). So 0-100 is the scale.
    # If any are <= 1, they might be ratios. Let's multiply by 100 if max < 1? 
    # Safer to just clip 0-100 for now, assuming most are correct.
    df_clean['credit_utilization'] = df_clean['credit_utilization'].clip(0, 100)
    
    # 3. Credit Mix: Should be 0-1 (Ratio of different credit types?)
    # Sample had 37.0 and 0.76. 
    df_clean['credit_mix'] = df_clean['credit_mix'].apply(lambda x: x/100 if x > 1 else x)
    df_clean['credit_mix'] = df_clean['credit_mix'].clip(0, 1)
    
    # 4. Ratios that should be 0-1
    ratio_cols = ['income_stability_index', 'digital_transaction_ratio', 'financial_literacy_score', 
                  'micro_loan_repayment', 'savings_ratio', 'debt_to_income_ratio']
    
    for col in ratio_cols:
        # If value > 1, assume it's percentage and divide by 100
        df_clean[col] = df_clean[col].apply(lambda x: x/100 if x > 1 else x)
        df_clean[col] = df_clean[col].clip(0, 1)
        
    # 5. Integer counts/years - just clip to reasonable bounds
    df_clean['age'] = df_clean['age'].clip(18, 70)
    df_clean['work_experience'] = df_clean['work_experience'].clip(0, 50)
    df_clean['length_credit_history'] = df_clean['length_credit_history'].clip(0, 30)
    df_clean['missed_payments_last_year'] = df_clean['missed_payments_last_year'].clip(0, 12)
    df_clean['new_credit_enquiries'] = df_clean['new_credit_enquiries'].clip(0, 20)
    df_clean['number_of_platforms'] = df_clean['number_of_platforms'].clip(1, 10)
    
    return df_clean

def calculate_consistent_scores(df):
    """
    Calculate credit scores based on the CLEANED features to ensure
    perfect correlation for the model to learn.
    """
    print("Regenerating consistent scores...")
    scores = []
    
    for i, row in df.iterrows():
        # Base Score Calculation (More sensitive to negative factors)
        
        # 1. Payment History (30%) - STRICTER
        # Using power of 3 to severely punish anything below 100%
        score_payment = (row['payment_history'] ** 3) * 100
        
        # Penalty for missed payments (Aggressive linear + exponential)
        missed_penalty = 0
        if row['missed_payments_last_year'] > 0:
            missed_penalty = (row['missed_payments_last_year'] * 25) + (row['missed_payments_last_year'] ** 2) * 5
        
        # 2. Credit Utilization (20%) - STRICTER
        if row['credit_utilization'] <= 10:
            util_score = 100
        elif row['credit_utilization'] <= 30:
            util_score = 90 + (30 - row['credit_utilization']) / 20 * 10
        else:
            util_score = max(0, 90 - (row['credit_utilization'] - 30) * 1.5)
        
        # 3. Length of Credit History (10%)
        history_score = min(row['length_credit_history'] / 8, 1.0) * 100
        
        # 4. Credit Mix & New Credit (10%)
        enquiry_penalty = min(row['new_credit_enquiries'] * 10, 100) # Reduced penalty slightly
        card_count_score = min(row['credit_card_count'] * 10, 50) # Up to 5 cards gives points
        mix_score = (row['credit_mix'] * 100 * 0.4) + ((100 - enquiry_penalty) * 0.4) + (card_count_score * 0.2)
        
        # 5. Gig Stability & Income (25%) - Increased weight
        # Include income, stability, platforms, hours, loyalty, experience
        income_score = min(row['monthly_income'] / 8000, 1.0) * 100
        platform_score = min(row['number_of_platforms'] / 5, 1.0) * 100
        hours_score = min(row['avg_weekly_hours'] / 40, 1.0) * 100
        loyalty_score = min(row['platform_loyalty_months'] / 36, 1.0) * 100
        exp_score = min(row['work_experience'] / 10, 1.0) * 100
        
        gig_score = (
            (row['income_stability_index'] * 100 * 0.20) +
            (income_score * 0.40) +  # Doubled weight for income
            (loyalty_score * 0.10) +
            (platform_score * 0.10) +
            (hours_score * 0.10) +
            (exp_score * 0.10)
        )
        
        # 6. Financial Health & Demographics (15%) - Reduced weight
        # Savings, DTI, Emergency, Digital, Micro-loans, Bank Age, Age, Education
        savings_score = min(row['savings_ratio'] / 0.2, 1.0) * 100
        dti_score = max(0, 100 - (row['debt_to_income_ratio'] * 100))
        emergency_score = min(row['emergency_fund_months'] / 6, 1.0) * 100
        bank_age_score = min(row['bank_account_age_months'] / 60, 1.0) * 100
        digital_score = row['digital_transaction_ratio'] * 100
        micro_score = row['micro_loan_repayment'] * 100
        age_score = min((row['age'] - 18) / 40, 1.0) * 100
        edu_score = (row['education_level'] / 5) * 100
        
        health_score = (
            (savings_score * 0.15) +
            (dti_score * 0.15) +
            (emergency_score * 0.15) +
            (bank_age_score * 0.15) +
            (digital_score * 0.10) +
            (micro_score * 0.10) +
            (age_score * 0.10) +
            (edu_score * 0.10)
        )
        
        # Weighted Sum
        raw_score = (
            (score_payment * 0.25) + 
            (util_score * 0.15) + 
            (history_score * 0.10) + 
            (mix_score * 0.10) + 
            (gig_score * 0.25) + # Increased from 0.20
            (health_score * 0.15) # Decreased from 0.20
        )
        
        # Map 0-100 raw_score to 300-850 range
        final_score = 300 + (raw_score / 100) * 550
        
        # Apply penalties directly to final score
        final_score -= missed_penalty
        
        # Apply bonuses
        if row['financial_literacy_score'] > 0.8:
            final_score += 15 # Increased bonus
            
        # Clip to valid range
        final_score = int(np.clip(final_score, 300, 850))
        scores.append(final_score)
        
    return scores

def engineer_features(X):
    """
    Same feature engineering as in the predictor class
    """
    X_enhanced = X.copy()
    
    # Create interaction features
    X_enhanced['payment_income_ratio'] = X['payment_history'] * X['income_stability_index']
    X_enhanced['savings_income_ratio'] = X['savings_ratio'] * X['income_stability_index']
    X_enhanced['utilization_squared'] = X['credit_utilization'] ** 2
    
    # Create age groups
    X_enhanced['age_group'] = pd.cut(X['age'], bins=[17, 25, 35, 45, 56], labels=[1, 2, 3, 4])
    X_enhanced['age_group'] = X_enhanced['age_group'].astype(float)
    X_enhanced['age_group'] = X_enhanced['age_group'].fillna(2)
    
    # Create experience categories
    X_enhanced['experience_category'] = pd.cut(X['work_experience'], 
                                             bins=[-1, 2, 5, 10, 16], 
                                             labels=[1, 2, 3, 4])
    X_enhanced['experience_category'] = X_enhanced['experience_category'].astype(float)
    X_enhanced['experience_category'] = X_enhanced['experience_category'].fillna(1)
    
    # Create risk score
    X_enhanced['financial_risk_score'] = (
        X['debt_to_income_ratio'] * 0.3 +
        (X['credit_utilization'] / 100) * 0.25 +
        X['missed_payments_last_year'] * 0.2 +
        (1 - X['payment_history']) * 0.25
    )
    
    return X_enhanced

def train_and_save_model(df):
    print("\nTraining model...")
    
    X = df.drop('credit_score', axis=1)
    y = df['credit_score']
    
    # Engineer features
    X_enhanced = engineer_features(X)
    feature_names = list(X_enhanced.columns)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_enhanced, y, test_size=0.2, random_state=42)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Gradient Boosting (Robust to non-linearities)
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"Model Performance:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.2f}")
    
    # Calculate feature importance
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    
    # Save
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'feature_importance': feature_importance
    }
    joblib.dump(model_data, 'credit_score_model.pkl')
    print("✅ Model saved to credit_score_model.pkl")
    
    return model, scaler, feature_names

def run_verification_tests(model, scaler, feature_names):
    print("\nRunning verification tests...")
    from test_predictions import run_tests
    # We can reuse the test logic but we need to make sure test_predictions uses the new model
    # Since we just saved it, run_tests should pick it up.
    # But run_tests loads from disk. Let's just call it.
    run_tests()

def main():
    print("="*80)
    print("FIXING DATASET AND RETRAINING MODEL")
    print("="*80)
    
    # 1. Load Data
    try:
        df = pd.read_csv('gig_worker_credit_dataset.csv')
    except:
        # Fallback to cleaned if exists, or original
        df = pd.read_csv('cleaned_gig_worker_data.csv')
        
    print(f"Loaded {len(df)} records")
    
    # 2. Clean Features
    df_clean = clean_features(df)
    
    # 3. Regenerate Scores
    df_clean['credit_score'] = calculate_consistent_scores(df_clean)
    
    # Save fixed dataset
    df_clean.to_csv('gig_worker_credit_dataset.csv', index=False)
    print("✅ Saved fixed dataset to gig_worker_credit_dataset.csv")
    
    # 4. Train Model
    model, scaler, feature_names = train_and_save_model(df_clean)
    
    # 5. Verify
    run_verification_tests(model, scaler, feature_names)

if __name__ == "__main__":
    main()