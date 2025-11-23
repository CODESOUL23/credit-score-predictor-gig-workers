import pandas as pd
import numpy as np
import joblib
import os

def engineer_features(X):
    """
    Replicate the feature engineering logic from CreditScorePredictor
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

def run_tests():
    print("="*80)
    print("RUNNING 20 COMPREHENSIVE TEST CASES")
    print("="*80)
    
    # Load model
    try:
        model_data = joblib.load('credit_score_model.pkl')
        model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Define 20 test cases covering various profiles
    test_cases = [
        # 1. Perfect Profile (Should be ~850)
        {
            'name': 'Perfect Profile',
            'payment_history': 1.0, 'credit_utilization': 5.0, 'length_credit_history': 20,
            'credit_mix': 1.0, 'new_credit_enquiries': 0, 'age': 45, 'work_experience': 15,
            'monthly_income': 15000, 'debt_to_income_ratio': 0.1, 'savings_ratio': 0.4,
            'emergency_fund_months': 12, 'number_of_platforms': 5, 'avg_weekly_hours': 40,
            'platform_loyalty_months': 60, 'income_stability_index': 1.0, 'digital_transaction_ratio': 1.0,
            'credit_card_count': 3, 'bank_account_age_months': 120, 'missed_payments_last_year': 0,
            'education_level': 5, 'financial_literacy_score': 1.0, 'micro_loan_repayment': 1.0
        },
        # 2. Excellent Profile (Should be ~800)
        {
            'name': 'Excellent Profile',
            'payment_history': 0.98, 'credit_utilization': 10.0, 'length_credit_history': 15,
            'credit_mix': 0.9, 'new_credit_enquiries': 1, 'age': 35, 'work_experience': 10,
            'monthly_income': 10000, 'debt_to_income_ratio': 0.2, 'savings_ratio': 0.3,
            'emergency_fund_months': 6, 'number_of_platforms': 4, 'avg_weekly_hours': 45,
            'platform_loyalty_months': 48, 'income_stability_index': 0.9, 'digital_transaction_ratio': 0.9,
            'credit_card_count': 4, 'bank_account_age_months': 96, 'missed_payments_last_year': 0,
            'education_level': 4, 'financial_literacy_score': 0.9, 'micro_loan_repayment': 1.0
        },
        # 3. Very Good Profile (Should be ~750)
        {
            'name': 'Very Good Profile',
            'payment_history': 0.95, 'credit_utilization': 20.0, 'length_credit_history': 10,
            'credit_mix': 0.8, 'new_credit_enquiries': 2, 'age': 30, 'work_experience': 8,
            'monthly_income': 8000, 'debt_to_income_ratio': 0.25, 'savings_ratio': 0.25,
            'emergency_fund_months': 4, 'number_of_platforms': 3, 'avg_weekly_hours': 40,
            'platform_loyalty_months': 36, 'income_stability_index': 0.85, 'digital_transaction_ratio': 0.8,
            'credit_card_count': 3, 'bank_account_age_months': 72, 'missed_payments_last_year': 0,
            'education_level': 4, 'financial_literacy_score': 0.8, 'micro_loan_repayment': 1.0
        },
        # 4. Good Profile (Should be ~700)
        {
            'name': 'Good Profile',
            'payment_history': 0.90, 'credit_utilization': 30.0, 'length_credit_history': 8,
            'credit_mix': 0.7, 'new_credit_enquiries': 2, 'age': 28, 'work_experience': 6,
            'monthly_income': 6000, 'debt_to_income_ratio': 0.3, 'savings_ratio': 0.2,
            'emergency_fund_months': 3, 'number_of_platforms': 3, 'avg_weekly_hours': 35,
            'platform_loyalty_months': 24, 'income_stability_index': 0.8, 'digital_transaction_ratio': 0.7,
            'credit_card_count': 2, 'bank_account_age_months': 48, 'missed_payments_last_year': 1,
            'education_level': 3, 'financial_literacy_score': 0.7, 'micro_loan_repayment': 0.9
        },
        # 5. Fair Profile (Should be ~650)
        {
            'name': 'Fair Profile',
            'payment_history': 0.85, 'credit_utilization': 45.0, 'length_credit_history': 5,
            'credit_mix': 0.6, 'new_credit_enquiries': 3, 'age': 26, 'work_experience': 4,
            'monthly_income': 4500, 'debt_to_income_ratio': 0.4, 'savings_ratio': 0.15,
            'emergency_fund_months': 2, 'number_of_platforms': 2, 'avg_weekly_hours': 30,
            'platform_loyalty_months': 18, 'income_stability_index': 0.7, 'digital_transaction_ratio': 0.6,
            'credit_card_count': 2, 'bank_account_age_months': 36, 'missed_payments_last_year': 2,
            'education_level': 3, 'financial_literacy_score': 0.6, 'micro_loan_repayment': 0.8
        },
        # 6. Poor Profile (Should be ~550)
        {
            'name': 'Poor Profile',
            'payment_history': 0.70, 'credit_utilization': 60.0, 'length_credit_history': 3,
            'credit_mix': 0.4, 'new_credit_enquiries': 5, 'age': 24, 'work_experience': 2,
            'monthly_income': 3000, 'debt_to_income_ratio': 0.5, 'savings_ratio': 0.1,
            'emergency_fund_months': 1, 'number_of_platforms': 2, 'avg_weekly_hours': 25,
            'platform_loyalty_months': 12, 'income_stability_index': 0.6, 'digital_transaction_ratio': 0.5,
            'credit_card_count': 3, 'bank_account_age_months': 24, 'missed_payments_last_year': 4,
            'education_level': 2, 'financial_literacy_score': 0.5, 'micro_loan_repayment': 0.7
        },
        # 7. Very Poor Profile (Should be ~450)
        {
            'name': 'Very Poor Profile',
            'payment_history': 0.50, 'credit_utilization': 80.0, 'length_credit_history': 1,
            'credit_mix': 0.2, 'new_credit_enquiries': 8, 'age': 22, 'work_experience': 1,
            'monthly_income': 2000, 'debt_to_income_ratio': 0.6, 'savings_ratio': 0.05,
            'emergency_fund_months': 0, 'number_of_platforms': 1, 'avg_weekly_hours': 20,
            'platform_loyalty_months': 6, 'income_stability_index': 0.4, 'digital_transaction_ratio': 0.3,
            'credit_card_count': 4, 'bank_account_age_months': 12, 'missed_payments_last_year': 6,
            'education_level': 2, 'financial_literacy_score': 0.3, 'micro_loan_repayment': 0.5
        },
        # 8. Critical Profile (Should be ~350)
        {
            'name': 'Critical Profile',
            'payment_history': 0.30, 'credit_utilization': 95.0, 'length_credit_history': 0,
            'credit_mix': 0.1, 'new_credit_enquiries': 10, 'age': 20, 'work_experience': 0,
            'monthly_income': 1500, 'debt_to_income_ratio': 0.8, 'savings_ratio': 0.0,
            'emergency_fund_months': 0, 'number_of_platforms': 1, 'avg_weekly_hours': 15,
            'platform_loyalty_months': 3, 'income_stability_index': 0.2, 'digital_transaction_ratio': 0.2,
            'credit_card_count': 5, 'bank_account_age_months': 6, 'missed_payments_last_year': 10,
            'education_level': 1, 'financial_literacy_score': 0.2, 'micro_loan_repayment': 0.3
        },
        # 9. High Income but Bad Habits (Should be low/fair)
        {
            'name': 'High Income Bad Habits',
            'payment_history': 0.60, 'credit_utilization': 85.0, 'length_credit_history': 10,
            'credit_mix': 0.5, 'new_credit_enquiries': 6, 'age': 40, 'work_experience': 15,
            'monthly_income': 12000, 'debt_to_income_ratio': 0.7, 'savings_ratio': 0.05,
            'emergency_fund_months': 1, 'number_of_platforms': 3, 'avg_weekly_hours': 50,
            'platform_loyalty_months': 48, 'income_stability_index': 0.8, 'digital_transaction_ratio': 0.9,
            'credit_card_count': 6, 'bank_account_age_months': 100, 'missed_payments_last_year': 5,
            'education_level': 4, 'financial_literacy_score': 0.4, 'micro_loan_repayment': 0.6
        },
        # 10. Low Income but Perfect Habits (Should be good/very good)
        {
            'name': 'Low Income Perfect Habits',
            'payment_history': 1.0, 'credit_utilization': 5.0, 'length_credit_history': 5,
            'credit_mix': 0.6, 'new_credit_enquiries': 0, 'age': 25, 'work_experience': 3,
            'monthly_income': 2500, 'debt_to_income_ratio': 0.1, 'savings_ratio': 0.3,
            'emergency_fund_months': 6, 'number_of_platforms': 2, 'avg_weekly_hours': 30,
            'platform_loyalty_months': 24, 'income_stability_index': 0.9, 'digital_transaction_ratio': 0.5,
            'credit_card_count': 1, 'bank_account_age_months': 36, 'missed_payments_last_year': 0,
            'education_level': 3, 'financial_literacy_score': 0.8, 'micro_loan_repayment': 1.0
        },
        # 11. Student/New Worker (Thin file)
        {
            'name': 'New Worker',
            'payment_history': 1.0, 'credit_utilization': 10.0, 'length_credit_history': 1,
            'credit_mix': 0.2, 'new_credit_enquiries': 1, 'age': 21, 'work_experience': 1,
            'monthly_income': 2000, 'debt_to_income_ratio': 0.1, 'savings_ratio': 0.2,
            'emergency_fund_months': 2, 'number_of_platforms': 2, 'avg_weekly_hours': 20,
            'platform_loyalty_months': 6, 'income_stability_index': 0.6, 'digital_transaction_ratio': 0.8,
            'credit_card_count': 1, 'bank_account_age_months': 12, 'missed_payments_last_year': 0,
            'education_level': 3, 'financial_literacy_score': 0.6, 'micro_loan_repayment': 1.0
        },
        # 12. Gig Power User (High platform usage)
        {
            'name': 'Gig Power User',
            'payment_history': 0.95, 'credit_utilization': 30.0, 'length_credit_history': 6,
            'credit_mix': 0.7, 'new_credit_enquiries': 2, 'age': 29, 'work_experience': 5,
            'monthly_income': 7000, 'debt_to_income_ratio': 0.3, 'savings_ratio': 0.2,
            'emergency_fund_months': 3, 'number_of_platforms': 8, 'avg_weekly_hours': 60,
            'platform_loyalty_months': 36, 'income_stability_index': 0.7, 'digital_transaction_ratio': 1.0,
            'credit_card_count': 3, 'bank_account_age_months': 48, 'missed_payments_last_year': 0,
            'education_level': 3, 'financial_literacy_score': 0.7, 'micro_loan_repayment': 0.9
        },
        # 13. Recovering Debtor (Improving history)
        {
            'name': 'Recovering Debtor',
            'payment_history': 0.80, 'credit_utilization': 50.0, 'length_credit_history': 12,
            'credit_mix': 0.6, 'new_credit_enquiries': 0, 'age': 38, 'work_experience': 10,
            'monthly_income': 5000, 'debt_to_income_ratio': 0.4, 'savings_ratio': 0.1,
            'emergency_fund_months': 1, 'number_of_platforms': 3, 'avg_weekly_hours': 40,
            'platform_loyalty_months': 24, 'income_stability_index': 0.8, 'digital_transaction_ratio': 0.6,
            'credit_card_count': 4, 'bank_account_age_months': 80, 'missed_payments_last_year': 1,
            'education_level': 3, 'financial_literacy_score': 0.6, 'micro_loan_repayment': 0.8
        },
        # 14. Maxed Out Cards (High utilization)
        {
            'name': 'Maxed Out Cards',
            'payment_history': 0.90, 'credit_utilization': 90.0, 'length_credit_history': 8,
            'credit_mix': 0.5, 'new_credit_enquiries': 0, 'age': 32, 'work_experience': 7,
            'monthly_income': 5500, 'debt_to_income_ratio': 0.6, 'savings_ratio': 0.05,
            'emergency_fund_months': 1, 'number_of_platforms': 3, 'avg_weekly_hours': 40,
            'platform_loyalty_months': 30, 'income_stability_index': 0.8, 'digital_transaction_ratio': 0.7,
            'credit_card_count': 5, 'bank_account_age_months': 60, 'missed_payments_last_year': 0,
            'education_level': 3, 'financial_literacy_score': 0.5, 'micro_loan_repayment': 0.7
        },
        # 15. Cash Only User (Low digital footprint)
        {
            'name': 'Cash Only User',
            'payment_history': 0.90, 'credit_utilization': 10.0, 'length_credit_history': 15,
            'credit_mix': 0.3, 'new_credit_enquiries': 0, 'age': 50, 'work_experience': 20,
            'monthly_income': 4000, 'debt_to_income_ratio': 0.2, 'savings_ratio': 0.3,
            'emergency_fund_months': 6, 'number_of_platforms': 1, 'avg_weekly_hours': 30,
            'platform_loyalty_months': 60, 'income_stability_index': 0.9, 'digital_transaction_ratio': 0.1,
            'credit_card_count': 1, 'bank_account_age_months': 150, 'missed_payments_last_year': 0,
            'education_level': 3, 'financial_literacy_score': 0.4, 'micro_loan_repayment': 0.5
        },
        # 16. Recent Graduate (High potential)
        {
            'name': 'Recent Graduate',
            'payment_history': 1.0, 'credit_utilization': 20.0, 'length_credit_history': 2,
            'credit_mix': 0.4, 'new_credit_enquiries': 2, 'age': 23, 'work_experience': 1,
            'monthly_income': 4000, 'debt_to_income_ratio': 0.3, 'savings_ratio': 0.1,
            'emergency_fund_months': 1, 'number_of_platforms': 2, 'avg_weekly_hours': 40,
            'platform_loyalty_months': 12, 'income_stability_index': 0.7, 'digital_transaction_ratio': 0.9,
            'credit_card_count': 2, 'bank_account_age_months': 24, 'missed_payments_last_year': 0,
            'education_level': 5, 'financial_literacy_score': 0.8, 'micro_loan_repayment': 1.0
        },
        # 17. Struggling Parent (High expenses)
        {
            'name': 'Struggling Parent',
            'payment_history': 0.85, 'credit_utilization': 70.0, 'length_credit_history': 10,
            'credit_mix': 0.6, 'new_credit_enquiries': 3, 'age': 35, 'work_experience': 10,
            'monthly_income': 4500, 'debt_to_income_ratio': 0.7, 'savings_ratio': 0.0,
            'emergency_fund_months': 0, 'number_of_platforms': 2, 'avg_weekly_hours': 50,
            'platform_loyalty_months': 36, 'income_stability_index': 0.8, 'digital_transaction_ratio': 0.6,
            'credit_card_count': 4, 'bank_account_age_months': 80, 'missed_payments_last_year': 2,
            'education_level': 3, 'financial_literacy_score': 0.5, 'micro_loan_repayment': 0.7
        },
        # 18. Crypto Enthusiast (High risk tolerance)
        {
            'name': 'Crypto Enthusiast',
            'payment_history': 0.80, 'credit_utilization': 40.0, 'length_credit_history': 4,
            'credit_mix': 0.5, 'new_credit_enquiries': 5, 'age': 27, 'work_experience': 4,
            'monthly_income': 6000, 'debt_to_income_ratio': 0.4, 'savings_ratio': 0.1,
            'emergency_fund_months': 2, 'number_of_platforms': 4, 'avg_weekly_hours': 35,
            'platform_loyalty_months': 18, 'income_stability_index': 0.5, 'digital_transaction_ratio': 1.0,
            'credit_card_count': 3, 'bank_account_age_months': 36, 'missed_payments_last_year': 1,
            'education_level': 4, 'financial_literacy_score': 0.7, 'micro_loan_repayment': 0.8
        },
        # 19. Retired Gig Worker (Stable but low income)
        {
            'name': 'Retired Gig Worker',
            'payment_history': 1.0, 'credit_utilization': 5.0, 'length_credit_history': 25,
            'credit_mix': 0.5, 'new_credit_enquiries': 0, 'age': 65, 'work_experience': 40,
            'monthly_income': 2000, 'debt_to_income_ratio': 0.1, 'savings_ratio': 0.2,
            'emergency_fund_months': 12, 'number_of_platforms': 1, 'avg_weekly_hours': 15,
            'platform_loyalty_months': 60, 'income_stability_index': 1.0, 'digital_transaction_ratio': 0.4,
            'credit_card_count': 2, 'bank_account_age_months': 200, 'missed_payments_last_year': 0,
            'education_level': 3, 'financial_literacy_score': 0.6, 'micro_loan_repayment': 1.0
        },
        # 20. Average Joe (Middle of the road)
        {
            'name': 'Average Joe',
            'payment_history': 0.90, 'credit_utilization': 30.0, 'length_credit_history': 7,
            'credit_mix': 0.5, 'new_credit_enquiries': 1, 'age': 30, 'work_experience': 5,
            'monthly_income': 5000, 'debt_to_income_ratio': 0.3, 'savings_ratio': 0.1,
            'emergency_fund_months': 3, 'number_of_platforms': 2, 'avg_weekly_hours': 40,
            'platform_loyalty_months': 24, 'income_stability_index': 0.8, 'digital_transaction_ratio': 0.7,
            'credit_card_count': 2, 'bank_account_age_months': 48, 'missed_payments_last_year': 0,
            'education_level': 3, 'financial_literacy_score': 0.6, 'micro_loan_repayment': 0.8
        }
    ]
    
    # Run predictions
    results = []
    
    for case in test_cases:
        # Prepare data
        df = pd.DataFrame([case])
        
        # Engineer features
        df_enhanced = engineer_features(df)
        
        # Ensure all features exist
        for feature in feature_names:
            if feature not in df_enhanced.columns:
                df_enhanced[feature] = 0
        
        # Select features
        X_test = df_enhanced[feature_names]
        
        # Predict (Always scale as the model is trained on scaled data)
        try:
            X_scaled = scaler.transform(X_test)
            pred = model.predict(X_scaled)[0]
        except Exception as e:
            print(f"Prediction error for {case['name']}: {e}")
            pred = 0
            
        results.append({
            'name': case['name'],
            'score': int(pred),
            'payment_history': case['payment_history'],
            'utilization': case['credit_utilization'],
            'missed': case['missed_payments_last_year']
        })
        
    # Display results
    print(f"{'PROFILE NAME':<25} | {'SCORE':<5} | {'PAYMENT':<7} | {'UTIL':<5} | {'MISSED':<6} | {'RATING'}")
    print("-" * 80)
    
    for r in results:
        score = r['score']
        rating = "Excellent" if score >= 750 else "Very Good" if score >= 700 else "Good" if score >= 650 else "Fair" if score >= 550 else "Poor" if score >= 500 else "Very Poor"
        print(f"{r['name']:<25} | {score:<5} | {r['payment_history']:<7.2f} | {r['utilization']:<5.1f} | {r['missed']:<6} | {rating}")

if __name__ == "__main__":
    run_tests()