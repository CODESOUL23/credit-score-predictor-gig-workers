#!/usr/bin/env python3
"""
Demo script to showcase the Credit Score Predictor functionality.
This script demonstrates the system with sample data.
"""

from credit_score_predictor import CreditScorePredictor
import json

def demo_prediction():
    """Run a demonstration of the credit score predictor."""
    
    print("🎯 CREDIT SCORE PREDICTOR DEMO")
    print("="*50)
    
    # Load the model
    predictor = CreditScorePredictor()
    try:
        predictor.load_model('credit_score_model.pkl')
        print("✅ Model loaded successfully!")
    except FileNotFoundError:
        print("❌ Model file not found. Training new model...")
        # Load and train if model doesn't exist
        X, y = predictor.load_and_preprocess_data('gig_worker_credit_dataset.csv')
        X_enhanced = predictor.engineer_features(X)
        predictor.train_models(X_enhanced, y)
        predictor.save_model('credit_score_model.pkl')
    
    # Demo scenarios
    scenarios = [
        {
            "name": "🌟 Excellent Gig Worker",
            "description": "High payment history, low credit utilization, good savings",
            "data": {
                'payment_history': 0.95,
                'credit_utilization': 15.0,
                'length_credit_history': 8,
                'credit_mix': 0.8,
                'new_credit_enquiries': 1,
                'income_stability_index': 0.9,
                'digital_transaction_ratio': 0.95,
                'savings_ratio': 0.25,
                'platform_loyalty_months': 36,
                'micro_loan_repayment': 0.98,
                'age': 35,
                'work_experience': 10,
                'monthly_income': 6500.0,
                'debt_to_income_ratio': 0.2,
                'number_of_platforms': 2,
                'avg_weekly_hours': 40.0,
                'emergency_fund_months': 8.0,
                'bank_account_age_months': 60,
                'credit_card_count': 3,
                'missed_payments_last_year': 0,
                'education_level': 4,
                'financial_literacy_score': 0.85
            }
        },
        {
            "name": "⚠️ Struggling Gig Worker",
            "description": "Poor payment history, high credit utilization, low savings",
            "data": {
                'payment_history': 0.65,
                'credit_utilization': 85.0,
                'length_credit_history': 2,
                'credit_mix': 0.3,
                'new_credit_enquiries': 5,
                'income_stability_index': 0.4,
                'digital_transaction_ratio': 0.6,
                'savings_ratio': 0.05,
                'platform_loyalty_months': 8,
                'micro_loan_repayment': 0.7,
                'age': 25,
                'work_experience': 2,
                'monthly_income': 2800.0,
                'debt_to_income_ratio': 0.6,
                'number_of_platforms': 5,
                'avg_weekly_hours': 25.0,
                'emergency_fund_months': 0.5,
                'bank_account_age_months': 18,
                'credit_card_count': 4,
                'missed_payments_last_year': 3,
                'education_level': 2,
                'financial_literacy_score': 0.3
            }
        },
        {
            "name": "📈 Average Gig Worker",
            "description": "Typical gig worker with room for improvement",
            "data": {
                'payment_history': 0.8,
                'credit_utilization': 50.0,
                'length_credit_history': 4,
                'credit_mix': 0.6,
                'new_credit_enquiries': 3,
                'income_stability_index': 0.7,
                'digital_transaction_ratio': 0.8,
                'savings_ratio': 0.12,
                'platform_loyalty_months': 18,
                'micro_loan_repayment': 0.85,
                'age': 30,
                'work_experience': 5,
                'monthly_income': 4200.0,
                'debt_to_income_ratio': 0.4,
                'number_of_platforms': 3,
                'avg_weekly_hours': 32.0,
                'emergency_fund_months': 2.0,
                'bank_account_age_months': 36,
                'credit_card_count': 2,
                'missed_payments_last_year': 1,
                'education_level': 3,
                'financial_literacy_score': 0.6
            }
        }
    ]
    
    # Run predictions for each scenario
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*60}")
        print(f"SCENARIO {i}: {scenario['name']}")
        print(f"{'='*60}")
        print(f"Description: {scenario['description']}")
        
        # Make prediction
        predicted_score, analysis = predictor.predict_credit_score(scenario['data'])
        
        # Display results
        print(f"\n📊 Predicted Credit Score: {predicted_score}")
        print(f"📋 Score Category: {analysis['score_category']}")
        
        # Create score visualization
        score_bar = create_score_bar(predicted_score)
        print(f"\n{score_bar}")
        
        # Show analysis
        if analysis['strengths']:
            print(f"\n✅ Strengths:")
            for strength in analysis['strengths']:
                print(f"   • {strength}")
        
        if analysis['areas_for_improvement']:
            print(f"\n⚠️  Areas for Improvement:")
            for area in analysis['areas_for_improvement']:
                print(f"   • {area}")
        
        if analysis['specific_suggestions']:
            print(f"\n💡 Specific Suggestions:")
            for suggestion in analysis['specific_suggestions']:
                print(f"   • {suggestion}")
        
        print(f"\n📈 Key Metrics:")
        print(f"   • Credit Utilization: {scenario['data']['credit_utilization']:.1f}%")
        print(f"   • Debt-to-Income: {scenario['data']['debt_to_income_ratio']:.1%}")
        print(f"   • Savings Rate: {scenario['data']['savings_ratio']:.1%}")
        print(f"   • Emergency Fund: {scenario['data']['emergency_fund_months']:.1f} months")

def create_score_bar(score: float) -> str:
    """Create a visual score bar."""
    bar_length = 50
    score_position = int((score - 300) / (850 - 300) * bar_length)
    
    bar = ""
    for i in range(bar_length):
        if i == score_position:
            bar += "█"
        elif i < score_position:
            if i < bar_length * 0.3:
                bar += "▓"  # Poor range
            elif i < bar_length * 0.6:
                bar += "▒"  # Fair range
            else:
                bar += "░"  # Good range
        else:
            bar += "."
    
    return f"300 {bar} 850"

def show_feature_importance():
    """Display feature importance from the trained model."""
    predictor = CreditScorePredictor()
    predictor.load_model('credit_score_model.pkl')
    
    if predictor.feature_importance is not None:
        print(f"\n{'='*60}")
        print("📊 TOP 10 MOST IMPORTANT FEATURES")
        print(f"{'='*60}")
        
        top_features = predictor.feature_importance.head(10)
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            importance_bar = "█" * int(row['importance'] * 50)
            print(f"{i:2d}. {row['feature']:<25} {importance_bar} {row['importance']:.3f}")

def main():
    """Main demo function."""
    print("🚀 Starting Credit Score Predictor Demo...\n")
    
    # Run demo predictions
    demo_prediction()
    
    # Show feature importance
    show_feature_importance()
    
    print(f"\n{'='*60}")
    print("✅ DEMO COMPLETE!")
    print(f"{'='*60}")
    print("💡 To use the interactive version:")
    print("   • Command line: python credit_score_cli.py")
    print("   • Web app: streamlit run streamlit_app.py")
    print("📚 Check README.md for complete documentation")

if __name__ == "__main__":
    main()