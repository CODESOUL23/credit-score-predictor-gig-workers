#!/usr/bin/env python3
"""
Credit Score Predictor CLI - Command Line Interface for Gig Workers
A simple command-line tool to predict credit scores and get personalized suggestions.
"""

import sys
from credit_score_predictor import CreditScorePredictor
from typing import Dict, Any

class CreditScoreCLI:
    def __init__(self):
        self.predictor = CreditScorePredictor()
        self.load_model()
    
    def load_model(self):
        """Load the trained model."""
        try:
            self.predictor.load_model('credit_score_model.pkl')
            print("✅ Model loaded successfully!")
        except FileNotFoundError:
            print("❌ Model file not found. Please train the model first by running:")
            print("python credit_score_predictor.py")
            sys.exit(1)
    
    def print_header(self):
        """Print application header."""
        print("\n" + "="*60)
        print("🎯 CREDIT SCORE PREDICTOR FOR GIG WORKERS")
        print("="*60)
        print("Get personalized credit score predictions and improvement suggestions!")
        print()
    
    def get_user_input(self) -> Dict[str, float]:
        """Collect user input through interactive prompts."""
        print("📝 Please provide your financial information:")
        print("(Press Enter for default values in brackets)\n")
        
        user_data = {}
        
        # Personal Information
        print("👤 PERSONAL INFORMATION")
        print("-" * 25)
        user_data['age'] = self.get_numeric_input("Age", 30, 18, 65)
        user_data['work_experience'] = self.get_numeric_input("Work Experience (years)", 5, 0, 20)
        user_data['education_level'] = self.get_education_level()
        
        # Financial History
        print("\n💳 FINANCIAL HISTORY")
        print("-" * 20)
        user_data['payment_history'] = self.get_numeric_input("Payment History (0.0-1.0)", 0.85, 0.0, 1.0)
        user_data['credit_utilization'] = self.get_numeric_input("Credit Utilization (%)", 45.0, 0.0, 100.0)
        user_data['length_credit_history'] = self.get_numeric_input("Length of Credit History (years)", 5, 0, 20)
        user_data['credit_mix'] = self.get_numeric_input("Credit Mix Diversity (0.0-1.0)", 0.7, 0.0, 1.0)
        user_data['new_credit_enquiries'] = self.get_numeric_input("New Credit Enquiries (last year)", 2, 0, 10)
        user_data['credit_card_count'] = self.get_numeric_input("Number of Credit Cards", 2, 0, 10)
        user_data['missed_payments_last_year'] = self.get_numeric_input("Missed Payments (last year)", 1, 0, 12)
        
        # Income and Gig Work
        print("\n💰 INCOME & GIG WORK")
        print("-" * 20)
        user_data['monthly_income'] = self.get_numeric_input("Monthly Income ($)", 4500, 1000, 15000)
        user_data['income_stability_index'] = self.get_numeric_input("Income Stability Index (0.0-1.0)", 0.75, 0.0, 1.0)
        user_data['number_of_platforms'] = self.get_numeric_input("Number of Gig Platforms", 3, 1, 10)
        user_data['avg_weekly_hours'] = self.get_numeric_input("Average Weekly Hours", 35.0, 10.0, 80.0)
        user_data['platform_loyalty_months'] = self.get_numeric_input("Platform Loyalty (months)", 24, 1, 60)
        
        # Financial Health
        print("\n🏦 FINANCIAL HEALTH")
        print("-" * 19)
        user_data['savings_ratio'] = self.get_numeric_input("Savings Ratio (% of income as decimal)", 0.15, 0.0, 0.5)
        user_data['debt_to_income_ratio'] = self.get_numeric_input("Debt-to-Income Ratio", 0.35, 0.0, 1.0)
        user_data['emergency_fund_months'] = self.get_numeric_input("Emergency Fund (months of expenses)", 2.0, 0.0, 24.0)
        
        # Additional Factors
        print("\n📊 ADDITIONAL FACTORS")
        print("-" * 21)
        user_data['digital_transaction_ratio'] = self.get_numeric_input("Digital Transaction Ratio (0.0-1.0)", 0.8, 0.0, 1.0)
        user_data['micro_loan_repayment'] = self.get_numeric_input("Micro Loan Repayment Rate (0.0-1.0)", 0.9, 0.0, 1.0)
        user_data['bank_account_age_months'] = self.get_numeric_input("Bank Account Age (months)", 48, 6, 120)
        user_data['financial_literacy_score'] = self.get_numeric_input("Financial Literacy Score (0.0-1.0)", 0.6, 0.0, 1.0)
        
        return user_data
    
    def get_numeric_input(self, prompt: str, default: float, min_val: float, max_val: float) -> float:
        """Get numeric input with validation."""
        while True:
            try:
                user_input = input(f"{prompt} [{default}]: ").strip()
                if user_input == "":
                    return default
                
                value = float(user_input)
                if min_val <= value <= max_val:
                    return value
                else:
                    print(f"❌ Please enter a value between {min_val} and {max_val}")
            except ValueError:
                print("❌ Please enter a valid number")
    
    def get_education_level(self) -> int:
        """Get education level with options."""
        print("Education Level:")
        print("  1 - High School")
        print("  2 - Some College")
        print("  3 - Bachelor's Degree")
        print("  4 - Master's Degree")
        print("  5 - PhD/Doctorate")
        
        while True:
            try:
                user_input = input("Select education level (1-5) [3]: ").strip()
                if user_input == "":
                    return 3
                
                value = int(user_input)
                if 1 <= value <= 5:
                    return value
                else:
                    print("❌ Please enter a number between 1 and 5")
            except ValueError:
                print("❌ Please enter a valid number")
    
    def display_results(self, predicted_score: float, analysis: Dict[str, Any]):
        """Display prediction results and analysis."""
        print("\n" + "="*60)
        print("🎯 CREDIT SCORE PREDICTION RESULTS")
        print("="*60)
        
        # Score display
        score_bar = self.create_score_bar(predicted_score)
        print(f"\n📊 Your Predicted Credit Score: {predicted_score}")
        print(f"📋 Score Category: {analysis['score_category']}")
        print(f"\n{score_bar}")
        
        # Analysis
        if analysis['strengths']:
            print(f"\n✅ YOUR STRENGTHS:")
            for i, strength in enumerate(analysis['strengths'], 1):
                print(f"   {i}. {strength}")
        
        if analysis['areas_for_improvement']:
            print(f"\n⚠️  AREAS FOR IMPROVEMENT:")
            for i, area in enumerate(analysis['areas_for_improvement'], 1):
                print(f"   {i}. {area}")
        
        if analysis['specific_suggestions']:
            print(f"\n💡 PERSONALIZED SUGGESTIONS:")
            for i, suggestion in enumerate(analysis['specific_suggestions'], 1):
                print(f"   {i}. {suggestion}")
        
        # Score interpretation
        self.display_score_interpretation(predicted_score)
    
    def create_score_bar(self, score: float) -> str:
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
    
    def display_score_interpretation(self, score: float):
        """Display score interpretation and next steps."""
        print(f"\n📈 SCORE INTERPRETATION:")
        
        if score >= 750:
            print("   🌟 EXCELLENT! You have exceptional credit.")
            print("   → You qualify for the best interest rates and terms.")
            print("   → Focus on maintaining your excellent financial habits.")
        elif score >= 700:
            print("   ✅ GOOD! You have strong credit.")
            print("   → You qualify for competitive interest rates.")
            print("   → Small improvements can get you to excellent credit.")
        elif score >= 650:
            print("   🔶 FAIR. You have average credit.")
            print("   → You may face higher interest rates.")
            print("   → Focus on the improvement suggestions above.")
        elif score >= 600:
            print("   ⚠️  POOR. Your credit needs work.")
            print("   → Limited credit options with higher costs.")
            print("   → Follow the suggestions to rebuild your credit.")
        else:
            print("   🚨 VERY POOR. Immediate action needed.")
            print("   → Very limited credit options.")
            print("   → Consider credit counseling and follow all suggestions.")
        
        print(f"\n📞 NEXT STEPS:")
        print("   1. Implement the personalized suggestions above")
        print("   2. Monitor your credit regularly")
        print("   3. Re-run this tool in 3-6 months to track progress")
        print("   4. Consider consulting a financial advisor for complex situations")
    
    def run(self):
        """Run the CLI application."""
        self.print_header()
        
        try:
            # Get user input
            user_data = self.get_user_input()
            
            print("\n🔮 Calculating your credit score...")
            
            # Make prediction
            predicted_score, analysis = self.predictor.predict_credit_score(user_data)
            
            # Display results
            self.display_results(predicted_score, analysis)
            
            # Ask for another prediction
            print(f"\n" + "="*60)
            another = input("Would you like to make another prediction? (y/n): ").strip().lower()
            if another in ['y', 'yes']:
                self.run()
            else:
                print("Thank you for using Credit Score Predictor! 👋")
                
        except KeyboardInterrupt:
            print("\n\nExiting... Thank you for using Credit Score Predictor! 👋")
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please try again or contact support.")

def main():
    """Main function."""
    cli = CreditScoreCLI()
    cli.run()

if __name__ == "__main__":
    main()