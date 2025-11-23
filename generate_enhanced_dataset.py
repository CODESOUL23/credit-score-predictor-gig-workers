"""
Generate a proper dataset that maintains the full 300-850 credit score range
with clear distinctions for different input qualities, based on the cleaned data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def normalize_features_preserve_range(df):
    """
    Normalize features to proper ranges while preserving the original credit score distribution.
    """
    df_norm = df.copy()
    
    # Fix payment_history: should be 0-1 (percentage)
    df_norm['payment_history'] = df_norm['payment_history'] / 100
    df_norm['payment_history'] = df_norm['payment_history'].clip(0, 1)
    
    # Credit utilization is already in good range (5-95)
    df_norm['credit_utilization'] = df_norm['credit_utilization'].clip(0, 100)
    
    # Fix credit_mix: normalize values > 1
    df_norm['credit_mix'] = np.where(df_norm['credit_mix'] > 1, 
                                    df_norm['credit_mix'] / 100, 
                                    df_norm['credit_mix'])
    df_norm['credit_mix'] = df_norm['credit_mix'].clip(0, 1)
    
    # Income stability, digital transaction ratio, savings ratio are mostly OK
    df_norm['income_stability_index'] = df_norm['income_stability_index'].clip(0, 1)
    df_norm['digital_transaction_ratio'] = df_norm['digital_transaction_ratio'].clip(0, 1)
    df_norm['savings_ratio'] = df_norm['savings_ratio'].clip(0, 0.6)  # Allow up to 60% savings
    
    # Fix micro_loan_repayment: normalize values > 1
    df_norm['micro_loan_repayment'] = np.where(df_norm['micro_loan_repayment'] > 1,
                                             df_norm['micro_loan_repayment'] / 100,
                                             df_norm['micro_loan_repayment'])
    df_norm['micro_loan_repayment'] = df_norm['micro_loan_repayment'].clip(0, 1)
    
    # Debt to income ratio is mostly good
    df_norm['debt_to_income_ratio'] = df_norm['debt_to_income_ratio'].clip(0, 1)
    
    # Financial literacy score is mostly good
    df_norm['financial_literacy_score'] = df_norm['financial_literacy_score'].clip(0, 1)
    
    # Keep other features in reasonable ranges
    df_norm['age'] = df_norm['age'].clip(18, 65)
    df_norm['work_experience'] = np.minimum(df_norm['work_experience'], df_norm['age'] - 18)
    df_norm['number_of_platforms'] = df_norm['number_of_platforms'].clip(1, 10)
    df_norm['avg_weekly_hours'] = df_norm['avg_weekly_hours'].clip(10, 80)
    df_norm['emergency_fund_months'] = df_norm['emergency_fund_months'].clip(0, 24)
    df_norm['credit_card_count'] = df_norm['credit_card_count'].clip(0, 15)
    df_norm['missed_payments_last_year'] = df_norm['missed_payments_last_year'].clip(0, 12)
    df_norm['education_level'] = df_norm['education_level'].clip(1, 5)
    
    return df_norm


def enhance_credit_score_range(df):
    """
    Enhance the credit score calculation to maintain full 300-850 range 
    with better distribution and clearer distinctions.
    """
    scores = []
    
    for _, row in df.iterrows():
        # Start with base components (300-750 range from traditional factors)
        base_score = 300
        traditional_max = 450  # Traditional factors can add up to 450 points
        
        # Traditional Credit Factors (70% of total range)
        
        # Payment History (40% of traditional) - Most important
        payment_component = row['payment_history'] * 0.40 * traditional_max
        
        # Credit Utilization (35% of traditional) - Lower is better, more sensitive
        # Use exponential decay for better distinction
        util_ratio = min(row['credit_utilization'] / 100, 1.0)
        util_component = (1 - util_ratio**0.8) * 0.35 * traditional_max
        
        # Length of Credit History (15% of traditional)
        history_component = min(row['length_credit_history'] / 25, 1.0) * 0.15 * traditional_max
        
        # Credit Mix (7% of traditional)
        mix_component = row['credit_mix'] * 0.07 * traditional_max
        
        # New Credit Inquiries (3% of traditional) - Fewer is better
        inquiry_component = (1 - min(row['new_credit_enquiries'] / 8, 1.0)) * 0.03 * traditional_max
        
        # Gig Worker Specific Factors (can add up to 100 more points, reaching 850)
        gig_bonus = 0
        gig_penalty = 0
        
        # Income Stability Bonus (up to +30)
        gig_bonus += row['income_stability_index']**1.5 * 30
        
        # Savings Ratio Bonus (up to +25) - More sensitive to higher savings
        savings_norm = min(row['savings_ratio'] / 0.4, 1.0)  # 40% savings = max
        gig_bonus += savings_norm**1.2 * 25
        
        # Emergency Fund Bonus (up to +20)
        emergency_norm = min(row['emergency_fund_months'] / 8, 1.0)
        gig_bonus += emergency_norm * 20
        
        # Financial Literacy Bonus (up to +15)
        gig_bonus += row['financial_literacy_score'] * 15
        
        # Micro Loan Repayment Bonus (up to +10)
        gig_bonus += row['micro_loan_repayment'] * 10
        
        # Platform Diversification (moderate bonus, up to +8)
        platform_norm = min(row['number_of_platforms'] / 6, 1.0)
        gig_bonus += platform_norm * 8
        
        # Age/Experience Bonus (up to +12 for mature, experienced workers)
        if row['age'] >= 30:
            age_bonus = min((row['age'] - 30) / 25, 1.0) * 8
            exp_bonus = min(row['work_experience'] / 15, 1.0) * 4
            gig_bonus += age_bonus + exp_bonus
        
        # Penalties that reduce score
        
        # Debt-to-Income Penalty (up to -40)
        debt_penalty = row['debt_to_income_ratio']**1.3 * 40
        gig_penalty += debt_penalty
        
        # Missed Payments Penalty (up to -35)
        missed_penalty = min(row['missed_payments_last_year'] / 8, 1.0)**1.2 * 35
        gig_penalty += missed_penalty
        
        # Low Income Instability Penalty (up to -20)
        if row['income_stability_index'] < 0.5:
            instability_penalty = (0.5 - row['income_stability_index']) * 40
            gig_penalty += instability_penalty
        
        # Calculate final score with more variation
        final_score = (base_score + 
                      payment_component + 
                      util_component + 
                      history_component + 
                      mix_component + 
                      inquiry_component + 
                      gig_bonus - 
                      gig_penalty)
        
        # Add some random variation for realism (±5 points)
        variation = np.random.normal(0, 3)
        final_score += variation
        
        # Ensure within valid range
        final_score = int(np.clip(final_score, 300, 850))
        scores.append(final_score)
    
    return scores


def main():
    print("="*70)
    print("GENERATING ENHANCED CREDIT DATASET WITH FULL RANGE")
    print("="*70)
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Load cleaned data
    df = pd.read_csv('cleaned_gig_worker_data.csv')
    print(f"\n✓ Loaded cleaned data: {len(df)} records")
    
    print(f"\nOriginal score range: {df['credit_score'].min()} - {df['credit_score'].max()}")
    
    # Normalize features
    print(f"✓ Normalizing features while preserving score range...")
    df_normalized = normalize_features_preserve_range(df)
    
    # Enhanced credit score calculation
    print(f"✓ Calculating enhanced credit scores with full 300-850 range...")
    df_normalized['credit_score'] = enhance_credit_score_range(df_normalized)
    
    # Show results
    print(f"\nEnhanced score range: {df_normalized['credit_score'].min()} - {df_normalized['credit_score'].max()}")
    print(f"Mean: {df_normalized['credit_score'].mean():.1f}")
    print(f"Std: {df_normalized['credit_score'].std():.1f}")
    
    # Show distribution
    print(f"\nEnhanced Score Distribution:")
    ranges = [
        (300, 500, "Very Poor"),
        (500, 600, "Poor"), 
        (600, 650, "Fair"),
        (650, 700, "Good"),
        (700, 750, "Very Good"),
        (750, 851, "Excellent")
    ]
    
    total_records = len(df_normalized)
    for min_score, max_score, category in ranges:
        count = len(df_normalized[(df_normalized['credit_score'] >= min_score) & (df_normalized['credit_score'] < max_score)])
        percentage = count / total_records * 100
        print(f"  {min_score}-{max_score-1} ({category}): {count} ({percentage:.1f}%)")
    
    # Save the enhanced dataset
    output_file = 'gig_worker_credit_dataset.csv'
    df_normalized.to_csv(output_file, index=False)
    print(f"\n✅ Saved enhanced dataset to: {output_file}")
    
    # Show sample records across score ranges
    print(f"\nSample records by score range:")
    print("="*70)
    
    sample_cols = ['payment_history', 'credit_utilization', 'savings_ratio', 
                  'debt_to_income_ratio', 'missed_payments_last_year', 'credit_score']
    
    for min_score, max_score, category in ranges:
        subset = df_normalized[(df_normalized['credit_score'] >= min_score) & 
                              (df_normalized['credit_score'] < max_score)]
        if len(subset) > 0:
            sample = subset.sample(min(2, len(subset)), random_state=42)
            print(f"\n{category} ({min_score}-{max_score-1}):")
            for _, row in sample.iterrows():
                print(f"  Score {int(row['credit_score'])}: payment={row['payment_history']:.3f}, "
                     f"util={row['credit_utilization']:.1f}%, savings={row['savings_ratio']:.3f}, "
                     f"debt={row['debt_to_income_ratio']:.3f}, missed={int(row['missed_payments_last_year'])}")
    
    print(f"\n" + "="*70)
    print("✅ ENHANCED DATASET GENERATION COMPLETE!")
    print("   Full 300-850 range with clear distinctions for user input changes")
    print("   Next step: Retrain model with: python credit_score_predictor.py")
    print("="*70)


if __name__ == "__main__":
    main()