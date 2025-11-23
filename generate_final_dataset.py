"""
Generate the final proper dataset that maintains the FULL 300-850 range
with excellent distribution and clear distinctions for user inputs.
"""

import pandas as pd
import numpy as np

def normalize_features_final(df):
    """
    Final normalization that keeps features in proper ranges.
    """
    df_norm = df.copy()
    
    # Fix payment_history: should be 0-1
    df_norm['payment_history'] = df_norm['payment_history'] / 100
    df_norm['payment_history'] = df_norm['payment_history'].clip(0, 1)
    
    # Keep credit_utilization as percentage (0-100)
    df_norm['credit_utilization'] = df_norm['credit_utilization'].clip(0, 100)
    
    # Fix credit_mix: normalize if > 1
    df_norm['credit_mix'] = np.where(df_norm['credit_mix'] > 1, 
                                    df_norm['credit_mix'] / 100, 
                                    df_norm['credit_mix'])
    df_norm['credit_mix'] = df_norm['credit_mix'].clip(0, 1)
    
    # Keep other ratios in 0-1 range
    ratio_cols = ['income_stability_index', 'digital_transaction_ratio', 'financial_literacy_score']
    for col in ratio_cols:
        df_norm[col] = df_norm[col].clip(0, 1)
    
    # Savings ratio can go up to 50%
    df_norm['savings_ratio'] = df_norm['savings_ratio'].clip(0, 0.5)
    
    # Debt to income up to 100%
    df_norm['debt_to_income_ratio'] = df_norm['debt_to_income_ratio'].clip(0, 1.0)
    
    # Fix micro_loan_repayment
    df_norm['micro_loan_repayment'] = np.where(df_norm['micro_loan_repayment'] > 1,
                                             df_norm['micro_loan_repayment'] / 100,
                                             df_norm['micro_loan_repayment'])
    df_norm['micro_loan_repayment'] = df_norm['micro_loan_repayment'].clip(0, 1)
    
    # Reasonable ranges for other features
    df_norm['age'] = df_norm['age'].clip(18, 65)
    df_norm['work_experience'] = np.minimum(df_norm['work_experience'], df_norm['age'] - 18).clip(0, 47)
    df_norm['length_credit_history'] = df_norm['length_credit_history'].clip(0, 25)
    df_norm['number_of_platforms'] = df_norm['number_of_platforms'].clip(1, 10)
    df_norm['avg_weekly_hours'] = df_norm['avg_weekly_hours'].clip(10, 80)
    df_norm['emergency_fund_months'] = df_norm['emergency_fund_months'].clip(0, 24)
    df_norm['credit_card_count'] = df_norm['credit_card_count'].clip(0, 15)
    df_norm['missed_payments_last_year'] = df_norm['missed_payments_last_year'].clip(0, 12)
    df_norm['education_level'] = df_norm['education_level'].clip(1, 5)
    df_norm['new_credit_enquiries'] = df_norm['new_credit_enquiries'].clip(0, 10)
    df_norm['platform_loyalty_months'] = df_norm['platform_loyalty_months'].clip(1, 120)
    df_norm['bank_account_age_months'] = df_norm['bank_account_age_months'].clip(6, 240)
    df_norm['monthly_income'] = df_norm['monthly_income'].clip(1000, 20000)
    
    return df_norm


def calculate_final_credit_scores(df):
    """
    Calculate credit scores using a method that ensures FULL 300-850 range distribution.
    """
    # First, use the original scores as a base and enhance them
    original_scores = df['credit_score'].copy()
    
    # Create score categories based on financial profile quality
    enhanced_scores = []
    
    for i, row in df.iterrows():
        # Calculate quality metrics (0-1 scale)
        
        # Payment quality (40% weight)
        payment_quality = row['payment_history'] * (1 - min(row['missed_payments_last_year'] / 8, 1.0))
        
        # Credit usage quality (30% weight)
        usage_quality = 1 - min(row['credit_utilization'] / 100, 1.0)
        
        # Financial stability (20% weight)  
        stability_quality = (
            row['income_stability_index'] * 0.4 +
            min(row['savings_ratio'] / 0.3, 1.0) * 0.3 +
            (1 - min(row['debt_to_income_ratio'], 1.0)) * 0.3
        )
        
        # Credit experience (10% weight)
        experience_quality = (
            min(row['length_credit_history'] / 20, 1.0) * 0.5 +
            row['credit_mix'] * 0.3 +
            (1 - min(row['new_credit_enquiries'] / 6, 1.0)) * 0.2
        )
        
        # Overall quality score (0-1)
        overall_quality = (
            payment_quality * 0.40 +
            usage_quality * 0.30 +
            stability_quality * 0.20 +
            experience_quality * 0.10
        )
        
        # Map quality to score range with full 300-850 spread
        if overall_quality >= 0.9:  # Exceptional (830-850)
            base_score = 830 + (overall_quality - 0.9) * 200  # 830-850
        elif overall_quality >= 0.8:  # Excellent (750-830)
            base_score = 750 + (overall_quality - 0.8) * 800  # 750-830
        elif overall_quality >= 0.7:  # Very Good (680-750)
            base_score = 680 + (overall_quality - 0.7) * 700  # 680-750
        elif overall_quality >= 0.6:  # Good (620-680)
            base_score = 620 + (overall_quality - 0.6) * 600  # 620-680
        elif overall_quality >= 0.4:  # Fair (520-620)
            base_score = 520 + (overall_quality - 0.4) * 500  # 520-620
        elif overall_quality >= 0.2:  # Poor (400-520)
            base_score = 400 + (overall_quality - 0.2) * 600  # 400-520
        else:  # Very Poor (300-400)
            base_score = 300 + overall_quality * 500  # 300-400
        
        # Add gig-worker specific adjustments (±30 points)
        gig_adjustment = 0
        
        # Platform diversification bonus
        gig_adjustment += min(row['number_of_platforms'] / 5, 1.0) * 15
        
        # Emergency fund bonus
        gig_adjustment += min(row['emergency_fund_months'] / 6, 1.0) * 10
        
        # Financial literacy bonus
        gig_adjustment += row['financial_literacy_score'] * 8
        
        # Micro loan repayment bonus
        gig_adjustment += row['micro_loan_repayment'] * 7
        
        # Age/experience penalty for very young or old workers
        if row['age'] < 25:
            gig_adjustment -= (25 - row['age']) * 2
        elif row['age'] > 60:
            gig_adjustment -= (row['age'] - 60) * 1
        
        # Low income penalty (below $3000/month)
        if row['monthly_income'] < 3000:
            gig_adjustment -= (3000 - row['monthly_income']) / 100
        
        # Final score calculation
        final_score = base_score + gig_adjustment
        
        # Add some natural variation (±8 points)
        variation = np.random.normal(0, 5)
        final_score += variation
        
        # Ensure within bounds
        final_score = int(np.clip(final_score, 300, 850))
        enhanced_scores.append(final_score)
    
    return enhanced_scores


def main():
    print("="*70)
    print("GENERATING FINAL DATASET WITH FULL 300-850 RANGE")
    print("="*70)
    
    # Set random seed for reproducibility
    np.random.seed(123)
    
    # Load cleaned data
    df = pd.read_csv('cleaned_gig_worker_data.csv')
    print(f"\n✓ Loaded cleaned data: {len(df)} records")
    print(f"Original range: {df['credit_score'].min()} - {df['credit_score'].max()}")
    
    # Normalize features
    print(f"✓ Normalizing features...")
    df_final = normalize_features_final(df)
    
    # Calculate final credit scores with full range
    print(f"✓ Calculating credit scores with FULL 300-850 range...")
    df_final['credit_score'] = calculate_final_credit_scores(df_final)
    
    # Results
    print(f"\nFinal score statistics:")
    print(f"  Range: {df_final['credit_score'].min()} - {df_final['credit_score'].max()}")
    print(f"  Mean: {df_final['credit_score'].mean():.1f}")
    print(f"  Std: {df_final['credit_score'].std():.1f}")
    
    # Distribution
    print(f"\nFinal Score Distribution:")
    ranges = [
        (300, 500, "Very Poor"),
        (500, 600, "Poor"), 
        (600, 650, "Fair"),
        (650, 700, "Good"),
        (700, 750, "Very Good"),
        (750, 851, "Excellent")
    ]
    
    for min_score, max_score, category in ranges:
        count = len(df_final[(df_final['credit_score'] >= min_score) & (df_final['credit_score'] < max_score)])
        percentage = count / len(df_final) * 100
        print(f"  {min_score}-{max_score-1} ({category}): {count} ({percentage:.1f}%)")
    
    # Save dataset
    output_file = 'gig_worker_credit_dataset.csv'
    df_final.to_csv(output_file, index=False)
    print(f"\n✅ Saved final dataset to: {output_file}")
    
    # Show examples across all ranges
    print(f"\nSample profiles across score ranges:")
    print("="*70)
    
    for min_score, max_score, category in ranges:
        subset = df_final[(df_final['credit_score'] >= min_score) & (df_final['credit_score'] < max_score)]
        if len(subset) > 0:
            sample = subset.iloc[0]  # Take first example
            print(f"\n{category} (Score: {sample['credit_score']}):")
            print(f"  Payment History: {sample['payment_history']:.3f} | Utilization: {sample['credit_utilization']:.1f}%")
            print(f"  Savings: {sample['savings_ratio']:.3f} | Debt Ratio: {sample['debt_to_income_ratio']:.3f}")
            print(f"  Missed Payments: {int(sample['missed_payments_last_year'])} | Income Stability: {sample['income_stability_index']:.3f}")
    
    print(f"\n" + "="*70)
    print("✅ FINAL DATASET COMPLETE!")
    print("   Full 300-850 range with clear distinctions")
    print("   Better distribution across all credit categories")
    print("   Next: Retrain model with python credit_score_predictor.py")
    print("="*70)


if __name__ == "__main__":
    main()