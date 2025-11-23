"""
Generate a proper, normalized gig worker credit dataset from the cleaned data.
This script ensures all features are in correct ranges and credit scores are realistic.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def normalize_features(df):
    """
    Normalize features to proper ranges based on real-world credit data standards.
    """
    df_norm = df.copy()
    
    # Fix payment_history: should be 0-1 (percentage)
    # Current range: 0.30 - 99.90, normalize to 0-1
    df_norm['payment_history'] = df_norm['payment_history'] / 100
    df_norm['payment_history'] = df_norm['payment_history'].clip(0, 1)
    
    # Fix credit_utilization: should be 0-100 (percentage)
    # Current range: 5.0 - 95.0, this looks correct
    df_norm['credit_utilization'] = df_norm['credit_utilization'].clip(0, 100)
    
    # Fix credit_mix: should be 0-1
    # Current: some values > 1, normalize
    df_norm['credit_mix'] = df_norm['credit_mix'] / 100 if df_norm['credit_mix'].max() > 1 else df_norm['credit_mix']
    df_norm['credit_mix'] = df_norm['credit_mix'].clip(0, 1)
    
    # Fix income_stability_index: should be 0-1
    df_norm['income_stability_index'] = df_norm['income_stability_index'].clip(0, 1)
    
    # Fix digital_transaction_ratio: should be 0-1
    df_norm['digital_transaction_ratio'] = df_norm['digital_transaction_ratio'].clip(0, 1)
    
    # Fix savings_ratio: should be 0-1 (0-100% of income)
    df_norm['savings_ratio'] = df_norm['savings_ratio'].clip(0, 1)
    
    # Fix micro_loan_repayment: should be 0-1
    # Current: some values > 1, normalize
    df_norm['micro_loan_repayment'] = df_norm['micro_loan_repayment'] / 100 if df_norm['micro_loan_repayment'].max() > 1 else df_norm['micro_loan_repayment']
    df_norm['micro_loan_repayment'] = df_norm['micro_loan_repayment'].clip(0, 1)
    
    # Fix debt_to_income_ratio: should be 0-1 (0-100% of income)
    df_norm['debt_to_income_ratio'] = df_norm['debt_to_income_ratio'].clip(0, 1)
    
    # Fix financial_literacy_score: should be 0-1
    df_norm['financial_literacy_score'] = df_norm['financial_literacy_score'].clip(0, 1)
    
    # Fix age: reasonable range 18-65
    df_norm['age'] = df_norm['age'].clip(18, 65)
    
    # Fix work_experience: should not exceed age-18
    df_norm['work_experience'] = np.minimum(df_norm['work_experience'], df_norm['age'] - 18)
    df_norm['work_experience'] = df_norm['work_experience'].clip(0, 47)
    
    # Fix number_of_platforms: reasonable range 1-10
    df_norm['number_of_platforms'] = df_norm['number_of_platforms'].clip(1, 10)
    
    # Fix avg_weekly_hours: reasonable range 10-80
    df_norm['avg_weekly_hours'] = df_norm['avg_weekly_hours'].clip(10, 80)
    
    # Fix emergency_fund_months: reasonable range 0-24
    df_norm['emergency_fund_months'] = df_norm['emergency_fund_months'].clip(0, 24)
    
    # Fix credit_card_count: reasonable range 0-15
    df_norm['credit_card_count'] = df_norm['credit_card_count'].clip(0, 15)
    
    # Fix missed_payments_last_year: reasonable range 0-12
    df_norm['missed_payments_last_year'] = df_norm['missed_payments_last_year'].clip(0, 12)
    
    # Fix education_level: 1-5 scale
    df_norm['education_level'] = df_norm['education_level'].clip(1, 5)
    
    return df_norm


def recalculate_credit_scores(df):
    """
    Recalculate credit scores using industry-standard formula with proper weights.
    """
    base_score = 300
    max_additional = 550  # To reach maximum 850
    
    scores = []
    
    for _, row in df.iterrows():
        # Payment History (35% weight) - Most important
        payment_score = row['payment_history'] * 0.35 * max_additional
        
        # Credit Utilization (30% weight) - Lower is better
        util_score = (1 - min(row['credit_utilization'] / 100, 1.0)) * 0.30 * max_additional
        
        # Length of Credit History (15% weight)
        history_score = min(row['length_credit_history'] / 20, 1.0) * 0.15 * max_additional
        
        # Credit Mix (10% weight)
        mix_score = row['credit_mix'] * 0.10 * max_additional
        
        # New Credit Inquiries (10% weight) - Fewer is better
        inquiry_score = (1 - min(row['new_credit_enquiries'] / 10, 1.0)) * 0.10 * max_additional
        
        # Gig worker specific adjustments (±100 points total)
        gig_adjustment = 0
        
        # Income stability (+25 max)
        gig_adjustment += row['income_stability_index'] * 25
        
        # Savings ratio (+20 max)
        gig_adjustment += min(row['savings_ratio'] / 0.3, 1.0) * 20
        
        # Debt to income penalty (-30 max)
        gig_adjustment -= row['debt_to_income_ratio'] * 30
        
        # Emergency fund (+15 max)
        gig_adjustment += min(row['emergency_fund_months'] / 6, 1.0) * 15
        
        # Missed payments penalty (-25 max)
        gig_adjustment -= min(row['missed_payments_last_year'] / 5, 1.0) * 25
        
        # Platform diversification (+10 max)
        gig_adjustment += min(row['number_of_platforms'] / 5, 1.0) * 10
        
        # Financial literacy (+15 max)
        gig_adjustment += row['financial_literacy_score'] * 15
        
        # Micro loan repayment (+20 max)
        gig_adjustment += row['micro_loan_repayment'] * 20
        
        # Age experience bonus (+10 max for mature workers)
        if row['age'] >= 35:
            gig_adjustment += min((row['age'] - 35) / 20, 1.0) * 10
        
        # Calculate final score
        final_score = base_score + payment_score + util_score + history_score + mix_score + inquiry_score + gig_adjustment
        
        # Ensure within valid range and add some natural variation
        final_score = int(np.clip(final_score, 300, 850))
        scores.append(final_score)
    
    return scores


def main():
    print("="*70)
    print("GENERATING PROPER GIG WORKER CREDIT DATASET")
    print("="*70)
    
    # Load cleaned data
    df = pd.read_csv('cleaned_gig_worker_data.csv')
    print(f"\n✓ Loaded cleaned data: {len(df)} records")
    
    # Show current issues
    print(f"\nCurrent data issues:")
    print(f"  payment_history range: {df['payment_history'].min():.2f} - {df['payment_history'].max():.2f} (should be 0-1)")
    print(f"  credit_utilization range: {df['credit_utilization'].min():.1f} - {df['credit_utilization'].max():.1f}% (OK)")
    print(f"  credit_score range: {df['credit_score'].min()} - {df['credit_score'].max()} (OK)")
    
    # Normalize features
    print(f"\n✓ Normalizing features to proper ranges...")
    df_normalized = normalize_features(df)
    
    # Recalculate credit scores with proper formula
    print(f"✓ Recalculating credit scores using industry-standard formula...")
    df_normalized['credit_score'] = recalculate_credit_scores(df_normalized)
    
    # Show results
    print(f"\nFixed data ranges:")
    print(f"  payment_history: {df_normalized['payment_history'].min():.3f} - {df_normalized['payment_history'].max():.3f}")
    print(f"  credit_utilization: {df_normalized['credit_utilization'].min():.1f} - {df_normalized['credit_utilization'].max():.1f}%")
    print(f"  credit_score: {df_normalized['credit_score'].min()} - {df_normalized['credit_score'].max()}")
    print(f"  Mean credit score: {df_normalized['credit_score'].mean():.1f}")
    print(f"  Std credit score: {df_normalized['credit_score'].std():.1f}")
    
    # Show distribution
    print(f"\nCredit Score Distribution:")
    ranges = [
        (300, 500, "Very Poor"),
        (500, 600, "Poor"), 
        (600, 650, "Fair"),
        (650, 700, "Good"),
        (700, 750, "Very Good"),
        (750, 851, "Excellent")
    ]
    
    for min_score, max_score, category in ranges:
        count = len(df_normalized[(df_normalized['credit_score'] >= min_score) & (df_normalized['credit_score'] < max_score)])
        percentage = count / len(df_normalized) * 100
        print(f"  {min_score}-{max_score-1} ({category}): {count} ({percentage:.1f}%)")
    
    # Save the proper dataset
    output_file = 'gig_worker_credit_dataset.csv'
    df_normalized.to_csv(output_file, index=False)
    print(f"\n✅ Saved proper dataset to: {output_file}")
    
    # Show sample records
    print(f"\nSample records:")
    print("="*70)
    sample_cols = ['payment_history', 'credit_utilization', 'savings_ratio', 'debt_to_income_ratio', 'credit_score']
    print(df_normalized[sample_cols].head(10).to_string(index=False))
    
    print(f"\n" + "="*70)
    print("✅ DATASET GENERATION COMPLETE!")
    print("  Next step: Retrain model with: python credit_score_predictor.py")
    print("="*70)


if __name__ == "__main__":
    main()