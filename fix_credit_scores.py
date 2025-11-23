"""
Fix corrupted credit scores in the dataset by regenerating them using a proper formula.
The current scores are all 300-454, but they should be 300-850.
"""

import pandas as pd
import numpy as np

def calculate_proper_credit_score(row):
    """
    Calculate credit score (300-850) based on financial factors.
    Uses industry-standard credit score components with adjusted weights for gig workers.
    """
    
    # Base score
    base_score = 300
    max_additional = 550  # Total possible points above base (to reach 850)
    
    # Payment History (35% weight) - Most important factor
    payment_score = row['payment_history'] * 0.35 * max_additional
    
    # Credit Utilization (30% weight) - Inverse relationship (lower is better)
    # Convert from 0-100 to 0-1, then inverse
    util_score = (1 - min(row['credit_utilization'] / 100, 1.0)) * 0.30 * max_additional
    
    # Length of Credit History (15% weight)
    # Normalize to 0-1 based on max of 20 years
    history_score = min(row['length_credit_history'] / 20, 1.0) * 0.15 * max_additional
    
    # Credit Mix (10% weight)
    mix_score = row['credit_mix'] * 0.10 * max_additional
    
    # New Credit Inquiries (10% weight) - Inverse (fewer is better)
    # Max 10 inquiries, normalize and inverse
    inquiry_score = (1 - min(row['new_credit_enquiries'] / 10, 1.0)) * 0.10 * max_additional
    
    # Gig worker specific factors (bonus/penalty) - up to ±50 points
    gig_adjustment = 0
    
    # Income stability bonus (up to +20)
    gig_adjustment += row['income_stability_index'] * 20
    
    # Savings ratio bonus (up to +15)
    gig_adjustment += (row['savings_ratio'] / 0.3) * 15  # 30% savings = max bonus
    
    # Debt to income penalty (up to -20)
    gig_adjustment -= row['debt_to_income_ratio'] * 20
    
    # Emergency fund bonus (up to +10)
    gig_adjustment += min(row['emergency_fund_months'] / 6, 1.0) * 10
    
    # Missed payments penalty (up to -15)
    gig_adjustment -= min(row['missed_payments_last_year'] / 5, 1.0) * 15
    
    # Platform diversification bonus (up to +5)
    gig_adjustment += min(row['number_of_platforms'] / 5, 1.0) * 5
    
    # Financial literacy bonus (up to +10)
    gig_adjustment += row['financial_literacy_score'] * 10
    
    # Micro loan repayment bonus (up to +15)
    gig_adjustment += row['micro_loan_repayment'] * 15
    
    # Calculate final score
    total_score = (
        base_score +
        payment_score +
        util_score +
        history_score +
        mix_score +
        inquiry_score +
        gig_adjustment
    )
    
    # Ensure score is within valid range
    final_score = int(np.clip(total_score, 300, 850))
    
    return final_score


def main():
    print("="*70)
    print("FIXING CORRUPTED CREDIT SCORES IN DATASET")
    print("="*70)
    
    # Load dataset
    df = pd.read_csv('gig_worker_credit_dataset.csv')
    print(f"\n✓ Loaded dataset: {len(df)} records")
    
    # Check current scores
    print(f"\nCurrent Credit Scores:")
    print(f"  Min: {df['credit_score'].min()}")
    print(f"  Max: {df['credit_score'].max()}")
    print(f"  Mean: {df['credit_score'].mean():.1f}")
    
    # Backup original scores
    df['old_credit_score'] = df['credit_score']
    
    # Calculate proper scores
    print(f"\n✓ Recalculating credit scores using proper formula...")
    df['credit_score'] = df.apply(calculate_proper_credit_score, axis=1)
    
    # Display new distribution
    print(f"\nNew Credit Scores:")
    print(f"  Min: {df['credit_score'].min()}")
    print(f"  Max: {df['credit_score'].max()}")
    print(f"  Mean: {df['credit_score'].mean():.1f}")
    print(f"  Std: {df['credit_score'].std():.1f}")
    
    print(f"\nScore Distribution:")
    print(f"  300-500 (Very Poor): {len(df[df['credit_score'] < 500])} ({len(df[df['credit_score'] < 500])/len(df)*100:.1f}%)")
    print(f"  500-600 (Poor): {len(df[(df['credit_score'] >= 500) & (df['credit_score'] < 600)])} ({len(df[(df['credit_score'] >= 500) & (df['credit_score'] < 600)])/len(df)*100:.1f}%)")
    print(f"  600-650 (Fair): {len(df[(df['credit_score'] >= 600) & (df['credit_score'] < 650)])} ({len(df[(df['credit_score'] >= 600) & (df['credit_score'] < 650)])/len(df)*100:.1f}%)")
    print(f"  650-700 (Good): {len(df[(df['credit_score'] >= 650) & (df['credit_score'] < 700)])} ({len(df[(df['credit_score'] >= 650) & (df['credit_score'] < 700)])/len(df)*100:.1f}%)")
    print(f"  700-750 (Very Good): {len(df[(df['credit_score'] >= 700) & (df['credit_score'] < 750)])} ({len(df[(df['credit_score'] >= 700) & (df['credit_score'] < 750)])/len(df)*100:.1f}%)")
    print(f"  750+ (Excellent): {len(df[df['credit_score'] >= 750])} ({len(df[df['credit_score'] >= 750])/len(df)*100:.1f}%)")
    
    # Save corrected dataset (without old_credit_score column)
    df_final = df.drop('old_credit_score', axis=1)
    
    # Backup original
    print(f"\n✓ Backing up original dataset to 'gig_worker_credit_dataset_BACKUP.csv'")
    import shutil
    shutil.copy('gig_worker_credit_dataset.csv', 'gig_worker_credit_dataset_BACKUP.csv')
    
    # Save fixed dataset
    df_final.to_csv('gig_worker_credit_dataset.csv', index=False)
    print(f"✓ Saved corrected dataset to 'gig_worker_credit_dataset.csv'")
    
    # Show some examples
    print(f"\n\nSample transformations:")
    print("="*70)
    sample_df = df[['payment_history', 'credit_utilization', 'savings_ratio', 
                    'debt_to_income_ratio', 'old_credit_score', 'credit_score']].head(10)
    print(sample_df.to_string(index=False))
    
    print(f"\n" + "="*70)
    print("✓ DATASET FIX COMPLETE!")
    print("  Next step: Retrain the model using: python credit_score_predictor.py")
    print("="*70)


if __name__ == "__main__":
    main()
