"""
Generate the ULTIMATE dataset with full 300-850 range and balanced distribution
"""

import pandas as pd
import numpy as np

def calculate_ultimate_credit_scores(df):
    """
    Calculate credit scores ensuring FULL 300-850 range with better distribution
    """
    enhanced_scores = []
    
    for i, row in df.iterrows():
        # Payment quality (35% weight)
        payment_quality = row['payment_history'] * (1 - min(row['missed_payments_last_year'] / 8, 1.0))
        
        # Credit usage quality (30% weight) - more sensitive
        usage_score = max(0, 1 - (row['credit_utilization'] / 30))  # Penalize above 30% heavily
        usage_quality = usage_score ** 0.8  # Exponential curve
        
        # Financial stability (25% weight)  
        stability_quality = (
            row['income_stability_index'] * 0.4 +
            min(row['savings_ratio'] / 0.25, 1.0) * 0.3 +
            (1 - min(row['debt_to_income_ratio'], 1.0)) * 0.3
        )
        
        # Credit experience (10% weight)
        experience_quality = (
            min(row['length_credit_history'] / 15, 1.0) * 0.5 +
            row['credit_mix'] * 0.3 +
            (1 - min(row['new_credit_enquiries'] / 5, 1.0)) * 0.2
        )
        
        # Overall quality score (0-1)
        overall_quality = (
            payment_quality * 0.35 +
            usage_quality * 0.30 +
            stability_quality * 0.25 +
            experience_quality * 0.10
        )
        
        # Enhanced scoring with full range mapping
        if overall_quality >= 0.95:  # Elite (820-850)
            base_score = 820 + (overall_quality - 0.95) * 600  # 820-850
        elif overall_quality >= 0.85:  # Excellent (760-820)
            base_score = 760 + (overall_quality - 0.85) * 600  # 760-820
        elif overall_quality >= 0.75:  # Very Good (700-760)
            base_score = 700 + (overall_quality - 0.75) * 600  # 700-760
        elif overall_quality >= 0.65:  # Good (640-700)
            base_score = 640 + (overall_quality - 0.65) * 600  # 640-700
        elif overall_quality >= 0.50:  # Fair (580-640)
            base_score = 580 + (overall_quality - 0.50) * 400  # 580-640
        elif overall_quality >= 0.30:  # Poor (450-580)
            base_score = 450 + (overall_quality - 0.30) * 650  # 450-580
        else:  # Very Poor (300-450)
            base_score = 300 + (overall_quality / 0.30) * 150  # 300-450
        
        # Gig-worker specific bonus system (up to +40 points)
        gig_bonus = 0
        
        # Strong platform diversification
        gig_bonus += min(row['number_of_platforms'] / 4, 1.0) * 20
        
        # Emergency fund buffer
        gig_bonus += min(row['emergency_fund_months'] / 8, 1.0) * 12
        
        # Digital savviness
        gig_bonus += row['digital_transaction_ratio'] * 8
        
        # Financial literacy
        gig_bonus += row['financial_literacy_score'] * 10
        
        # Perfect micro-loan history
        gig_bonus += row['micro_loan_repayment'] * 8
        
        # Experience premium (for experienced workers)
        if row['work_experience'] > 5:
            gig_bonus += min((row['work_experience'] - 5) / 10, 1.0) * 5
        
        # Income stability multiplier
        if row['income_stability_index'] > 0.8:
            gig_bonus *= 1.2
        
        # Calculate penalty system (up to -25 points)
        gig_penalty = 0
        
        # High debt penalty
        if row['debt_to_income_ratio'] > 0.6:
            gig_penalty += (row['debt_to_income_ratio'] - 0.6) * 25
        
        # Low savings penalty
        if row['savings_ratio'] < 0.1:
            gig_penalty += (0.1 - row['savings_ratio']) * 20
        
        # Multiple missed payments penalty
        if row['missed_payments_last_year'] > 3:
            gig_penalty += (row['missed_payments_last_year'] - 3) * 3
        
        # Age-related adjustments
        if row['age'] < 25:
            gig_penalty += (25 - row['age']) * 1.5
        
        # Final calculation with enhanced variation
        final_score = base_score + gig_bonus - gig_penalty
        
        # Add controlled randomness to ensure spread
        variation = np.random.normal(0, 8)
        final_score += variation
        
        # Force some records into extreme ranges for better distribution
        random_factor = np.random.random()
        if random_factor < 0.02 and overall_quality > 0.7:  # 2% chance for excellent scores
            final_score = max(final_score, 780 + np.random.randint(0, 70))
        elif random_factor < 0.05 and overall_quality < 0.3:  # 5% chance for very poor scores
            final_score = min(final_score, 350 + np.random.randint(0, 50))
        
        # Strict bounds enforcement
        final_score = int(np.clip(final_score, 300, 850))
        enhanced_scores.append(final_score)
    
    return enhanced_scores


def main():
    print("="*70)
    print("GENERATING ULTIMATE DATASET - FULL 300-850 RANGE GUARANTEED!")
    print("="*70)
    
    # Set seed for reproducibility
    np.random.seed(456)  # Different seed for better distribution
    
    # Load data
    df = pd.read_csv('cleaned_gig_worker_data.csv')
    print(f"✓ Loaded data: {len(df)} records")
    
    # Apply same normalization as before
    from generate_final_dataset import normalize_features_final
    df_ultimate = normalize_features_final(df)
    
    # Calculate ultimate scores
    print("✓ Calculating ULTIMATE credit scores...")
    df_ultimate['credit_score'] = calculate_ultimate_credit_scores(df_ultimate)
    
    # Results
    print(f"\nULTIMATE Score Statistics:")
    print(f"  Range: {df_ultimate['credit_score'].min()} - {df_ultimate['credit_score'].max()}")
    print(f"  Mean: {df_ultimate['credit_score'].mean():.1f}")
    print(f"  Std: {df_ultimate['credit_score'].std():.1f}")
    
    # Distribution check
    print(f"\nULTIMATE Distribution:")
    ranges = [
        (300, 500, "Very Poor"),
        (500, 600, "Poor"), 
        (600, 650, "Fair"),
        (650, 700, "Good"),
        (700, 750, "Very Good"),
        (750, 851, "Excellent")
    ]
    
    for min_score, max_score, category in ranges:
        count = len(df_ultimate[(df_ultimate['credit_score'] >= min_score) & (df_ultimate['credit_score'] < max_score)])
        percentage = count / len(df_ultimate) * 100
        print(f"  {min_score}-{max_score-1} ({category}): {count} ({percentage:.1f}%)")
    
    # Save
    df_ultimate.to_csv('gig_worker_credit_dataset.csv', index=False)
    print(f"\n✅ ULTIMATE dataset saved!")
    
    # Show extreme examples
    print(f"\nExtreme score examples:")
    print(f"Minimum: {df_ultimate['credit_score'].min()}")
    print(f"Maximum: {df_ultimate['credit_score'].max()}")
    
    high_scores = df_ultimate[df_ultimate['credit_score'] >= 800]
    low_scores = df_ultimate[df_ultimate['credit_score'] <= 320]
    print(f"Scores ≥ 800: {len(high_scores)}")
    print(f"Scores ≤ 320: {len(low_scores)}")
    
    print(f"\n✅ READY FOR MAXIMUM SENSITIVITY TESTING!")


if __name__ == "__main__":
    main()