"""
Clean the dataset to fix scores outside valid 300-850 range
"""
import pandas as pd
import numpy as np

def main():
    print("Cleaning dataset to fix invalid score ranges...")
    
    # Load dataset
    df = pd.read_csv('gig_worker_credit_dataset.csv')
    print(f"Original dataset: {len(df)} records")
    print(f"Score range: {df['credit_score'].min():.1f} - {df['credit_score'].max():.1f}")
    
    # Clean scores - clip to valid 300-850 range
    df_clean = df.copy()
    df_clean['credit_score'] = df_clean['credit_score'].clip(300, 850).round().astype(int)
    
    print(f"\nAfter cleaning:")
    print(f"Score range: {df_clean['credit_score'].min()} - {df_clean['credit_score'].max()}")
    print(f"Mean: {df_clean['credit_score'].mean():.1f}")
    print(f"Std: {df_clean['credit_score'].std():.1f}")
    
    # Check distribution
    ranges = [
        (300, 500, "Very Poor"),
        (500, 600, "Poor"), 
        (600, 650, "Fair"),
        (650, 700, "Good"),
        (700, 750, "Very Good"),
        (750, 851, "Excellent")
    ]
    
    print(f"\nFinal Distribution:")
    for min_score, max_score, category in ranges:
        count = len(df_clean[(df_clean['credit_score'] >= min_score) & (df_clean['credit_score'] < max_score)])
        percentage = count / len(df_clean) * 100
        print(f"  {min_score}-{max_score-1} ({category}): {count} ({percentage:.1f}%)")
    
    # Save cleaned dataset
    df_clean.to_csv('gig_worker_credit_dataset.csv', index=False)
    print(f"\n✅ Cleaned dataset saved!")
    print("Ready for model training.")

if __name__ == "__main__":
    main()