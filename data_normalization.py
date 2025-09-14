#!/usr/bin/env python3
"""
Data Normalization Script for Gig Worker Credit Score Dataset
============================================================

This script takes the cleaned data from data_cleaning.py and applies various
normalization techniques to prepare it for machine learning. It handles
feature scaling, encoding, and final preprocessing steps.

Usage: python data_normalization.py
Input: cleaned_gig_worker_data.csv
Output: gig_worker_credit_dataset.csv (final normalized dataset)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings

warnings.filterwarnings('ignore')

class GigWorkerDataNormalizer:
    def __init__(self):
        """Initialize the data normalizer with normalization parameters."""
        self.scalers = {}
        self.encoders = {}
        
    def load_cleaned_data(self, filename='cleaned_gig_worker_data.csv'):
        """Load the cleaned data from CSV file."""
        try:
            df = pd.read_csv(filename)
            print(f"==> Loaded cleaned data: {filename}")
            print(f"==> Dataset shape: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"ERROR: {filename} not found. Please run data_cleaning.py first.")
            return None
    
    def normalize_data(self, df):
        """
        Apply comprehensive normalization to the cleaned dataset.
        """
        print("\n==> Starting data normalization process...")
        
        df_norm = df.copy()
        
        # 1. Feature Engineering & Transformation
        print("1. Applying feature engineering transformations...")
        
        # Convert credit utilization from percentage to ratio (0-1)
        df_norm['credit_utilization'] = df_norm['credit_utilization'] / 100.0
        
        # Ensure payment history is in 0-1 range
        if df_norm['payment_history'].max() > 1:
            df_norm['payment_history'] = df_norm['payment_history'] / 100.0
        
        # Ensure credit mix is in 0-1 range
        if df_norm['credit_mix'].max() > 1:
            df_norm['credit_mix'] = df_norm['credit_mix'] / 100.0
        
        # Ensure micro loan repayment is in 0-1 range
        if df_norm['micro_loan_repayment'].max() > 1:
            df_norm['micro_loan_repayment'] = df_norm['micro_loan_repayment'] / 100.0
        
        print("   --> Converted percentage features to ratios")
        
        # 2. Handle feature scaling for different feature types
        print("2. Applying feature-specific normalization...")
        
        # Features that should remain in their original scale (already 0-1)
        ratio_features = ['payment_history', 'credit_utilization', 'credit_mix', 
                         'income_stability_index', 'digital_transaction_ratio', 
                         'savings_ratio', 'micro_loan_repayment', 'financial_literacy_score']
        
        # Features that need Min-Max scaling (bounded ranges)
        bounded_features = ['debt_to_income_ratio']  # 0-1 range
        
        # Features that need robust scaling (may have outliers)
        robust_features = ['monthly_income', 'avg_weekly_hours', 'emergency_fund_months']
        
        # Count/discrete features (keep as integers but ensure reasonable ranges)
        count_features = ['age', 'work_experience', 'length_credit_history', 
                         'new_credit_enquiries', 'number_of_platforms', 
                         'platform_loyalty_months', 'bank_account_age_months',
                         'credit_card_count', 'missed_payments_last_year', 'education_level']
        
        # Apply Min-Max scaling to bounded features
        for feature in bounded_features:
            if feature in df_norm.columns:
                scaler = MinMaxScaler()
                df_norm[feature] = scaler.fit_transform(df_norm[[feature]]).flatten()
                self.scalers[feature] = scaler
        
        # Apply Robust scaling to features prone to outliers
        for feature in robust_features:
            if feature in df_norm.columns:
                # Use robust scaling but then normalize to reasonable range
                scaler = RobustScaler()
                scaled_values = scaler.fit_transform(df_norm[[feature]]).flatten()
                
                # Apply additional min-max to keep in reasonable range
                min_max_scaler = MinMaxScaler()
                df_norm[feature] = min_max_scaler.fit_transform(scaled_values.reshape(-1, 1)).flatten()
                
                # Scale back to reasonable ranges
                if feature == 'monthly_income':
                    df_norm[feature] = df_norm[feature] * 14000 + 1000  # 1000-15000 range
                elif feature == 'avg_weekly_hours':
                    df_norm[feature] = df_norm[feature] * 70 + 10  # 10-80 range
                elif feature == 'emergency_fund_months':
                    df_norm[feature] = df_norm[feature] * 24  # 0-24 range
                
                self.scalers[feature] = (scaler, min_max_scaler)
        
        print("   --> Applied robust scaling to income and time-based features")
        
        # 3. Normalize ratio features to ensure they're properly bounded
        print("3. Ensuring ratio features are properly bounded...")
        
        for feature in ratio_features:
            if feature in df_norm.columns:
                # Clip to [0, 1] range
                df_norm[feature] = np.clip(df_norm[feature], 0, 1)
        
        print("   --> Bounded ratio features to [0, 1] range")
        
        # 4. Apply intelligent feature transformations
        print("4. Applying intelligent feature transformations...")
        
        # Log transform for highly skewed features (if needed)
        skewed_features = ['monthly_income']
        for feature in skewed_features:
            if feature in df_norm.columns:
                # Check skewness
                skewness = df_norm[feature].skew()
                if abs(skewness) > 1:  # Highly skewed
                    # Apply log transformation and then normalize
                    df_norm[feature] = np.log1p(df_norm[feature])
                    scaler = MinMaxScaler()
                    df_norm[feature] = scaler.fit_transform(df_norm[[feature]]).flatten()
                    # Scale back to reasonable range
                    df_norm[feature] = df_norm[feature] * 14000 + 1000
                    print(f"   --> Applied log transformation to {feature}")
        
        # 5. Final data quality checks and adjustments
        print("5. Final data quality checks...")
        
        # Ensure all values are realistic
        df_norm = self._apply_realistic_constraints(df_norm)
        
        # Round values appropriately
        df_norm = self._apply_precision_rounding(df_norm)
        
        # 6. Generate synthetic variance to match target distribution
        print("6. Applying final distribution adjustments...")
        df_norm = self._match_target_distribution(df_norm)
        
        print("==> Data normalization complete!")
        return df_norm
    
    def _apply_realistic_constraints(self, df):
        """Apply realistic constraints to ensure data quality."""
        df_constrained = df.copy()
        
        # Ensure bounded features stay in bounds
        ratio_cols = ['payment_history', 'credit_utilization', 'credit_mix', 
                     'income_stability_index', 'digital_transaction_ratio', 
                     'savings_ratio', 'micro_loan_repayment', 'financial_literacy_score',
                     'debt_to_income_ratio']
        
        for col in ratio_cols:
            if col in df_constrained.columns:
                df_constrained[col] = np.clip(df_constrained[col], 0, 1)
        
        # Ensure count features are non-negative integers
        count_cols = ['new_credit_enquiries', 'credit_card_count', 'missed_payments_last_year']
        for col in count_cols:
            if col in df_constrained.columns:
                df_constrained[col] = np.clip(df_constrained[col], 0, None).round().astype(int)
        
        # Ensure age is realistic
        if 'age' in df_constrained.columns:
            df_constrained['age'] = np.clip(df_constrained['age'], 18, 65).round().astype(int)
        
        # Ensure work experience doesn't exceed age
        if 'work_experience' in df_constrained.columns and 'age' in df_constrained.columns:
            max_experience = df_constrained['age'] - 16  # Started working at 16
            df_constrained['work_experience'] = np.minimum(
                df_constrained['work_experience'], max_experience
            ).clip(0).round().astype(int)
        
        return df_constrained
    
    def _apply_precision_rounding(self, df):
        """Apply appropriate precision rounding to different feature types."""
        df_rounded = df.copy()
        
        # Round ratio features to 2 decimal places
        ratio_cols = ['payment_history', 'credit_utilization', 'credit_mix', 
                     'income_stability_index', 'digital_transaction_ratio', 
                     'savings_ratio', 'micro_loan_repayment', 'financial_literacy_score',
                     'debt_to_income_ratio']
        
        for col in ratio_cols:
            if col in df_rounded.columns:
                df_rounded[col] = df_rounded[col].round(2)
        
        # Round monetary values to 2 decimal places
        if 'monthly_income' in df_rounded.columns:
            df_rounded['monthly_income'] = df_rounded['monthly_income'].round(2)
        
        # Round time-based features to 1 decimal place
        time_cols = ['avg_weekly_hours', 'emergency_fund_months']
        for col in time_cols:
            if col in df_rounded.columns:
                df_rounded[col] = df_rounded[col].round(1)
        
        # Ensure integer columns are integers
        int_cols = ['age', 'work_experience', 'length_credit_history', 'new_credit_enquiries',
                   'number_of_platforms', 'platform_loyalty_months', 'bank_account_age_months',
                   'credit_card_count', 'missed_payments_last_year', 'education_level', 'credit_score']
        
        for col in int_cols:
            if col in df_rounded.columns:
                df_rounded[col] = df_rounded[col].round().astype(int)
        
        return df_rounded
    
    def _match_target_distribution(self, df):
        """Apply final adjustments to match the target distribution."""
        df_final = df.copy()
        
        # Add some controlled randomness to create realistic variance
        np.random.seed(42)
        
        # Slightly adjust continuous features to create more realistic distributions
        continuous_features = ['payment_history', 'credit_utilization', 'income_stability_index',
                              'digital_transaction_ratio', 'savings_ratio', 'micro_loan_repayment',
                              'financial_literacy_score', 'debt_to_income_ratio', 'monthly_income',
                              'avg_weekly_hours', 'emergency_fund_months']
        
        for feature in continuous_features:
            if feature in df_final.columns:
                # Add small random noise (±2% of the range)
                feature_range = df_final[feature].max() - df_final[feature].min()
                noise = np.random.normal(0, feature_range * 0.01, len(df_final))
                df_final[feature] = df_final[feature] + noise
                
                # Re-apply constraints
                if feature in ['payment_history', 'credit_utilization', 'credit_mix', 
                              'income_stability_index', 'digital_transaction_ratio', 
                              'savings_ratio', 'micro_loan_repayment', 'financial_literacy_score']:
                    df_final[feature] = np.clip(df_final[feature], 0, 1)
                elif feature == 'monthly_income':
                    df_final[feature] = np.clip(df_final[feature], 1000, 15000)
                elif feature == 'avg_weekly_hours':
                    df_final[feature] = np.clip(df_final[feature], 10, 80)
                elif feature == 'emergency_fund_months':
                    df_final[feature] = np.clip(df_final[feature], 0, 24)
        
        # Re-apply precision rounding after noise addition
        df_final = self._apply_precision_rounding(df_final)
        
        return df_final
    
    def generate_synthetic_credit_scores(self, df):
        """
        Generate realistic credit scores based on the features.
        This simulates how credit scores would be calculated from the features.
        """
        print("7. Generating realistic credit scores...")
        
        # Define feature weights based on real credit scoring models
        weights = {
            'payment_history': 35,          # Most important factor
            'credit_utilization': -25,      # Lower is better
            'length_credit_history': 15,    # Longer is better
            'credit_mix': 10,              # Diversity is good
            'new_credit_enquiries': -10,    # Too many is bad
            'income_stability_index': 8,    # Stability is good
            'savings_ratio': 6,            # Savings indicate responsibility
            'debt_to_income_ratio': -12,    # Lower is better
            'emergency_fund_months': 5,     # Financial cushion is good
            'missed_payments_last_year': -20, # Very negative impact
            'financial_literacy_score': 4,  # Knowledge helps
            'micro_loan_repayment': 8,      # Good repayment history
            'age': 2,                      # Slight bonus for age/stability
            'work_experience': 3,          # Experience indicates stability
        }
        
        # Calculate base score
        base_score = 300  # Minimum credit score
        score_range = 550  # Maximum additional points (300 + 550 = 850)
        
        calculated_scores = np.full(len(df), base_score, dtype=float)
        
        for feature, weight in weights.items():
            if feature in df.columns:
                if weight > 0:
                    # Positive contribution
                    if feature in ['age', 'work_experience', 'length_credit_history', 'emergency_fund_months']:
                        # Normalize these features to 0-1 scale first
                        normalized = (df[feature] - df[feature].min()) / (df[feature].max() - df[feature].min())
                        calculated_scores += normalized * weight
                    else:
                        calculated_scores += df[feature] * weight
                else:
                    # Negative contribution
                    if feature == 'credit_utilization':
                        # Higher utilization is worse
                        calculated_scores += (1 - df[feature]) * abs(weight)
                    elif feature in ['new_credit_enquiries', 'missed_payments_last_year']:
                        # Normalize and invert these features
                        if df[feature].max() > 0:
                            normalized = df[feature] / df[feature].max()
                            calculated_scores += (1 - normalized) * abs(weight)
                    elif feature == 'debt_to_income_ratio':
                        calculated_scores += (1 - df[feature]) * abs(weight)
        
        # Add some realistic randomness
        np.random.seed(42)
        random_factor = np.random.normal(0, 20, len(df))  # ±20 points random variation
        calculated_scores += random_factor
        
        # Ensure scores are within valid range
        calculated_scores = np.clip(calculated_scores, 300, 850)
        
        # Round to integers
        calculated_scores = calculated_scores.round().astype(int)
        
        return calculated_scores
    
    def save_normalized_data(self, df, filename='gig_worker_credit_dataset.csv'):
        """Save the normalized data to CSV file."""
        df.to_csv(filename, index=False)
        print(f"\n==> Normalized data saved to: {filename}")
        print(f"==> Final dataset shape: {df.shape}")
        
        # Display final statistics
        print(f"\n==> Final dataset statistics:")
        print(df.describe().round(2))
        
        # Show sample of final data
        print(f"\n==> Sample of final normalized data:")
        print(df.head(3).to_string())

def main():
    """Main execution function."""
    print("=" * 60)
    print("*** GIG WORKER CREDIT DATA NORMALIZATION PIPELINE ***")
    print("=" * 60)
    
    # Initialize normalizer
    normalizer = GigWorkerDataNormalizer()
    
    # Load cleaned data
    cleaned_data = normalizer.load_cleaned_data('cleaned_gig_worker_data.csv')
    if cleaned_data is None:
        return
    
    # Normalize the data
    normalized_data = normalizer.normalize_data(cleaned_data)
    
    # Generate realistic credit scores
    normalized_data['credit_score'] = normalizer.generate_synthetic_credit_scores(normalized_data)
    
    # Save the final normalized dataset
    normalizer.save_normalized_data(normalized_data, 'gig_worker_credit_dataset.csv')
    
    print("\n" + "=" * 60)
    print("*** DATA NORMALIZATION COMPLETE! ***")
    print("==> Final dataset ready: gig_worker_credit_dataset.csv")
    print("==> This dataset is now ready for machine learning!")
    print("=" * 60)

if __name__ == "__main__":
    main()