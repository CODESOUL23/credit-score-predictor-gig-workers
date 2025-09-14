#!/usr/bin/env python3
"""
Data Cleaning Script for Gig Worker Credit Score Dataset
========================================================

This script simulates the data cleaning process for raw gig worker financial data.
It takes messy, real-world data with missing values, duplicates, outliers, and
inconsistent formats, then cleans it into a standardized format.

Usage: python data_cleaning.py
Output: cleaned_gig_worker_data.csv
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import random

warnings.filterwarnings('ignore')

class GigWorkerDataCleaner:
    def __init__(self):
        """Initialize the data cleaner with cleaning parameters."""
        self.cleaned_data = None
        
    def generate_raw_messy_data(self, n_samples=5000):
        """
        Generate realistic messy raw data that simulates real-world data collection issues.
        This represents what you might get from multiple data sources, APIs, and manual entry.
        """
        print("==> Generating realistic messy raw data...")
        
        # Set seed for reproducibility
        np.random.seed(42)
        random.seed(42)
        
        raw_data = []
        
        for i in range(n_samples + 500):  # Generate extra to account for duplicates we'll remove
            # Simulate different data quality issues
            missing_prob = 0.15  # 15% chance of missing data
            
            record = {}
            
            # Payment History (with various formats and missing data)
            if random.random() < missing_prob:
                record['payment_history'] = None
            else:
                # Sometimes as percentage, sometimes as decimal, sometimes with text
                val = np.random.uniform(0.3, 1.0)
                if random.random() < 0.3:
                    record['payment_history'] = f"{val*100:.1f}%"  # Percentage format
                elif random.random() < 0.1:
                    record['payment_history'] = "Good" if val > 0.8 else "Fair" if val > 0.6 else "Poor"
                else:
                    record['payment_history'] = val
            
            # Credit Utilization (sometimes over 100%, sometimes negative due to errors)
            if random.random() < missing_prob:
                record['credit_utilization'] = None
            else:
                val = np.random.uniform(5, 95)
                if random.random() < 0.05:  # 5% bad data
                    val = np.random.choice([-1, 150, 200])  # Clearly wrong values
                record['credit_utilization'] = val
            
            # Credit History Length (sometimes in months, sometimes inconsistent)
            if random.random() < missing_prob:
                record['length_credit_history'] = None
            else:
                val = np.random.randint(0, 20)
                if random.random() < 0.2:
                    record['length_credit_history'] = f"{val} years"  # Text format
                else:
                    record['length_credit_history'] = val
            
            # Credit Mix (decimal format, sometimes as percentage)
            if random.random() < missing_prob:
                record['credit_mix'] = None
            else:
                val = np.random.uniform(0.2, 0.9)
                if random.random() < 0.3:
                    record['credit_mix'] = f"{val*100:.0f}%"
                else:
                    record['credit_mix'] = val
            
            # New Credit Enquiries (sometimes negative due to data entry errors)
            if random.random() < missing_prob:
                record['new_credit_enquiries'] = None
            else:
                val = np.random.randint(0, 8)
                if random.random() < 0.05:
                    val = -1  # Bad data
                record['new_credit_enquiries'] = val
            
            # Income Stability Index
            if random.random() < missing_prob:
                record['income_stability_index'] = None
            else:
                record['income_stability_index'] = np.random.uniform(0.2, 1.0)
            
            # Digital Transaction Ratio
            if random.random() < missing_prob:
                record['digital_transaction_ratio'] = None
            else:
                record['digital_transaction_ratio'] = np.random.uniform(0.1, 1.0)
            
            # Savings Ratio
            if random.random() < missing_prob:
                record['savings_ratio'] = None
            else:
                val = np.random.uniform(0.0, 0.5)
                if random.random() < 0.1:
                    val = np.random.uniform(0.5, 2.0)  # Unrealistic high savings
                record['savings_ratio'] = val
            
            # Platform Loyalty (in months, sometimes with text)
            if random.random() < missing_prob:
                record['platform_loyalty_months'] = None
            else:
                val = np.random.randint(1, 60)
                if random.random() < 0.15:
                    record['platform_loyalty_months'] = f"{val}m"  # With 'm' suffix
                else:
                    record['platform_loyalty_months'] = val
            
            # Micro Loan Repayment Rate
            if random.random() < missing_prob:
                record['micro_loan_repayment'] = None
            else:
                val = np.random.uniform(0.4, 1.0)
                if random.random() < 0.2:
                    record['micro_loan_repayment'] = f"{val*100:.0f}%"
                else:
                    record['micro_loan_repayment'] = val
            
            # Age (sometimes with obvious errors)
            if random.random() < missing_prob:
                record['age'] = None
            else:
                val = np.random.randint(18, 65)
                if random.random() < 0.02:
                    val = np.random.choice([5, 150])  # Clearly wrong ages
                record['age'] = val
            
            # Work Experience
            if random.random() < missing_prob:
                record['work_experience'] = None
            else:
                record['work_experience'] = np.random.randint(0, 20)
            
            # Monthly Income (sometimes with currency symbols, sometimes unrealistic)
            if random.random() < missing_prob:
                record['monthly_income'] = None
            else:
                val = np.random.uniform(1000, 15000)
                if random.random() < 0.05:
                    val = np.random.choice([100, 100000])  # Unrealistic values
                if random.random() < 0.3:
                    record['monthly_income'] = f"${val:.2f}"  # With currency symbol
                else:
                    record['monthly_income'] = val
            
            # Debt to Income Ratio
            if random.random() < missing_prob:
                record['debt_to_income_ratio'] = None
            else:
                val = np.random.uniform(0.1, 0.8)
                if random.random() < 0.05:
                    val = np.random.uniform(1.0, 5.0)  # Unrealistic high ratios
                record['debt_to_income_ratio'] = val
            
            # Number of Platforms
            if random.random() < missing_prob:
                record['number_of_platforms'] = None
            else:
                record['number_of_platforms'] = np.random.randint(1, 8)
            
            # Average Weekly Hours
            if random.random() < missing_prob:
                record['avg_weekly_hours'] = None
            else:
                val = np.random.uniform(10, 80)
                if random.random() < 0.05:
                    val = np.random.choice([5, 120])  # Unrealistic hours
                record['avg_weekly_hours'] = val
            
            # Emergency Fund (in months)
            if random.random() < missing_prob:
                record['emergency_fund_months'] = None
            else:
                record['emergency_fund_months'] = np.random.uniform(0, 24)
            
            # Bank Account Age
            if random.random() < missing_prob:
                record['bank_account_age_months'] = None
            else:
                record['bank_account_age_months'] = np.random.randint(6, 120)
            
            # Credit Card Count
            if random.random() < missing_prob:
                record['credit_card_count'] = None
            else:
                record['credit_card_count'] = np.random.randint(0, 10)
            
            # Missed Payments
            if random.random() < missing_prob:
                record['missed_payments_last_year'] = None
            else:
                record['missed_payments_last_year'] = np.random.randint(0, 12)
            
            # Education Level
            if random.random() < missing_prob:
                record['education_level'] = None
            else:
                val = np.random.randint(1, 6)
                if random.random() < 0.3:
                    # Sometimes as text
                    education_map = {1: "High School", 2: "Some College", 3: "Bachelor's", 
                                   4: "Master's", 5: "PhD"}
                    record['education_level'] = education_map.get(val, val)
                else:
                    record['education_level'] = val
            
            # Financial Literacy Score
            if random.random() < missing_prob:
                record['financial_literacy_score'] = None
            else:
                record['financial_literacy_score'] = np.random.uniform(0.2, 1.0)
            
            # Credit Score (target variable)
            record['credit_score'] = np.random.randint(300, 850)
            
            raw_data.append(record)
        
        # Create DataFrame
        df = pd.DataFrame(raw_data)
        
        # Add some duplicate rows (simulate data collection errors)
        duplicates = df.sample(n=200, random_state=42)
        df = pd.concat([df, duplicates], ignore_index=True)
        
        print(f"==> Generated {len(df)} raw records with realistic data quality issues")
        return df
    
    def clean_data(self, raw_df):
        """
        Comprehensive data cleaning pipeline.
        """
        print("\n==> Starting data cleaning process...")
        
        df = raw_df.copy()
        initial_rows = len(df)
        
        # 1. Remove exact duplicates
        print("1. Removing duplicate records...")
        df = df.drop_duplicates()
        print(f"   Removed {initial_rows - len(df)} duplicate records")
        
        # 2. Handle missing values
        print("2. Handling missing values...")
        missing_before = df.isnull().sum().sum()
        
        # For numeric columns, use median imputation
        numeric_columns = ['payment_history', 'credit_utilization', 'income_stability_index',
                          'digital_transaction_ratio', 'savings_ratio', 'micro_loan_repayment',
                          'age', 'work_experience', 'monthly_income', 'debt_to_income_ratio',
                          'avg_weekly_hours', 'emergency_fund_months', 'financial_literacy_score']
        
        for col in numeric_columns:
            if col in df.columns:
                # First clean the column format, then impute
                df[col] = self._clean_numeric_column(df[col])
                if df[col].isnull().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
        
        # For categorical columns, use mode imputation
        categorical_columns = ['education_level', 'credit_mix', 'length_credit_history',
                              'new_credit_enquiries', 'number_of_platforms', 'credit_card_count',
                              'missed_payments_last_year', 'bank_account_age_months',
                              'platform_loyalty_months']
        
        for col in categorical_columns:
            if col in df.columns:
                df[col] = self._clean_categorical_column(df[col])
                if df[col].isnull().any():
                    mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 0
                    df[col] = df[col].fillna(mode_val)
        
        missing_after = df.isnull().sum().sum()
        print(f"   Handled {missing_before - missing_after} missing values")
        
        # 3. Remove outliers and unrealistic values
        print("3. Removing outliers and unrealistic values...")
        initial_outlier_rows = len(df)
        
        # Remove clearly wrong ages
        df = df[(df['age'] >= 18) & (df['age'] <= 65)]
        
        # Remove unrealistic income values
        df = df[(df['monthly_income'] >= 500) & (df['monthly_income'] <= 20000)]
        
        # Remove impossible credit utilization
        df = df[(df['credit_utilization'] >= 0) & (df['credit_utilization'] <= 100)]
        
        # Remove unrealistic debt ratios
        df = df[df['debt_to_income_ratio'] <= 1.0]
        
        # Remove unrealistic savings ratios
        df = df[df['savings_ratio'] <= 0.6]
        
        # Remove negative enquiries
        df = df[df['new_credit_enquiries'] >= 0]
        
        # Remove unrealistic working hours
        df = df[(df['avg_weekly_hours'] >= 5) & (df['avg_weekly_hours'] <= 80)]
        
        print(f"   Removed {initial_outlier_rows - len(df)} outlier records")
        
        # 4. Standardize data types
        print("4. Standardizing data types...")
        
        # Ensure integer columns are integers
        int_columns = ['age', 'work_experience', 'new_credit_enquiries', 'number_of_platforms',
                      'credit_card_count', 'missed_payments_last_year', 'education_level',
                      'bank_account_age_months', 'platform_loyalty_months', 'credit_score',
                      'length_credit_history']
        
        for col in int_columns:
            if col in df.columns:
                df[col] = df[col].round().astype(int)
        
        # Ensure float columns are floats with reasonable precision
        float_columns = ['payment_history', 'credit_utilization', 'credit_mix',
                        'income_stability_index', 'digital_transaction_ratio', 'savings_ratio',
                        'micro_loan_repayment', 'monthly_income', 'debt_to_income_ratio',
                        'avg_weekly_hours', 'emergency_fund_months', 'financial_literacy_score']
        
        for col in float_columns:
            if col in df.columns:
                df[col] = df[col].round(4).astype(float)
        
        # 5. Final data validation
        print("5. Final data validation...")
        
        # Ensure all required columns are present
        required_columns = ['payment_history', 'credit_utilization', 'length_credit_history',
                           'credit_mix', 'new_credit_enquiries', 'income_stability_index',
                           'digital_transaction_ratio', 'savings_ratio', 'platform_loyalty_months',
                           'micro_loan_repayment', 'age', 'work_experience', 'monthly_income',
                           'debt_to_income_ratio', 'number_of_platforms', 'avg_weekly_hours',
                           'emergency_fund_months', 'bank_account_age_months', 'credit_card_count',
                           'missed_payments_last_year', 'education_level', 'financial_literacy_score',
                           'credit_score']
        
        # Reorder columns to match target format
        df = df[required_columns]
        
        # Keep only the first n_samples rows
        df = df.head(5000).reset_index(drop=True)
        
        print(f"==> Data cleaning complete! Final dataset: {len(df)} records")
        return df
    
    def _clean_numeric_column(self, series):
        """Clean a numeric column by removing non-numeric characters and converting to float."""
        def clean_value(val):
            if pd.isna(val):
                return val
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                # Remove percentage signs, dollar signs, and other text
                val = val.replace('%', '').replace('$', '').replace(',', '')
                # Handle text values
                if val.lower() in ['good', 'excellent']:
                    return 0.9
                elif val.lower() in ['fair', 'average']:
                    return 0.7
                elif val.lower() in ['poor', 'bad']:
                    return 0.5
                else:
                    try:
                        return float(val)
                    except:
                        return np.nan
            return val
        
        return series.apply(clean_value)
    
    def _clean_categorical_column(self, series):
        """Clean a categorical column by standardizing formats."""
        def clean_value(val):
            if pd.isna(val):
                return val
            if isinstance(val, str):
                # Remove common suffixes
                val = val.replace('m', '').replace(' years', '').replace('%', '')
                # Handle education levels
                if 'high school' in val.lower():
                    return 1
                elif 'some college' in val.lower():
                    return 2
                elif 'bachelor' in val.lower():
                    return 3
                elif 'master' in val.lower():
                    return 4
                elif 'phd' in val.lower() or 'doctoral' in val.lower():
                    return 5
                else:
                    try:
                        return float(val)
                    except:
                        return np.nan
            return val
        
        return series.apply(clean_value)
    
    def save_cleaned_data(self, df, filename='cleaned_gig_worker_data.csv'):
        """Save the cleaned data to CSV file."""
        df.to_csv(filename, index=False)
        print(f"\n==> Cleaned data saved to: {filename}")
        print(f"==> Dataset shape: {df.shape}")
        print(f"==> Missing values: {df.isnull().sum().sum()}")
        
        # Display sample statistics
        print("\n==> Sample data statistics:")
        print(df.describe().round(2))

def main():
    """Main execution function."""
    print("=" * 60)
    print("*** GIG WORKER CREDIT DATA CLEANING PIPELINE ***")
    print("=" * 60)
    
    # Initialize cleaner
    cleaner = GigWorkerDataCleaner()
    
    # Generate raw messy data (simulating real-world data collection)
    raw_data = cleaner.generate_raw_messy_data(n_samples=5000)
    
    # Save raw data for reference
    raw_data.to_csv('raw_gig_worker_data.csv', index=False)
    print(f"==> Raw messy data saved to: raw_gig_worker_data.csv")
    
    # Clean the data
    cleaned_data = cleaner.clean_data(raw_data)
    
    # Save cleaned data
    cleaner.save_cleaned_data(cleaned_data, 'cleaned_gig_worker_data.csv')
    
    print("\n" + "=" * 60)
    print("*** DATA CLEANING COMPLETE! ***")
    print("==> Next step: Run data_normalization.py to normalize this data")
    print("=" * 60)

if __name__ == "__main__":
    main()