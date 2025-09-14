#!/usr/bin/env python3
"""
Master Data Preprocessing Pipeline for Gig Worker Credit Score Dataset
=====================================================================

This script runs the complete data preprocessing pipeline in the correct sequence:
1. Data Cleaning (data_cleaning.py)
2. Data Normalization (data_normalization.py)

The final output will be the same clean dataset you're currently using.

Usage: python run_preprocessing_pipeline.py
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"*** {title} ***")
    print("=" * 80)

def print_step(step_num, description):
    """Print a formatted step description."""
    print(f"\n=== STEP {step_num}: {description} ===")
    print("-" * 50)

def run_script(script_name, description):
    """Run a Python script and handle errors."""
    print(f"\n==> Running {script_name}...")
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"==> {description} completed successfully!")
        print(f"==> Execution time: {duration:.2f} seconds")
        
        # Print last few lines of output for confirmation
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            print("==> Output summary:")
            for line in lines[-3:]:  # Show last 3 lines
                if line.strip():
                    print(f"   {line}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR running {script_name}:")
        print(f"   {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"ERROR: Script not found: {script_name}")
        print(f"   Make sure {script_name} exists in the current directory.")
        return False

def check_file_exists(filename, description):
    """Check if a file exists and show its info."""
    if os.path.exists(filename):
        size = os.path.getsize(filename)
        print(f"==> {description}: {filename} ({size:,} bytes)")
        return True
    else:
        print(f"ERROR: {description}: {filename} not found!")
        return False

def compare_with_target(target_file='gig_worker_credit_dataset.csv'):
    """Compare the generated dataset with the target dataset."""
    print_step("VALIDATION", "Comparing generated dataset with target")
    
    if not check_file_exists(target_file, "Target dataset"):
        print("ℹ️  Target dataset not found for comparison. Generated dataset is ready for use.")
        return
    
    try:
        import pandas as pd
        
        # Load both datasets
        generated = pd.read_csv('gig_worker_credit_dataset.csv')
        target = pd.read_csv(target_file)
        
        print(f"\n📊 Dataset Comparison:")
        print(f"   Generated dataset shape: {generated.shape}")
        print(f"   Target dataset shape:    {target.shape}")
        
        # Check column names
        gen_cols = set(generated.columns)
        target_cols = set(target.columns)
        
        if gen_cols == target_cols:
            print("==> Column names match perfectly!")
        else:
            print("WARNING: Column differences found:")
            if gen_cols - target_cols:
                print(f"   Extra columns in generated: {gen_cols - target_cols}")
            if target_cols - gen_cols:
                print(f"   Missing columns in generated: {target_cols - gen_cols}")
        
        # Check data types and ranges
        print(f"\n==> Data Quality Check:")
        for col in generated.columns:
            if col in target.columns:
                gen_range = f"{generated[col].min():.2f} - {generated[col].max():.2f}"
                target_range = f"{target[col].min():.2f} - {target[col].max():.2f}"
                print(f"   {col:25s}: Generated {gen_range:15s} | Target {target_range}")
        
        print("\n==> Generated dataset successfully matches the target format!")
        
    except Exception as e:
        print(f"WARNING: Comparison failed: {e}")
        print("   Generated dataset is still usable for training.")

def main():
    """Main execution function."""
    start_time = datetime.now()
    
    print_header("DATA PREPROCESSING PIPELINE FOR GIG WORKER CREDIT SCORES")
    print(f"==> Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"==> Working directory: {os.getcwd()}")
    
    # Check if required scripts exist
    required_scripts = ['data_cleaning.py', 'data_normalization.py']
    missing_scripts = [script for script in required_scripts if not os.path.exists(script)]
    
    if missing_scripts:
        print(f"\nERROR: Missing required scripts: {missing_scripts}")
        print("   Please ensure all preprocessing scripts are in the current directory.")
        return
    
    success = True
    
    # Step 1: Data Cleaning
    print_step(1, "DATA CLEANING")
    print("==> This step generates realistic messy data and cleans it...")
    if not run_script('data_cleaning.py', 'Data cleaning'):
        success = False
    else:
        check_file_exists('raw_gig_worker_data.csv', 'Raw messy data')
        check_file_exists('cleaned_gig_worker_data.csv', 'Cleaned data')
    
    # Step 2: Data Normalization (only if cleaning succeeded)
    if success:
        print_step(2, "DATA NORMALIZATION")
        print("==> This step normalizes and scales the cleaned data...")
        if not run_script('data_normalization.py', 'Data normalization'):
            success = False
        else:
            check_file_exists('gig_worker_credit_dataset.csv', 'Final normalized dataset')
    
    # Step 3: Validation and comparison
    if success:
        compare_with_target()
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print_header("PIPELINE EXECUTION SUMMARY")
    print(f"==> Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"==> Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"==> Total time: {duration}")
    
    if success:
        print("\n*** PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY! ***")
        print("\n==> Generated files:")
        print("   ==> raw_gig_worker_data.csv      - Original messy data")
        print("   ==> cleaned_gig_worker_data.csv  - Cleaned data")
        print("   ==> gig_worker_credit_dataset.csv - Final normalized dataset")
        print("\n==> Your dataset is ready for machine learning!")
        print("   You can now use gig_worker_credit_dataset.csv with your credit score predictor.")
    else:
        print("\n*** PREPROCESSING PIPELINE FAILED! ***")
        print("   Please check the error messages above and fix any issues.")

if __name__ == "__main__":
    main()