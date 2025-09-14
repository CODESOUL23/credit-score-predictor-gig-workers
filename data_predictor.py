import pandas as pd
import numpy as np

np.random.seed(42)
n = 5000  # Increased dataset size for better model training

# Generate more comprehensive features for gig workers
df = pd.DataFrame({
    "payment_history": np.round(np.random.uniform(0.6, 1.0, n), 2),
    "credit_utilization": np.round(np.random.uniform(10, 90, n), 2),
    "length_credit_history": np.random.randint(0, 15, n),
    "credit_mix": np.round(np.random.uniform(0.2, 1.0, n), 2),
    "new_credit_enquiries": np.random.poisson(2, n),
    "income_stability_index": np.round(np.random.uniform(0.3, 1.0, n), 2),
    "digital_transaction_ratio": np.round(np.random.uniform(0.2, 1.0, n), 2),
    "savings_ratio": np.round(np.random.uniform(0.05, 0.5, n), 2),
    "platform_loyalty_months": np.random.randint(1, 60, n),
    "micro_loan_repayment": np.round(np.random.uniform(0.5, 1.0, n), 2),
    "age": np.random.randint(18, 55, n),
    "work_experience": np.random.randint(0, 15, n),
    
    # Additional features for better analysis
    "monthly_income": np.round(np.random.uniform(1500, 8000, n), 2),
    "debt_to_income_ratio": np.round(np.random.uniform(0.1, 0.6, n), 2),
    "number_of_platforms": np.random.randint(1, 6, n),
    "avg_weekly_hours": np.round(np.random.uniform(10, 60, n), 1),
    "emergency_fund_months": np.round(np.random.uniform(0, 12, n), 1),
    "bank_account_age_months": np.random.randint(6, 120, n),
    "credit_card_count": np.random.randint(0, 8, n),
    "missed_payments_last_year": np.random.poisson(1, n),
    "education_level": np.random.choice([1, 2, 3, 4, 5], n),  # 1=High School, 5=Graduate
    "financial_literacy_score": np.round(np.random.uniform(0.3, 1.0, n), 2)
})

# Create more realistic credit score with multiple factor influences
df["credit_score"] = (
    300
    + df["payment_history"] * 200
    + (1 - df["credit_utilization"]/100) * 120
    + df["income_stability_index"] * 100
    + df["micro_loan_repayment"] * 80
    + (df["savings_ratio"] * 200)
    + (df["emergency_fund_months"] * 5)
    + (df["bank_account_age_months"] * 0.5)
    + (df["education_level"] * 10)
    + (df["financial_literacy_score"] * 50)
    - (df["debt_to_income_ratio"] * 100)
    - (df["missed_payments_last_year"] * 25)
    - (df["new_credit_enquiries"] * 10)
    + np.random.normal(0, 25, n)
).clip(300, 850).round().astype(int)

df.to_csv("gig_worker_credit_dataset.csv", index=False)
print(f"Dataset created with {n} records and {len(df.columns)} features")
print(df.head())
