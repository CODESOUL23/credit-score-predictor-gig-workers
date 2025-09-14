# 🎯 Credit Score Predictor for Gig Workers

A comprehensive machine learning system that predicts credit scores for gig workers and provides personalized suggestions for improvement. This project includes multiple interfaces (CLI, Web App) and uses advanced ML techniques to analyze financial data.

## 🌟 Features

- **Accurate Predictions**: Uses Linear Regression with R² = 0.857 on 5000+ records
- **Comprehensive Analysis**: Analyzes 22+ financial features with engineered features
- **Personalized Suggestions**: Provides actionable recommendations for credit improvement
- **Multiple Interfaces**: Command-line tool and web application
- **Gig Worker Focused**: Tailored specifically for freelancers and gig economy workers
- **Real-time Analysis**: Instant predictions with detailed breakdowns

## 📊 Model Performance

- **Dataset**: 5,000 synthetic records with 23 features
- **Best Model**: Linear Regression
- **R² Score**: 0.857 (85.7% variance explained)
- **MAE**: 19.45 points
- **RMSE**: 24.28 points

## 🗂️ Project Structure

```
Credit score predictor for GIG workers/
├── data_predictor.py          # Dataset generation script
├── credit_score_predictor.py  # Main ML model and training
├── credit_score_cli.py        # Command-line interface
├── streamlit_app.py          # Web application interface
├── gig_worker_credit_dataset.csv  # Generated dataset
├── credit_score_model.pkl    # Trained model (created after training)
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib streamlit plotly
```

### 2. Generate Dataset and Train Model

```bash
python data_predictor.py          # Generate the dataset
python credit_score_predictor.py  # Train the model
```

### 3. Use the System

#### Option A: Command-Line Interface (Recommended for first-time users)
```bash
python credit_score_cli.py
```

#### Option B: Web Application
```bash
streamlit run streamlit_app.py
```

## 📝 Input Features

### Personal Information
- **Age**: 18-65 years
- **Work Experience**: 0-20 years
- **Education Level**: 1-5 (High School to PhD)

### Financial History
- **Payment History**: 0.0-1.0 (payment reliability)
- **Credit Utilization**: 0-100% (percentage of credit used)
- **Length of Credit History**: 0-20 years
- **Credit Mix**: 0.0-1.0 (diversity of credit types)
- **New Credit Enquiries**: 0-10 (recent credit applications)
- **Credit Card Count**: 0-10 cards
- **Missed Payments**: 0-12 in last year

### Income & Gig Work
- **Monthly Income**: $1,000-$15,000
- **Income Stability Index**: 0.0-1.0 (consistency of income)
- **Number of Platforms**: 1-10 gig platforms
- **Average Weekly Hours**: 10-80 hours
- **Platform Loyalty**: 1-60 months

### Financial Health
- **Savings Ratio**: 0.0-0.5 (percentage of income saved)
- **Debt-to-Income Ratio**: 0.0-1.0
- **Emergency Fund**: 0-24 months of expenses

### Additional Factors
- **Digital Transaction Ratio**: 0.0-1.0
- **Micro Loan Repayment**: 0.0-1.0
- **Bank Account Age**: 6-120 months
- **Financial Literacy Score**: 0.0-1.0

## 📊 Credit Score Ranges

| Score Range | Category | Description |
|-------------|----------|-------------|
| 750-850 | Excellent | Best rates and terms available |
| 700-749 | Good | Competitive rates and terms |
| 650-699 | Fair | Average rates, room for improvement |
| 600-649 | Poor | Higher rates, limited options |
| 300-599 | Very Poor | Very limited credit options |

## 💡 Personalized Suggestions

The system provides specific, actionable recommendations such as:

- **Payment History**: Set up automatic payments
- **Credit Utilization**: Reduce usage below 30%
- **Savings**: Increase to 10-20% of income
- **Emergency Fund**: Build to 3-6 months of expenses
- **Debt Management**: Strategies to reduce debt-to-income ratio
- **Income Stability**: Diversify income sources

## 🔧 Advanced Usage

### Training Custom Models

Modify `credit_score_predictor.py` to experiment with different models:

```python
# Add new models to the training pipeline
models = {
    'Random Forest': RandomForestRegressor(n_estimators=200),
    'XGBoost': XGBRegressor(),  # Add xgboost to requirements
    'Gradient Boosting': GradientBoostingRegressor()
}
```

### Feature Engineering

The system automatically creates engineered features:
- Payment-Income interaction
- Savings-Income interaction
- Credit utilization squared
- Age and experience categories
- Financial risk score

### Model Evaluation

```python
# View feature importance
predictor.plot_feature_importance(top_n=15)

# Cross-validation scores
from sklearn.model_selection import cross_val_score
scores = cross_val_score(predictor.model, X, y, cv=5)
print(f"CV Score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

## 📱 Web Application Features

The Streamlit web app includes:
- Interactive sliders for all input features
- Real-time credit score gauge
- Financial health radar chart
- Color-coded suggestions and recommendations
- Key metrics comparison with ideal values

## 🎯 Use Cases

### For Gig Workers
- Understand your creditworthiness
- Get specific improvement strategies
- Track progress over time
- Prepare for loan applications

### For Financial Advisors
- Assess client financial health
- Provide data-driven recommendations
- Demonstrate impact of financial changes
- Educational tool for financial literacy

### For Researchers
- Study gig economy financial patterns
- Analyze feature importance in credit scoring
- Benchmark against traditional credit models
- Develop gig worker-specific financial products

## 🔒 Privacy & Security

- All data processing happens locally
- No personal information is stored
- Model predictions are immediate and private
- Synthetic training data protects real user privacy

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add new features or improvements
4. Test thoroughly
5. Submit a pull request

### Potential Improvements
- Add more sophisticated models (Neural Networks, XGBoost)
- Implement real-time feature importance analysis
- Add data visualization for trends
- Create mobile app interface
- Integrate with actual credit bureau APIs (with permissions)

## 📚 Technical Details

### Data Processing Pipeline
1. **Data Generation**: Synthetic data with realistic distributions
2. **Feature Engineering**: Create interaction and derived features
3. **Preprocessing**: Handle missing values, scale features
4. **Model Training**: Compare multiple algorithms
5. **Validation**: Cross-validation and holdout testing

### Model Architecture
- **Base Features**: 22 original features
- **Engineered Features**: 5 additional derived features
- **Total Features**: 27 features for prediction
- **Target**: Credit scores 300-850 range

### Performance Metrics
- **R² Score**: Explained variance (higher is better)
- **MAE**: Mean Absolute Error in credit score points
- **RMSE**: Root Mean Square Error in credit score points

## 🐛 Troubleshooting

### Common Issues

1. **Model file not found**
   ```bash
   # Retrain the model
   python credit_score_predictor.py
   ```

2. **Missing dependencies**
   ```bash
   # Install all required packages
   pip install -r requirements.txt  # Create this file with all dependencies
   ```

3. **Web app not loading**
   ```bash
   # Check Streamlit installation
   streamlit --version
   # Try running on different port
   streamlit run streamlit_app.py --server.port 8502
   ```

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Created for gig workers to better understand and improve their financial health through data-driven insights.

---

**Note**: This system uses synthetic data for training and is designed for educational and informational purposes. For actual financial decisions, consult with qualified financial advisors and use official credit reports.