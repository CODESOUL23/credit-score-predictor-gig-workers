# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
from credit_score_predictor import CreditScorePredictor
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict
import joblib
# Import company risk analysis logic
from company_risk_analysis import predict_default_probability, load_model as load_company_model

# Page configuration
st.set_page_config(
    page_title="Credit Score Predictor for Gig Workers",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #ddd;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .suggestion-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #0066cc;
        margin: 0.5rem 0;
        color: #0d47a1; /* Dark blue text for contrast */
    }
    .strength-box {
        background-color: #f0fff0;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #00cc66;
        margin: 0.5rem 0;
        color: #1B5E20; /* Dark green text for contrast */
    }
    .improvement-box {
        background-color: #fff8e1; /* Slightly different yellow */
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffa500;
        margin: 0.5rem 0;
        color: #E65100; /* Dark orange text for contrast */
    }
    .section-header {
        color: purple;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 2rem 0 1rem 0;
        padding: 1rem;
        border-radius: 8px;
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 5px solid #007bff;
    }
    .analysis-container {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    .score-category {
        font-size: 1.8rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained credit score model."""
    try:
        predictor = CreditScorePredictor()
        predictor.load_model('credit_score_model.pkl')
        return predictor
    except FileNotFoundError:
        st.error("Model file not found. Please train the model first by running credit_score_predictor.py")
        return None

def get_score_color(score):
    """Get color based on credit score."""
    if score >= 750:
        return "#22c55e"  # Green
    elif score >= 700:
        return "#3b82f6"  # Blue
    elif score >= 650:
        return "#f59e0b"  # Orange
    elif score >= 600:
        return "#ef4444"  # Red
    else:
        return "#dc2626"  # Dark Red

def create_gauge_chart(score):
    """Create a gauge chart for credit score."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Credit Score", 'font': {'size': 24}},
        delta = {'reference': 650},
        gauge = {
            'axis': {'range': [None, 850], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': get_score_color(score)},
            'steps': [
                {'range': [300, 600], 'color': "lightgray"},
                {'range': [600, 650], 'color': "gray"},
                {'range': [650, 700], 'color': "lightblue"},
                {'range': [700, 750], 'color': "blue"},
                {'range': [750, 850], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 700
            }
        }
    ))
    
    fig.update_layout(height=350, font={'color': "darkblue", 'family': "Arial"})
    return fig


def gig_worker_tab():
    st.markdown('<h1 class="main-header">🎯 Credit Score Predictor for Gig Workers</h1>', unsafe_allow_html=True)
    # ...existing code for gig worker UI (copy from above main)...
    # [PASTE the entire previous main() function body here, but remove the function definition and dedent]

def company_risk_tab():
    st.markdown('<h1 class="main-header">🏦 Company Risk Analysis</h1>', unsafe_allow_html=True)
    st.write("""
    This tool allows financial institutions to assess the risk of loan default for applicants. Enter applicant data below to get a probability of default and an approval/rejection decision.
    """)
    # Company-side input fields
    with st.form("company_risk_form"):
        st.subheader("Applicant Information")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        income = st.number_input("Annual Income ($)", min_value=1000, max_value=1000000, value=50000)
        loan_amount = st.number_input("Loan Amount ($)", min_value=500, max_value=500000, value=10000)
        employment_length = st.number_input("Employment Length (years)", min_value=0, max_value=50, value=5)
        credit_history_length = st.number_input("Credit History Length (years)", min_value=0, max_value=50, value=7)
        num_of_loans = st.number_input("Number of Existing Loans", min_value=0, max_value=20, value=2)
        num_of_defaults = st.number_input("Number of Past Defaults", min_value=0, max_value=10, value=0)
        submitted = st.form_submit_button("Predict Risk")
    if submitted:
        applicant_data = {
            'age': age,
            'income': income,
            'loan_amount': loan_amount,
            'employment_length': employment_length,
            'credit_history_length': credit_history_length,
            'num_of_loans': num_of_loans,
            'num_of_defaults': num_of_defaults
        }
        with st.spinner("Analyzing risk..."):
            model = load_company_model()
            result = predict_default_probability(applicant_data, model)
        st.success(f"Probability of Default: {result['probability_of_default']:.2%}")
        if result['decision'] == 'Reject':
            st.error(f"Decision: {result['decision']} (High Risk)")
        else:
            st.info(f"Decision: {result['decision']} (Low Risk)")

def main():
    tab1, tab2 = st.tabs(["👤 Gig Worker Credit Score", "🏦 Company Risk Analysis"])
    with tab1:
        gig_worker_tab()
    with tab2:
        company_risk_tab()

if __name__ == "__main__":
    main()
