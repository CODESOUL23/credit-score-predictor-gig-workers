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
        color: #0d47a1;
    }
    .strength-box {
        background-color: #f0fff0;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #00cc66;
        margin: 0.5rem 0;
        color: #1B5E20;
    }
    .improvement-box {
        background-color: #fff8e1;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffa500;
        margin: 0.5rem 0;
        color: #E65100;
    }
    .high-priority-box {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #d32f2f;
        margin: 0.5rem 0;
        color: #c62828;
        box-shadow: 0 2px 4px rgba(211, 47, 47, 0.2);
        animation: pulse 2s infinite;
    }
    .insight-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 0.5rem 0;
        color: #2e7d32;
        font-style: italic;
    }
    @keyframes pulse {
        0% { box-shadow: 0 2px 4px rgba(211, 47, 47, 0.2); }
        50% { box-shadow: 0 4px 8px rgba(211, 47, 47, 0.4); }
        100% { box-shadow: 0 2px 4px rgba(211, 47, 47, 0.2); }
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
        import os
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'credit_score_model.pkl')
        
        predictor = CreditScorePredictor()
        predictor.load_model(model_path)
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
    
    # Load model
    predictor = load_model()
    if predictor is None:
        st.stop()
    
    # Sidebar for user input
    st.sidebar.header("📝 Enter Your Financial Information")
    st.sidebar.markdown("---")
    
    # Personal Information
    st.sidebar.markdown("### 👤 Personal Details")
    age = st.sidebar.slider("Age", 18, 65, 30)
    work_experience = st.sidebar.slider("Work Experience (years)", 0, 20, 5)
    education_level = st.sidebar.selectbox(
        "Education Level", 
        [1, 2, 3, 4, 5], 
        index=2,
        format_func=lambda x: {1: "High School", 2: "Some College", 3: "Bachelor's", 4: "Master's", 5: "PhD"}[x]
    )
    st.sidebar.markdown("---")
    
    # Financial History
    st.sidebar.markdown("### 💳 Financial History")
    payment_history = st.sidebar.slider("Payment History (0-1)", 0.0, 1.0, 0.85, 0.01, help="Higher is better - represents reliability in making payments")
    credit_utilization = st.sidebar.slider("Credit Utilization (%)", 0.0, 100.0, 45.0, help="Lower is better - percentage of available credit being used")
    length_credit_history = st.sidebar.slider("Length of Credit History (years)", 0, 20, 5)
    credit_mix = st.sidebar.slider("Credit Mix Diversity (0-1)", 0.0, 1.0, 0.7, 0.01, help="Higher is better - variety of credit types")
    new_credit_enquiries = st.sidebar.slider("New Credit Enquiries (last year)", 0, 10, 2, help="Lower is better - recent credit applications")
    credit_card_count = st.sidebar.slider("Number of Credit Cards", 0, 10, 2)
    missed_payments_last_year = st.sidebar.slider("Missed Payments (last year)", 0, 12, 1, help="Lower is better")
    st.sidebar.markdown("---")
    
    # Income and Gig Work
    st.sidebar.markdown("### 💰 Income & Gig Work")
    monthly_income = st.sidebar.number_input("Monthly Income ($)", 1000, 15000, 4500)
    income_stability_index = st.sidebar.slider("Income Stability Index (0-1)", 0.0, 1.0, 0.75, 0.01, help="Higher is better - consistency of income")
    number_of_platforms = st.sidebar.slider("Number of Gig Platforms", 1, 10, 3)
    avg_weekly_hours = st.sidebar.slider("Average Weekly Hours", 10.0, 80.0, 35.0)
    platform_loyalty_months = st.sidebar.slider("Platform Loyalty (months)", 1, 60, 24, help="Months with primary platform")
    st.sidebar.markdown("---")
    
    # Financial Health
    st.sidebar.markdown("### 🏦 Financial Health")
    savings_ratio = st.sidebar.slider("Savings Ratio (% of income)", 0.0, 0.5, 0.15, 0.01, help="Higher is better - portion of income saved monthly")
    debt_to_income_ratio = st.sidebar.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.35, 0.01, help="Lower is better - monthly debt payments vs income")
    emergency_fund_months = st.sidebar.slider("Emergency Fund (months of expenses)", 0.0, 24.0, 2.0, help="Higher is better - months of expenses covered")
    st.sidebar.markdown("---")
    
    # Additional Factors
    st.sidebar.markdown("### 📊 Additional Factors")
    digital_transaction_ratio = st.sidebar.slider("Digital Transaction Ratio (0-1)", 0.0, 1.0, 0.8, 0.01, help="Proportion of digital vs cash transactions")
    micro_loan_repayment = st.sidebar.slider("Micro Loan Repayment Rate (0-1)", 0.0, 1.0, 0.9, 0.01, help="Success rate in repaying small loans")
    bank_account_age_months = st.sidebar.slider("Bank Account Age (months)", 6, 120, 48)
    financial_literacy_score = st.sidebar.slider("Financial Literacy Score (0-1)", 0.0, 1.0, 0.6, 0.01, help="Self-assessed financial knowledge")
    
    # Predict button
    if st.sidebar.button("🔮 Predict My Credit Score", type="primary"):
        with st.spinner("🔍 Analyzing your financial profile..."):
            # Prepare user data
            user_data = {
                'payment_history': payment_history,
                'credit_utilization': credit_utilization,
                'length_credit_history': length_credit_history,
                'credit_mix': credit_mix,
                'new_credit_enquiries': new_credit_enquiries,
                'income_stability_index': income_stability_index,
                'digital_transaction_ratio': digital_transaction_ratio,
                'savings_ratio': savings_ratio,
                'platform_loyalty_months': platform_loyalty_months,
                'micro_loan_repayment': micro_loan_repayment,
                'age': age,
                'work_experience': work_experience,
                'monthly_income': monthly_income,
                'debt_to_income_ratio': debt_to_income_ratio,
                'number_of_platforms': number_of_platforms,
                'avg_weekly_hours': avg_weekly_hours,
                'emergency_fund_months': emergency_fund_months,
                'bank_account_age_months': bank_account_age_months,
                'credit_card_count': credit_card_count,
                'missed_payments_last_year': missed_payments_last_year,
                'education_level': education_level,
                'financial_literacy_score': financial_literacy_score
            }
            
            # Make prediction
            predicted_score, analysis = predictor.predict_credit_score(user_data)
            
            # Store results in session state
            st.session_state.predicted_score = predicted_score
            st.session_state.analysis = analysis
            st.session_state.user_data = user_data
            
            st.success("✅ Analysis complete! Check your results below.")
    
    # Display results if available
    if hasattr(st.session_state, 'predicted_score'):
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Credit Score Display
            st.plotly_chart(create_gauge_chart(st.session_state.predicted_score), use_container_width=True)
            
            # Score Category
            category = st.session_state.analysis['score_category']
            color = get_score_color(st.session_state.predicted_score)
            
            st.markdown(f"""
            <div class="score-category" style="background: linear-gradient(135deg, {color}20 0%, {color}40 100%); border: 3px solid {color}; color: {color};">
                {category} Credit Score
                <br><span style="font-size: 1.2rem; font-weight: normal;">Score: {st.session_state.predicted_score}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Analysis Results
            #st.markdown('<div class="analysis-container">', unsafe_allow_html=True)
            
            st.markdown('<div class="section-header">📊 Financial Analysis</div>', unsafe_allow_html=True)
            
            # Strengths
            if st.session_state.analysis['strengths']:
                st.markdown("### ✅ Your Financial Strengths")
                for strength in st.session_state.analysis['strengths']:
                    st.markdown(f"""
                    <div class="strength-box">
                        ✓ {strength}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Areas for Improvement
            if st.session_state.analysis['areas_for_improvement']:
                st.markdown("### ⚠️ Areas for Improvement")
                for area in st.session_state.analysis['areas_for_improvement']:
                    # Determine CSS class based on priority indicator
                    if area.startswith('🔴'):
                        css_class = "high-priority-box"
                        display_text = area[2:].strip()  # Remove emoji and space
                    elif area.startswith('🟡'):
                        css_class = "suggestion-box"
                        display_text = area[2:].strip()  # Remove emoji and space
                    elif area.startswith('🟢'):
                        css_class = "improvement-box"
                        display_text = area[2:].strip()  # Remove emoji and space
                    else:
                        css_class = "improvement-box"
                        display_text = area
                    
                    st.markdown(f"""
                    <div class="{css_class}">
                        • {display_text}
                    </div>
                    """, unsafe_allow_html=True)
            
            # ML Insights (if available) - Show as additional information
            if 'ml_insights' in st.session_state.analysis and st.session_state.analysis['ml_insights']:
                st.markdown("### 🧠 AI Insights")
                for insight in st.session_state.analysis['ml_insights']:
                    st.markdown(f"""
                    <div class="insight-box">
                        🧠 {insight}
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

def company_risk_tab():
    st.markdown('<h1 class="main-header">🏦 Company Risk Analysis</h1>', unsafe_allow_html=True)
    st.write("""
    This tool allows financial institutions to assess the risk of loan default for applicants. 
    Enter applicant data below to get a probability of default and an approval/rejection decision.
    """)
    
    # Company-side input fields
    with st.form("company_risk_form"):
        st.subheader("📋 Applicant Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            income = st.number_input("Annual Income ($)", min_value=1000, max_value=1000000, value=50000)
            loan_amount = st.number_input("Loan Amount ($)", min_value=500, max_value=500000, value=10000)
            employment_length = st.number_input("Employment Length (years)", min_value=0, max_value=50, value=5)
        
        with col2:
            credit_history_length = st.number_input("Credit History Length (years)", min_value=0, max_value=50, value=7)
            num_of_loans = st.number_input("Number of Existing Loans", min_value=0, max_value=20, value=2)
            num_of_defaults = st.number_input("Number of Past Defaults", min_value=0, max_value=10, value=0)
            
        submitted = st.form_submit_button("🔍 Predict Risk", type="primary")
    
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
        
        with st.spinner("🔍 Analyzing risk..."):
            try:
                model = load_company_model()
                result = predict_default_probability(applicant_data, model)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Probability of Default", 
                        f"{result['probability_of_default']:.2%}",
                        delta=f"{result['probability_of_default'] - 0.5:.2%} from threshold"
                    )
                
                with col2:
                    st.metric(
                        "Credit Score", 
                        f"{result['credit_score']:.0f}",
                        delta=f"{result['credit_score'] - 650:.0f} from fair credit"
                    )
                
                with col3:
                    if result['decision'] == 'Reject':
                        st.error(f"🚫 Decision: {result['decision']} (High Risk)")
                    else:
                        st.success(f"✅ Decision: {result['decision']} (Low Risk)")
                
                # Risk visualization
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = result['probability_of_default'] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Default Risk (%)", 'font': {'size': 20}},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "red" if result['probability_of_default'] >= 0.5 else "green"},
                        'steps': [
                            {'range': [0, 25], 'color': "lightgreen"},
                            {'range': [25, 50], 'color': "yellow"},
                            {'range': [50, 75], 'color': "orange"},
                            {'range': [75, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Applicant summary
                st.subheader("📊 Applicant Summary")
                summary_data = {
                    "Metric": ["Age", "Annual Income", "Loan Amount", "Employment Length", "Credit History", "Existing Loans", "Past Defaults"],
                    "Value": [
                        f"{age} years",
                        f"${income:,}",
                        f"${loan_amount:,}",
                        f"{employment_length} years",
                        f"{credit_history_length} years",
                        num_of_loans,
                        num_of_defaults
                    ]
                }
                st.table(pd.DataFrame(summary_data))
                
            except Exception as e:
                st.error(f"Error in risk analysis: {str(e)}")

def main():
    # Create tabs
    tab1, tab2 = st.tabs(["👤 Gig Worker Credit Score", "🏦 Company Risk Analysis"])
    
    with tab1:
        gig_worker_tab()
    
    with tab2:
        company_risk_tab()
    
    # Footer
    st.markdown("---")
    st.info("""
    **Disclaimer:** These predictions are based on machine learning analysis and are for educational purposes only. 
    Actual credit scores and risk assessments may vary and depend on additional factors not captured in these models.
    """)

if __name__ == "__main__":
    main()