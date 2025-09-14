# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from credit_score_predictor import CreditScorePredictor
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict
import joblib

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

def main():
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
            st.markdown('<div class="analysis-container">', unsafe_allow_html=True)
            
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
                    st.markdown(f"""
                    <div class="improvement-box">
                        • {area}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Specific Suggestions
            if st.session_state.analysis['specific_suggestions']:
                st.markdown("### 💡 Personalized Suggestions")
                for suggestion in st.session_state.analysis['specific_suggestions']:
                    st.markdown(f"""
                    <div class="suggestion-box">
                        💡 {suggestion}
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Financial Overview Chart
        st.markdown('<div class="section-header">📈 Your Financial Profile Overview</div>', unsafe_allow_html=True)
        
        # Create radar chart for financial metrics
        categories = ['Payment History', 'Credit Utilization', 'Savings Ratio', 
                     'Income Stability', 'Financial Literacy', 'Emergency Fund']
        
        # Normalize values for radar chart (0-100 scale)
        values = [
            st.session_state.user_data['payment_history'] * 100,
            (100 - st.session_state.user_data['credit_utilization']),  # Invert for better visualization
            st.session_state.user_data['savings_ratio'] * 200,  # Scale up
            st.session_state.user_data['income_stability_index'] * 100,
            st.session_state.user_data['financial_literacy_score'] * 100,
            min(st.session_state.user_data['emergency_fund_months'] * 16.67, 100)  # Scale to 100
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Your Profile',
            line=dict(color='#2196F3', width=3),
            fillcolor='rgba(33, 150, 243, 0.25)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(size=12),
                    gridcolor="lightgray"
                ),
                angularaxis=dict(
                    tickfont=dict(size=12)
                )),
            showlegend=True,
            title=dict(
                text="Financial Health Radar Chart",
                font=dict(size=18),
                x=0.5
            ),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Key Metrics Summary
        st.markdown('<div class="section-header">📋 Key Financial Metrics Comparison</div>', unsafe_allow_html=True)
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            util_delta = 30 - st.session_state.user_data['credit_utilization']
            st.metric(
                "Credit Utilization",
                f"{st.session_state.user_data['credit_utilization']:.1f}%",
                delta=f"{util_delta:.1f}% from ideal (30%)",
                delta_color="inverse"
            )
        
        with metric_col2:
            debt_delta = 0.3 - st.session_state.user_data['debt_to_income_ratio']
            st.metric(
                "Debt-to-Income",
                f"{st.session_state.user_data['debt_to_income_ratio']:.1%}",
                delta=f"{debt_delta:.1%} from ideal (30%)",
                delta_color="inverse"
            )
        
        with metric_col3:
            savings_delta = st.session_state.user_data['savings_ratio'] - 0.2
            st.metric(
                "Savings Rate",
                f"{st.session_state.user_data['savings_ratio']:.1%}",
                delta=f"{savings_delta:.1%} from ideal (20%)"
            )
        
        with metric_col4:
            emergency_delta = st.session_state.user_data['emergency_fund_months'] - 6
            st.metric(
                "Emergency Fund",
                f"{st.session_state.user_data['emergency_fund_months']:.1f} months",
                delta=f"{emergency_delta:.1f} from ideal (6 months)"
            )
    
        # Add footer with disclaimer
        st.markdown("---")
        st.info("""
        **Disclaimer:** This credit score prediction is based on machine learning analysis and is for educational purposes only. 
        Actual credit scores may vary and depend on additional factors not captured in this model.
        
        **Model Performance:** R² = 0.857 (85.7% accuracy) trained on 5,000 gig worker profiles.
        """)

if __name__ == "__main__":
    main()
