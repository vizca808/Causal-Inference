import streamlit as st
import pandas as pd
import os
from PIL import Image

# Setup page configuration
st.set_page_config(
    page_title="Causal Inference Dashboard",
    page_icon="🔬",
    layout="wide"
)

# Sidebar Navigation
st.sidebar.title("🔬 Causal Inference")
st.sidebar.write("Analyze the true causal impact of job training on earnings.")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["1. Executive Summary", "2. Data & Confounders", "3. Causal Analysis & HTE", "4. Feature Importance"])

# Directory where plots are stored
PLOTS_DIR = "plots"

def load_image(image_name):
    path = os.path.join(PLOTS_DIR, image_name)
    if os.path.exists(path):
        return Image.open(path)
    return None

if page == "1. Executive Summary":
    st.title("Executive Summary: Job Training Impact")
    st.markdown("""
    Welcome to the Causal Inference Web Dashboard! 
    
    This platform evaluates whether participating in a job training program genuinely *causes* an increase in earnings, moving beyond simple correlation.
    
    **Key Findings:**
    - The program has a statistically significant positive impact on earnings.
    - Advanced methods like Causal Forests and DoWhy consistently estimate the Average Treatment Effect (ATE) to be around **$1,600**.
    - The naive mean difference underestimates the true impact because it ignores selection bias (confounders).
    """)
    
    st.info("💡 **Navigation:** Use the sidebar on the left to explore the detailed analysis phases.")

    st.subheader("Comparison of ATE Across Methods")
    img = load_image("05_method_comparison.png")
    if img:
        st.image(img, use_column_width=True)
    else:
        st.warning("Method comparison plot not found. Run the main pipeline first.")

elif page == "2. Data & Confounders":
    st.title("Data Exploration & Structural Assumptions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Earnings Distribution")
        st.write("Baseline comparison of earnings between the treatment group (trained) and control group (untrained).")
        img_eda = load_image("01_eda_earnings_distribution.png")
        if img_eda:
            st.image(img_eda, use_column_width=True)
            
    with col2:
        st.subheader("Causal DAG (Directed Acyclic Graph)")
        st.write("This graph maps out our assumptions. Variables like Age and Education are *confounders* that affect both the likelihood of getting training and the final earnings.")
        img_dag = load_image("02_causal_dag.png")
        if img_dag:
            st.image(img_dag, use_column_width=True)

elif page == "3. Causal Analysis & HTE":
    st.title("Heterogeneous Treatment Effects (HTE)")
    
    st.markdown("""
    Not everyone benefits equally from the program. Using **EconML's Causal Forest**, we can estimate the *Individual Treatment Effect* (ITE) to find out who benefits the most.
    """)
    
    img_hte = load_image("03_hte_by_age.png")
    if img_hte:
        st.image(img_hte, width=800)
    else:
        st.warning("HTE plot not found.")
        
    st.success("Targeting marketing or program outreach to the highest-benefiting demographics maximizes ROI.")

elif page == "4. Feature Importance":
    st.title("What Drives the Treatment Effect?")
    
    st.markdown("""
    Using **SHAP (SHapley Additive exPlanations)**, we break down the Causal Forest model to understand which variables have the biggest influence on making the program successful for an individual.
    """)
    
    img_shap = load_image("04_shap_summary.png")
    if img_shap:
        st.image(img_shap, width=800)
    else:
        st.warning("SHAP summary plot not found.")

st.sidebar.markdown("---")
st.sidebar.info("Developed with ❤️ using Python & Streamlit.")
