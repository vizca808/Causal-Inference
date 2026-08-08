# Product Requirements Document (PRD)
## Causal Inference Analysis Pipeline

### 1. Project Overview
**Name:** Automated Causal Inference & Insights Pipeline
**Objective:** To build an automated data science pipeline that calculates the true causal impact of a treatment (e.g., job training) on an outcome (e.g., earnings) by accounting for confounding variables, moving beyond naive correlation.

### 2. Problem Statement
Traditional analytical methods (like naive mean differences or simple correlations) fail to identify causal relationships because they do not account for confounding variables (e.g., age, education) that influence both the treatment assignment and the outcome. This leads to biased decision-making (e.g., overestimating a marketing campaign's effectiveness).

### 3. Target Audience
- Data Scientists and Analysts seeking a starting template for causal inference.
- Business Stakeholders and Marketers who need data-driven insights to optimize targeting (e.g., finding out which demographic responds best to an intervention).

### 4. Core Features & Requirements

#### 4.1. Automated Data Processing
- **Requirement:** The system must automatically fetch the standard dataset (LaLonde) or generate a robust synthetic dummy dataset if the download fails.
- **Requirement:** Conduct Exploratory Data Analysis (EDA) and visualize the baseline outcome distributions between the treatment and control groups.

#### 4.2. Causal Modeling & Estimation
- **Requirement:** Implement a Causal Directed Acyclic Graph (DAG) to explicitly map structural assumptions.
- **Requirement:** Execute multiple causal inference methodologies to ensure robustness:
  - Propensity Score Matching (PSM)
  - Inverse Propensity Weighting (IPW)
  - Difference-in-Differences (DiD)
  - DoWhy Linear Regression
  - EconML Causal Forest

#### 4.3. Interpretability & Heterogeneous Effects
- **Requirement:** Use Machine Learning (Causal Forest) to calculate Individual Treatment Effects (ITE) and identify Heterogeneous Treatment Effects (HTE) to see which segments benefit the most.
- **Requirement:** Use SHAP (SHapley Additive exPlanations) to extract feature importance and visualize the top drivers of the causal effect.

#### 4.4. Automated Reporting
- **Requirement:** Output all plots as high-quality PNG images to a dedicated `plots/` directory.
- **Requirement:** Aggregate all results, business insights, and plots into a cohesive, non-technical PDF report.
- **Requirement:** Output the final PDF report to a dedicated `insight/` directory.

### 5. Technical Stack
- **Language:** Python 3.10+
- **Libraries:**
  - Causal Inference: `dowhy`, `econml`
  - Interpretability: `shap`
  - Modeling: `statsmodels`, `scikit-learn`
  - Data & Visualization: `pandas`, `numpy`, `matplotlib`, `seaborn`, `networkx`
  - Reporting: `fpdf2`

### 6. Future Enhancements (V2)
- Add Support for A/B Testing Data Integration.
- Add Refutation tests to explicitly test the robustness of the causal estimates against unobserved confounders.
- Build a Streamlit web dashboard for interactive causal DAG modification and real-time inference.
