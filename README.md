# 🔬 Causal Inference Analysis Pipeline

An end-to-end Machine Learning pipeline for Causal Inference analysis using the classic LaLonde dataset (evaluating the impact of job training programs on earnings).

Unlike standard correlation or predictive models, this project uses causal inference techniques to account for confounders and estimate the true **Average Treatment Effect (ATE)** and **Heterogeneous Treatment Effects (HTE)**.

## 🚀 Features

- **Automated Data Processing & EDA**: Generates distribution comparisons between treatment and control groups.
- **Causal DAG Visualization**: Maps out the structural assumptions of the causal model.
- **Multiple Estimation Methods**: Compares results across 5 different causal methods:
  - Propensity Score Matching (PSM)
  - Inverse Propensity Weighting (IPW)
  - Difference-in-Differences (DiD)
  - DoWhy Linear Regression
  - EconML Causal Forest
- **Model Interpretability (SHAP)**: Extracts feature importance to understand which variables drive the treatment effect.
- **Automated PDF Reporting**: Generates an executive summary PDF with business insights and embedded plots.

## 📁 Directory Structure

```text
causal_inference_analysis/
├── main.py                 # The main execution pipeline
├── report_generator.py     # Module for automated PDF generation
├── requirements.txt        # Python dependencies
├── setup_and_run.ps1       # Setup script for Windows PowerShell
├── plots/                  # Generated plots and visualizations
└── insight/                # Generated PDF executive report
```

## 🛠️ Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vizca808/Causal-Inference.git
   cd Causal-Inference
   ```

2. **Run the automated setup (Windows):**
   ```powershell
   .\setup_and_run.ps1
   ```

   *Alternatively, set it up manually:*
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: .\venv\Scripts\activate
   pip install -r requirements.txt
   python main.py
   ```

3. **View the Results:**
   Check the `plots/` folder for generated visual graphs and the `insight/` folder for the comprehensive PDF report.

## 📊 Methods Overview

- **Propensity Score Matching (PSM)**: Matches treated individuals with similar control individuals based on their probability of receiving treatment.
- **EconML Causal Forest**: A powerful machine learning approach that calculates individualized treatment effects (ITE) to find out exactly *who* benefits the most from the program.
- **DoWhy Framework**: Microsoft's framework that explicitly identifies causal estimands from a DAG before estimation, followed by refutation tests to check robustness.

## 📝 License

This project is open-source and available under the MIT License.
