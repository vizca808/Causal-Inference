import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import dowhy
from dowhy import CausalModel
from econml.dml import CausalForestDML
import shap
import warnings
import report_generator

# Suppress some warnings for cleaner output
warnings.filterwarnings('ignore')

# -------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------
OUTPUT_DIR = "plots"
INSIGHT_DIR = "insight"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INSIGHT_DIR, exist_ok=True)

print("="*50)
print("[ CAUSAL INFERENCE ANALYSIS PIPELINE ]")
print("="*50)

# -------------------------------------------------------------------
# Phase 1: Load Data & Exploratory Data Analysis (EDA)
# -------------------------------------------------------------------
print("\n[Phase 1] Loading LaLonde Dataset and performing EDA...")
try:
    url = "https://raw.githubusercontent.com/juba/pycausalimpact/master/tests/fixtures/lalonde.csv"
    data = pd.read_csv(url)
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])
    data = data.rename(columns={'treat': 'treatment', 'educ': 'education', 'hispan': 'hispanic'})
except Exception as e:
    print("Could not download LaLonde dataset. Generating synthetic dummy data...")
    np.random.seed(42)
    n = 700
    data = pd.DataFrame({
        'treatment': np.random.binomial(1, 0.3, n),
        'age': np.random.randint(20, 50, n),
        'education': np.random.randint(8, 16, n),
        'black': np.random.binomial(1, 0.2, n),
        'hispanic': np.random.binomial(1, 0.1, n),
        'married': np.random.binomial(1, 0.4, n),
        'nodegree': np.random.binomial(1, 0.3, n),
        're74': np.random.normal(5000, 2000, n).clip(0),
        're75': np.random.normal(5000, 2000, n).clip(0)
    })
    data['re78'] = data['re75'] + data['treatment'] * 1500 + np.random.normal(0, 1000, n)

# Print basic stats
print(f"Data shape: {data.shape}")
print("\nTreatment Group Counts:")
print(data['treatment'].value_counts())

# Save EDA Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(data[data['treatment']==1]['re78'], ax=axes[0], color='#2ecc71', bins=20)
axes[0].set_title('Treatment Group — Earnings (1978)')
axes[0].set_xlabel('Earnings ($)')

sns.histplot(data[data['treatment']==0]['re78'], ax=axes[1], color='#e74c3c', bins=20)
axes[1].set_title('Control Group — Earnings (1978)')
axes[1].set_xlabel('Earnings ($)')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_eda_earnings_distribution.png")
plt.close()
print(f"[OK] Saved EDA distribution plot to '{OUTPUT_DIR}/01_eda_earnings_distribution.png'")

# Covariates
covariates = ['age', 'education', 'black', 'hispanic', 'married', 'nodegree', 're74', 're75']

# -------------------------------------------------------------------
# Phase 2: Causal Graph (DAG)
# -------------------------------------------------------------------
print("\n[Phase 2] Generating Causal Directed Acyclic Graph (DAG)...")

G = nx.DiGraph()
G.add_edges_from([
    ('Age', 'Treatment'), ('Age', 'Earnings'),
    ('Education', 'Treatment'), ('Education', 'Earnings'),
    ('Prior_Earnings', 'Treatment'), ('Prior_Earnings', 'Earnings'),
    ('Race', 'Treatment'), ('Race', 'Earnings'),
    ('Treatment', 'Earnings')
])

pos = {
    'Age': (0, 2), 'Education': (2, 2),
    'Race': (0, 0), 'Prior_Earnings': (2, 0),
    'Treatment': (0.5, 1), 'Earnings': (1.5, 1)
}

plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_color='#3498db',
        node_size=3000, font_size=11, font_weight='bold',
        edge_color='#7f8c8d', arrows=True, arrowsize=20)
plt.title('Causal DAG — Training Program -> Earnings', fontsize=14)
plt.savefig(f"{OUTPUT_DIR}/02_causal_dag.png")
plt.close()
print(f"[OK] Saved Causal DAG to '{OUTPUT_DIR}/02_causal_dag.png'")

# -------------------------------------------------------------------
# Phase 3A: Propensity Score Matching (PSM)
# -------------------------------------------------------------------
print("\n[Phase 3A] Estimating Average Treatment Effect (ATE) with PSM...")

X = data[covariates]
T = data['treatment']

# Estimate propensity score
ps_model = LogisticRegression(max_iter=1000)
ps_model.fit(X, T)
data['propensity_score'] = ps_model.predict_proba(X)[:, 1]

# Match using Nearest Neighbors
treated = data[data['treatment'] == 1]
control = data[data['treatment'] == 0]

nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn.fit(control[['propensity_score']])
distances, indices = nn.kneighbors(treated[['propensity_score']])

matched_control = control.iloc[indices.flatten()]
ate_psm = treated['re78'].mean() - matched_control['re78'].mean()
print(f"   --> ATE (PSM): ${ate_psm:.2f}")

# -------------------------------------------------------------------
# Phase 3B: Inverse Propensity Weighting (IPW)
# -------------------------------------------------------------------
print("\n[Phase 3B] Estimating ATE with IPW...")

data['weight'] = np.where(
    data['treatment'] == 1,
    1 / data['propensity_score'],
    1 / (1 - data['propensity_score'])
)

ate_ipw = (
    np.average(data[data['treatment']==1]['re78'], weights=data[data['treatment']==1]['weight']) -
    np.average(data[data['treatment']==0]['re78'], weights=data[data['treatment']==0]['weight'])
)
print(f"   --> ATE (IPW): ${ate_ipw:.2f}")

# -------------------------------------------------------------------
# Phase 3C: Difference-in-Differences (DiD)
# -------------------------------------------------------------------
print("\n[Phase 3C] Estimating ATE with Difference-in-Differences (DiD)...")

# Difference between outcome (1978) and pre-treatment earnings (1975)
data['diff'] = data['re78'] - data['re75']

did_model = smf.ols('diff ~ treatment', data=data).fit()
ate_did = did_model.params['treatment']
print(f"   --> ATE (DiD): ${ate_did:.2f}")

# -------------------------------------------------------------------
# Phase 3D: DoWhy Framework
# -------------------------------------------------------------------
print("\n[Phase 3D] Running DoWhy Causal Pipeline...")

# Step 1: Define Model
model = CausalModel(
    data=data,
    treatment='treatment',
    outcome='re78',
    common_causes=covariates
)

# Step 2: Identify Effect
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

# Step 3: Estimate Effect
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression"
)
ate_dowhy = estimate.value
print(f"   --> ATE (DoWhy Linear Regression): ${ate_dowhy:.2f}")

# -------------------------------------------------------------------
# Phase 3E: Heterogeneous Treatment Effects (HTE) with EconML
# -------------------------------------------------------------------
print("\n[Phase 3E] Estimating Heterogeneous Treatment Effects (Causal Forest)...")

# We use CausalForestDML to see who benefits the most
causal_forest = CausalForestDML(
    model_y='auto',
    model_t='auto',
    n_estimators=100,
    random_state=42
)

# For causal_forest, features X must be array-like
X_features = data[covariates]
y = data['re78'].values
t = data['treatment'].values

causal_forest.fit(Y=y, T=t, X=X_features)

# Calculate Individual Treatment Effect (ITE)
data['ite'] = causal_forest.effect(X_features)
ate_econml = data['ite'].mean()
print(f"   --> ATE (EconML Causal Forest): ${ate_econml:.2f}")

print("\nTop 5 individuals who benefited the most from the program:")
print(data.nlargest(5, 'ite')[['age', 'education', 're75', 'ite']])

# Visualizing HTE by Age
plt.figure(figsize=(10, 5))
scatter = plt.scatter(data['age'], data['ite'], alpha=0.6, c=data['ite'], cmap='coolwarm')
plt.colorbar(scatter, label='Treatment Effect ($)')
plt.xlabel('Age')
plt.ylabel('Individual Treatment Effect ($)')
plt.title('Heterogeneous Treatment Effects by Age')
plt.axhline(y=0, color='red', linestyle='--')
plt.savefig(f"{OUTPUT_DIR}/03_hte_by_age.png")
plt.close()
print(f"[OK] Saved HTE plot to '{OUTPUT_DIR}/03_hte_by_age.png'")

# -------------------------------------------------------------------
# Phase 4: Model Interpretability with SHAP
# -------------------------------------------------------------------
print("\n[Phase 4] Generating SHAP values for Interpretability...")

try:
    # EconML provides a built-in wrapper for SHAP
    shap_values = causal_forest.shap_values(X_features)
    # For a single outcome and single treatment, it returns a dict of dicts
    # e.g., {'Y0': {'T0': array}}
    y_key = list(shap_values.keys())[0]
    t_key = list(shap_values[y_key].keys())[0]
    shaps = shap_values[y_key][t_key]
    
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shaps, X_features, show=False)
    plt.savefig(f"{OUTPUT_DIR}/04_shap_summary.png", bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved SHAP summary plot to '{OUTPUT_DIR}/04_shap_summary.png'")
except Exception as e:
    print(f"Warning: SHAP calculation failed. Error: {e}")

# -------------------------------------------------------------------
# Phase 5: Final Comparison
# -------------------------------------------------------------------
print("\n[Phase 5] Summary of Average Treatment Effects (ATE)")

naive_ate = data[data['treatment']==1]['re78'].mean() - data[data['treatment']==0]['re78'].mean()

results = pd.DataFrame({
    'Method': ['Naive Mean Diff', 'PSM', 'IPW', 'DiD', 'DoWhy (LR)', 'EconML (Forest)'],
    'ATE ($)': [naive_ate, ate_psm, ate_ipw, ate_did, ate_dowhy, ate_econml]
})
print("\n" + results.to_string(index=False) + "\n")

# Plot the comparison
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#bdc3c7', '#2ecc71', '#3498db', '#e67e22', '#9b59b6', '#e74c3c']
bars = ax.bar(results['Method'], results['ATE ($)'], color=colors)
ax.set_ylabel('Average Treatment Effect ($)')
ax.set_title('Comparison of ATE across Methods')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

for bar, val in zip(bars, results['ATE ($)']):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
            f'${val:,.0f}', ha='center', va='bottom', fontweight='bold')

plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_method_comparison.png")
plt.close()
print(f"[OK] Saved Method Comparison plot to '{OUTPUT_DIR}/05_method_comparison.png'")

# -------------------------------------------------------------------
# Phase 6: PDF Report Generation
# -------------------------------------------------------------------
print("\n[Phase 6] Generating PDF Report with Business Insights...")
ate_dict = {
    'Naive Mean Diff': naive_ate,
    'Propensity Score Matching (PSM)': ate_psm,
    'Inverse Propensity Weighting (IPW)': ate_ipw,
    'Difference-in-Differences (DiD)': ate_did,
    'DoWhy (Linear Regression)': ate_dowhy,
    'EconML (Causal Forest)': ate_econml
}
pdf_path = report_generator.generate_pdf_report(ate_dict, OUTPUT_DIR, INSIGHT_DIR)
print(f"[OK] Saved PDF Report to '{pdf_path}'")

print("\n" + "="*50)
print("[ ANALYSIS COMPLETE! ]")
print(f"All plots are in '{OUTPUT_DIR}' and the PDF report is in '{INSIGHT_DIR}'.")
print("="*50)
