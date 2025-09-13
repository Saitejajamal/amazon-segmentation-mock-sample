# ------------------------------
# Loan Default Prediction Case Study
# ------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# 1. Generate synthetic dataset
np.random.seed(42)
n_applicants = 300

data = pd.DataFrame({
    "income": np.random.normal(50000, 15000, n_applicants).clip(15000, 120000),
    "loan_amount": np.random.normal(20000, 10000, n_applicants).clip(5000, 80000),
    "credit_score": np.random.normal(650, 80, n_applicants).clip(300, 850),
    "employment_years": np.random.randint(0, 30, n_applicants),
    "age": np.random.randint(21, 65, n_applicants),
    "previous_loans": np.random.randint(0, 5, n_applicants)
})

# Define target: Default (1) / Repay (0)
# Heuristic rule: more likely default if loan high, credit score low, income low
data["default"] = (
    (data["loan_amount"] > 30000) &
    (data["credit_score"] < 600) &
    (data["income"] < 40000)
).astype(int)

print("Sample Data:")
print(data.head())

# 2. Features and Target
X = data.drop("default", axis=1)
y = data["default"]

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. PCA for dimensionality reduction (2 components for visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# 5. Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 6. Evaluation
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

report = classification_report(y_test, y_pred)
print("\nClassification Report:\n", report)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print(f"\nROC AUC Score: {roc_auc:.2f}")

# 7. Visualization - PCA Clusters
plt.figure(figsize=(6, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="bwr", alpha=0.7)
plt.title("PCA of Loan Applicants")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Default/Repay")
plt.show()

# 8. Visualization - ROC Curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="blue", label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.title("ROC Curve for Loan Default Prediction")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# 9. Extra: Probability Insights
p_credit_low = data[data["credit_score"] < 600]["default"].mean()
print(f"\nProbability of default given Credit Score < 600: {p_credit_low:.2f}")

p_high_loan = data[data["loan_amount"] > 40000]["default"].mean()
print(f"Probability of default given Loan Amount > 40,000: {p_high_loan:.2f}")
