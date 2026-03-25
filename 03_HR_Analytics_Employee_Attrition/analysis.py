#!/usr/bin/env python3
"""
Project 3: HR Analytics - Predict Employee Attrition
Uses: Pandas, Seaborn, Scikit-learn, Matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# 1. GENERATE SAMPLE HR DATA (replace with IBM HR dataset)
# ------------------------------------------------------------------
np.random.seed(42)
n = 1500
departments = ['Sales', 'R&D', 'HR', 'Marketing', 'IT']
salary_bands = ['Low', 'Medium', 'High']

df = pd.DataFrame({
    'EmployeeID': range(1, n+1),
    'Age': np.random.randint(22, 60, n),
    'Department': np.random.choice(departments, n, p=[0.3, 0.25, 0.1, 0.15, 0.2]),
    'MonthlyIncome': np.random.normal(6000, 2500, n).clip(2000),
    'YearsAtCompany': np.random.exponential(5, n).astype(int).clip(0, 30),
    'OverTime': np.random.choice(['Yes', 'No'], n, p=[0.3, 0.7]),
    'JobSatisfaction': np.random.randint(1, 5, n),
    'WorkLifeBalance': np.random.randint(1, 5, n),
    'NumCompaniesWorked': np.random.randint(0, 10, n),
    'DistanceFromHome': np.random.randint(1, 30, n),
    'PercentSalaryHike': np.random.randint(10, 25, n),
    'YearsSinceLastPromotion': np.random.randint(0, 10, n),
})

# Create attrition (correlated)
attrition_prob = (
    0.1
    + (df['OverTime'] == 'Yes').astype(float) * 0.2
    - df['MonthlyIncome'] / 50000
    - df['JobSatisfaction'] * 0.05
    + df['DistanceFromHome'] * 0.005
    - df['YearsAtCompany'] * 0.01
).clip(0.05, 0.7)
df['Attrition'] = np.random.binomial(1, attrition_prob)
df['SalaryBand'] = pd.cut(df['MonthlyIncome'], bins=3, labels=salary_bands)

print(f"Dataset: {df.shape}")
print(f"Attrition Rate: {df['Attrition'].mean():.1%}")

# ------------------------------------------------------------------
# 2. EDA
# ------------------------------------------------------------------
print("\n--- Department-wise Attrition ---")
dept_attr = df.groupby('Department')['Attrition'].mean().sort_values(ascending=False)
print(dept_attr.round(3))

print("\n--- Salary Band Attrition ---")
sal_attr = df.groupby('SalaryBand')['Attrition'].mean()
print(sal_attr.round(3))

# ------------------------------------------------------------------
# 3. PREPROCESSING
# ------------------------------------------------------------------
le = LabelEncoder()
df_model = df.copy()
for col in ['Department', 'OverTime', 'SalaryBand']:
    df_model[col] = le.fit_transform(df_model[col])

features = ['Age', 'Department', 'MonthlyIncome', 'YearsAtCompany', 'OverTime',
            'JobSatisfaction', 'WorkLifeBalance', 'NumCompaniesWorked',
            'DistanceFromHome', 'PercentSalaryHike', 'YearsSinceLastPromotion']
X = df_model[features]
y = df_model['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# ------------------------------------------------------------------
# 4. MODEL TRAINING
# ------------------------------------------------------------------
# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_sc, y_train)
lr_pred = lr.predict(X_test_sc)

# Decision Tree
dt = DecisionTreeClassifier(max_depth=6, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

print("\n" + "="*50)
print("MODEL RESULTS")
print("="*50)
for name, pred in [('Logistic Regression', lr_pred), ('Decision Tree', dt_pred)]:
    print(f"\n{name}:")
    print(f"  Accuracy: {accuracy_score(y_test, pred):.4f}")
    print(classification_report(y_test, pred, target_names=['Stay', 'Leave']))

# ------------------------------------------------------------------
# 5. FEATURE IMPORTANCE (as SHAP proxy)
# ------------------------------------------------------------------
feat_imp = pd.Series(dt.feature_importances_, index=features).sort_values(ascending=True)

# ------------------------------------------------------------------
# 6. VISUALIZATIONS
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('HR Analytics - Employee Attrition Dashboard', fontsize=16, fontweight='bold')

# Confusion Matrix
cm = confusion_matrix(y_test, dt_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['Stay', 'Leave'], yticklabels=['Stay', 'Leave'])
axes[0, 0].set_title('Confusion Matrix (Decision Tree)')
axes[0, 0].set_ylabel('Actual')
axes[0, 0].set_xlabel('Predicted')

# Feature Importance
feat_imp.plot(kind='barh', ax=axes[0, 1], color='#e74c3c')
axes[0, 1].set_title('Feature Importance')

# Department-wise attrition
dept_attr.plot(kind='bar', ax=axes[1, 0], color='#3498db')
axes[1, 0].set_title('Attrition Rate by Department')
axes[1, 0].set_ylabel('Attrition Rate')
axes[1, 0].tick_params(axis='x', rotation=45)

# OverTime impact
ot_attr = df.groupby('OverTime')['Attrition'].mean()
ot_attr.plot(kind='bar', ax=axes[1, 1], color=['#2ecc71', '#e74c3c'])
axes[1, 1].set_title('Attrition Rate by OverTime')
axes[1, 1].set_ylabel('Attrition Rate')
axes[1, 1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('hr_attrition_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nDashboard saved as hr_attrition_dashboard.png")

# ------------------------------------------------------------------
# 7. ATTRITION PREVENTION SUGGESTIONS
# ------------------------------------------------------------------
print("\n" + "="*60)
print("ATTRITION PREVENTION SUGGESTIONS")
print("="*60)
print("1. Implement overtime caps - OverTime is the strongest predictor")
print("2. Target retention bonuses for employees with 2-5 years tenure")
print("3. Improve job satisfaction through regular feedback cycles")
print("4. Offer remote work options for employees with long commutes")
print("5. Review compensation for low salary band employees")
