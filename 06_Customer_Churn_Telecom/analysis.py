#!/usr/bin/env python3
"""
Project 6: Customer Churn Analysis for Telecom Industry
Uses: Pandas, Scikit-learn, Matplotlib, Seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
n = 2000
df = pd.DataFrame({
    'customerID': [f'CUST_{i:05d}' for i in range(n)],
    'gender': np.random.choice(['Male', 'Female'], n),
    'tenure': np.random.randint(1, 72, n),
    'contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.5, 0.3, 0.2]),
    'monthly_charges': np.random.uniform(20, 120, n).round(2),
    'total_charges': np.random.uniform(100, 8000, n).round(2),
    'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.35, 0.4, 0.25]),
    'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n),
    'num_complaints': np.random.poisson(1, n),
    'call_duration_avg': np.random.uniform(1, 30, n).round(1),
    'recharge_frequency': np.random.poisson(3, n) + 1,
})

churn_prob = (
    0.1
    + (df['contract'] == 'Month-to-month').astype(float) * 0.2
    + (df['payment_method'] == 'Electronic check').astype(float) * 0.1
    + (df['internet_service'] == 'Fiber optic').astype(float) * 0.1
    - df['tenure'] * 0.003
    + df['num_complaints'] * 0.05
    + df['monthly_charges'] * 0.001
).clip(0.05, 0.7)
df['churn'] = np.random.binomial(1, churn_prob)

print(f"Dataset: {df.shape}")
print(f"Churn Rate: {df['churn'].mean():.1%}")

# Preprocess
le = LabelEncoder()
df_model = df.copy()
for col in ['gender', 'contract', 'internet_service', 'payment_method']:
    df_model[col] = le.fit_transform(df_model[col])

features = ['gender', 'tenure', 'contract', 'monthly_charges', 'total_charges',
            'internet_service', 'payment_method', 'num_complaints',
            'call_duration_avg', 'recharge_frequency']
X = df_model[features]
y = df_model['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
pred = model.predict(X_test)
pred_proba = model.predict_proba(X_test)[:, 1]

print(f"\nAccuracy: {accuracy_score(y_test, pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, pred_proba):.4f}")
print(classification_report(y_test, pred, target_names=['No Churn', 'Churn']))

# Segmentation
df['churn_risk'] = model.predict_proba(df_model[features])[:, 1]
df['segment'] = pd.cut(df['churn_risk'], bins=[0, 0.3, 0.6, 1.0], labels=['Loyal', 'Dormant', 'At Risk'])
print("\n--- Customer Segments ---")
print(df['segment'].value_counts())

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Telecom Customer Churn Analysis', fontsize=16, fontweight='bold')

feat_imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
feat_imp.plot(kind='barh', ax=axes[0, 0], color='#e74c3c')
axes[0, 0].set_title('Feature Importance')

fpr, tpr, _ = roc_curve(y_test, pred_proba)
axes[0, 1].plot(fpr, tpr, color='#3498db', lw=2)
axes[0, 1].plot([0, 1], [0, 1], 'k--')
axes[0, 1].set_title(f'ROC Curve (AUC: {roc_auc_score(y_test, pred_proba):.3f})')
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')

seg_counts = df['segment'].value_counts()
axes[1, 0].pie(seg_counts, labels=seg_counts.index, autopct='%1.1f%%',
               colors=['#2ecc71', '#f39c12', '#e74c3c'])
axes[1, 0].set_title('Customer Segments')

cm = confusion_matrix(y_test, pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
axes[1, 1].set_title('Confusion Matrix')

plt.tight_layout()
plt.savefig('churn_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nDashboard saved as churn_dashboard.png")
