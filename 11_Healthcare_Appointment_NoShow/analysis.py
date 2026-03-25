#!/usr/bin/env python3
"""
Project 11: Healthcare Appointment No-Show Prediction
Uses: Pandas, Scikit-learn, Matplotlib, Seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
n = 5000
df = pd.DataFrame({
    'patient_id': range(1, n+1),
    'age': np.random.randint(1, 85, n),
    'gender': np.random.choice(['M', 'F'], n),
    'neighborhood': np.random.choice(['Area_A', 'Area_B', 'Area_C', 'Area_D'], n),
    'scholarship': np.random.choice([0, 1], n, p=[0.8, 0.2]),
    'hypertension': np.random.choice([0, 1], n, p=[0.8, 0.2]),
    'diabetes': np.random.choice([0, 1], n, p=[0.9, 0.1]),
    'sms_received': np.random.choice([0, 1], n, p=[0.4, 0.6]),
    'weekday': np.random.choice(['Mon', 'Tue', 'Wed', 'Thu', 'Fri'], n),
    'wait_days': np.random.randint(0, 60, n),
})

# No-show probability
noshow_prob = (
    0.15
    + (df['sms_received'] == 0).astype(float) * 0.1
    + (df['wait_days'] > 20).astype(float) * 0.1
    + ((df['age'] >= 18) & (df['age'] <= 30)).astype(float) * 0.08
    - (df['hypertension'] == 1).astype(float) * 0.05
    + (df['weekday'] == 'Mon').astype(float) * 0.05
).clip(0.05, 0.5)
df['no_show'] = np.random.binomial(1, noshow_prob)

print(f"Dataset: {df.shape}")
print(f"No-Show Rate: {df['no_show'].mean():.1%}")

# EDA
print("\n--- No-Show by SMS Received ---")
print(df.groupby('sms_received')['no_show'].mean().round(3))
print("\n--- No-Show by Age Group ---")
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 30, 45, 60, 100], labels=['0-18', '18-30', '30-45', '45-60', '60+'])
print(df.groupby('age_group')['no_show'].mean().round(3))

# Model
le = LabelEncoder()
df_model = df.copy()
for col in ['gender', 'neighborhood', 'weekday', 'age_group']:
    df_model[col] = le.fit_transform(df_model[col])

features = ['age', 'gender', 'scholarship', 'hypertension', 'diabetes',
            'sms_received', 'weekday', 'wait_days']
X = df_model[features]
y = df_model['no_show']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

dt = DecisionTreeClassifier(max_depth=6, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print(f"\nDecision Tree Accuracy: {accuracy_score(y_test, dt_pred):.4f}")
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
print("\n" + classification_report(y_test, rf_pred, target_names=['Show', 'No-Show']))

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Healthcare No-Show Prediction Dashboard', fontsize=16, fontweight='bold')

sms_noshow = df.groupby('sms_received')['no_show'].mean()
axes[0, 0].bar(['No SMS', 'SMS Received'], sms_noshow.values, color=['#e74c3c', '#2ecc71'])
axes[0, 0].set_title('No-Show Rate by SMS Status')
axes[0, 0].set_ylabel('No-Show Rate')

age_noshow = df.groupby('age_group')['no_show'].mean()
age_noshow.plot(kind='bar', ax=axes[0, 1], color='#3498db')
axes[0, 1].set_title('No-Show Rate by Age Group')
axes[0, 1].tick_params(axis='x', rotation=45)

cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
            xticklabels=['Show', 'No-Show'], yticklabels=['Show', 'No-Show'])
axes[1, 0].set_title('Confusion Matrix')

feat_imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)
feat_imp.plot(kind='barh', ax=axes[1, 1], color='#f39c12')
axes[1, 1].set_title('Feature Importance')

plt.tight_layout()
plt.savefig('noshow_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nDashboard saved as noshow_dashboard.png")

print("\n=== OPTIMIZATION RECOMMENDATIONS ===")
print("1. Send automated SMS reminders to all patients (reduces no-shows by ~10%)")
print("2. Keep booking windows under 20 days when possible")
print("3. Overbook by 10-15% for Monday morning slots")
print("4. Target younger patients (18-30) with extra reminders")
