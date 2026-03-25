#!/usr/bin/env python3
"""
Project 5: E-commerce Return Rate Reduction Analysis
Uses: Pandas, Scikit-learn, Matplotlib, Seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
n = 2000
categories = ['Clothing', 'Electronics', 'Home & Kitchen', 'Books', 'Sports', 'Beauty']
suppliers = ['Supplier_A', 'Supplier_B', 'Supplier_C', 'Supplier_D']
regions = ['North', 'South', 'East', 'West']
channels = ['Organic', 'Paid Ads', 'Email', 'Social Media', 'Referral']

df = pd.DataFrame({
    'order_id': [f'ORD_{i:05d}' for i in range(n)],
    'category': np.random.choice(categories, n, p=[0.25, 0.2, 0.2, 0.15, 0.1, 0.1]),
    'supplier': np.random.choice(suppliers, n),
    'region': np.random.choice(regions, n),
    'channel': np.random.choice(channels, n),
    'price': np.random.uniform(10, 500, n).round(2),
    'discount_pct': np.random.uniform(0, 0.5, n).round(2),
    'delivery_days': np.random.randint(1, 15, n),
    'customer_orders': np.random.poisson(5, n) + 1,
    'rating': np.random.uniform(1, 5, n).round(1),
})

# Return probability
return_prob = (
    0.08
    + (df['category'] == 'Clothing').astype(float) * 0.15
    + (df['category'] == 'Electronics').astype(float) * 0.1
    + df['discount_pct'] * 0.3
    + (df['delivery_days'] > 7).astype(float) * 0.1
    - df['rating'] * 0.03
).clip(0.02, 0.6)
df['returned'] = np.random.binomial(1, return_prob)

print(f"Dataset: {df.shape}")
print(f"Overall Return Rate: {df['returned'].mean():.1%}")

# EDA
print("\n--- Return Rate by Category ---")
print(df.groupby('category')['returned'].mean().sort_values(ascending=False).round(3))

# Model
le = LabelEncoder()
df_model = df.copy()
for col in ['category', 'supplier', 'region', 'channel']:
    df_model[col] = le.fit_transform(df_model[col])

features = ['category', 'supplier', 'region', 'channel', 'price', 'discount_pct',
            'delivery_days', 'customer_orders', 'rating']
X = df_model[features]
y = df_model['returned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_sc, y_train)
pred = model.predict(X_test_sc)

print(f"\nModel Accuracy: {accuracy_score(y_test, pred):.4f}")
print(classification_report(y_test, pred, target_names=['Not Returned', 'Returned']))

# Risk scores
df['return_risk'] = model.predict_proba(scaler.transform(df_model[features]))[:, 1]
high_risk = df[df['return_risk'] > 0.5][['order_id', 'category', 'price', 'return_risk']]
high_risk.to_csv('high_risk_products.csv', index=False)
print(f"\nHigh-risk products exported: {len(high_risk)} orders")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('E-commerce Return Rate Analysis', fontsize=16, fontweight='bold')

cat_ret = df.groupby('category')['returned'].mean().sort_values()
cat_ret.plot(kind='barh', ax=axes[0, 0], color='#e74c3c')
axes[0, 0].set_title('Return Rate by Category')

sns.boxplot(data=df, x='returned', y='discount_pct', ax=axes[0, 1], palette='Set2')
axes[0, 1].set_title('Discount % vs Return Status')
axes[0, 1].set_xticklabels(['Not Returned', 'Returned'])

cm = confusion_matrix(y_test, pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=axes[1, 0],
            xticklabels=['Keep', 'Return'], yticklabels=['Keep', 'Return'])
axes[1, 0].set_title('Confusion Matrix')

coefs = pd.Series(model.coef_[0], index=features).sort_values()
coefs.plot(kind='barh', ax=axes[1, 1], color='#3498db')
axes[1, 1].set_title('Feature Coefficients')

plt.tight_layout()
plt.savefig('return_analysis_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("Dashboard saved as return_analysis_dashboard.png")
