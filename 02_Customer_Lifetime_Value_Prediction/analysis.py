#!/usr/bin/env python3
"""
Project 2: Customer Lifetime Value Prediction Model
Uses: Pandas, Scikit-learn, XGBoost (or RandomForest), Matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# 1. GENERATE SAMPLE DATA (replace with real dataset)
# ------------------------------------------------------------------
np.random.seed(42)
n_customers = 1000

customers = pd.DataFrame({
    'customer_id': [f'CUST_{i:04d}' for i in range(n_customers)],
    'recency': np.random.exponential(30, n_customers).astype(int),  # days since last purchase
    'frequency': np.random.poisson(5, n_customers) + 1,              # number of purchases
    'monetary': np.random.exponential(200, n_customers),              # total spend
    'tenure_days': np.random.randint(30, 730, n_customers),           # days as customer
    'avg_order_value': np.random.exponential(50, n_customers) + 10,
    'num_categories': np.random.randint(1, 8, n_customers),
    'avg_discount': np.random.uniform(0, 0.3, n_customers),
})

# Create target LTV (correlated with features)
customers['ltv'] = (
    customers['frequency'] * customers['avg_order_value'] * 2.5
    + customers['monetary'] * 0.5
    - customers['recency'] * 2
    + customers['tenure_days'] * 0.3
    + np.random.normal(0, 50, n_customers)
).clip(0)

print(f"Dataset: {customers.shape[0]} customers, {customers.shape[1]} features")
print(f"\nLTV Statistics:\n{customers['ltv'].describe()}")

# ------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ------------------------------------------------------------------
features = ['recency', 'frequency', 'monetary', 'tenure_days',
            'avg_order_value', 'num_categories', 'avg_discount']
X = customers[features]
y = customers['ltv']

# Scale features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)

# ------------------------------------------------------------------
# 3. TRAIN/TEST SPLIT & MODEL TRAINING
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Gradient Boosting (as XGBoost alternative)
gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)

# ------------------------------------------------------------------
# 4. EVALUATION
# ------------------------------------------------------------------
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)
for name, pred in [('Random Forest', rf_pred), ('Gradient Boosting', gb_pred)]:
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    print(f"\n{name}:")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R2:   {r2:.4f}")

# ------------------------------------------------------------------
# 5. FEATURE IMPORTANCE
# ------------------------------------------------------------------
feat_imp = pd.Series(gb.feature_importances_, index=features).sort_values(ascending=True)

# ------------------------------------------------------------------
# 6. CUSTOMER SEGMENTATION
# ------------------------------------------------------------------
customers['predicted_ltv'] = gb.predict(scaler.transform(customers[features]))
customers['ltv_segment'] = pd.qcut(customers['predicted_ltv'], q=3, labels=['Low', 'Medium', 'High'])

segment_summary = customers.groupby('ltv_segment').agg({
    'predicted_ltv': ['mean', 'count'],
    'frequency': 'mean',
    'monetary': 'mean'
}).round(2)
print("\n--- Customer Segments ---")
print(segment_summary)

# ------------------------------------------------------------------
# 7. VISUALIZATIONS
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Customer Lifetime Value Analysis', fontsize=16, fontweight='bold')

# Actual vs Predicted
axes[0, 0].scatter(y_test, gb_pred, alpha=0.5, color='#3498db', s=20)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[0, 0].set_xlabel('Actual LTV')
axes[0, 0].set_ylabel('Predicted LTV')
axes[0, 0].set_title('Actual vs Predicted LTV')

# Feature Importance
feat_imp.plot(kind='barh', ax=axes[0, 1], color='#2ecc71')
axes[0, 1].set_title('Feature Importance')
axes[0, 1].set_xlabel('Importance')

# LTV Distribution by Segment
for seg in ['Low', 'Medium', 'High']:
    subset = customers[customers['ltv_segment'] == seg]['predicted_ltv']
    axes[1, 0].hist(subset, alpha=0.6, label=seg, bins=20)
axes[1, 0].set_title('LTV Distribution by Segment')
axes[1, 0].set_xlabel('Predicted LTV')
axes[1, 0].legend()

# Segment summary
seg_counts = customers['ltv_segment'].value_counts()
axes[1, 1].pie(seg_counts, labels=seg_counts.index, autopct='%1.1f%%',
               colors=['#e74c3c', '#f39c12', '#2ecc71'])
axes[1, 1].set_title('Customer Segment Distribution')

plt.tight_layout()
plt.savefig('ltv_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()

# Export predictions
customers[['customer_id', 'predicted_ltv', 'ltv_segment']].to_csv('ltv_predictions.csv', index=False)
print("\nPredictions exported to ltv_predictions.csv")
print("Dashboard saved as ltv_dashboard.png")
