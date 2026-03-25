#!/usr/bin/env python3
"""
Project 7: Financial KPI Analysis for a Startup
Uses: Pandas, Matplotlib, Seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
months = pd.date_range('2023-01', periods=12, freq='MS')

# Simulate startup financials
revenue = np.array([10, 15, 22, 30, 38, 45, 55, 63, 72, 85, 95, 110]) * 1000
expenses = np.array([50, 52, 48, 55, 53, 58, 60, 57, 62, 65, 63, 68]) * 1000
new_customers = np.array([20, 25, 35, 40, 45, 50, 60, 55, 65, 75, 80, 90])
total_customers = np.cumsum(new_customers) - np.array([0, 2, 5, 8, 12, 15, 18, 22, 25, 30, 35, 38])
marketing_spend = np.array([15, 18, 16, 20, 18, 22, 25, 20, 24, 28, 25, 30]) * 1000

df = pd.DataFrame({
    'month': months,
    'revenue': revenue,
    'expenses': expenses,
    'new_customers': new_customers,
    'total_customers': total_customers,
    'marketing_spend': marketing_spend,
})

# KPI Calculations
df['burn_rate'] = df['expenses'] - df['revenue']
df['cac'] = df['marketing_spend'] / df['new_customers']
df['arpu'] = df['revenue'] / df['total_customers']
df['ltv'] = df['arpu'] * 12  # Simplified: ARPU * avg lifetime (12 months)
df['ltv_cac_ratio'] = df['ltv'] / df['cac']
df['gross_margin'] = (df['revenue'] - df['expenses']) / df['revenue'] * 100
cash_balance = 500000  # Starting cash
df['cash_remaining'] = cash_balance - df['burn_rate'].cumsum()
df['runway_months'] = df['cash_remaining'] / df['burn_rate'].rolling(3).mean()

print("=== STARTUP FINANCIAL KPI SUMMARY ===")
print(df[['month', 'revenue', 'expenses', 'burn_rate', 'cac', 'ltv', 'ltv_cac_ratio']].to_string(index=False))

# Cohort analysis (simplified)
print("\n--- Monthly Cohort Retention ---")
cohort_retention = {}
for i, m in enumerate(months[:6]):
    retention = [100]
    for j in range(1, 6):
        retention.append(round(100 * (0.85 ** j) + np.random.uniform(-5, 5), 1))
    cohort_retention[m.strftime('%b')] = retention
cohort_df = pd.DataFrame(cohort_retention, index=[f'M+{i}' for i in range(6)])
print(cohort_df)

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Startup Financial KPI Dashboard', fontsize=16, fontweight='bold')

# Revenue vs Expenses
axes[0, 0].plot(df['month'], df['revenue']/1000, marker='o', label='Revenue', color='#2ecc71')
axes[0, 0].plot(df['month'], df['expenses']/1000, marker='s', label='Expenses', color='#e74c3c')
axes[0, 0].fill_between(df['month'], df['revenue']/1000, df['expenses']/1000, alpha=0.1,
                          where=df['revenue'] > df['expenses'], color='green')
axes[0, 0].fill_between(df['month'], df['revenue']/1000, df['expenses']/1000, alpha=0.1,
                          where=df['revenue'] <= df['expenses'], color='red')
axes[0, 0].set_title('Revenue vs Expenses (K$)')
axes[0, 0].legend()
axes[0, 0].tick_params(axis='x', rotation=45)

# LTV:CAC Ratio
colors = ['#2ecc71' if x >= 3 else '#f39c12' if x >= 2 else '#e74c3c' for x in df['ltv_cac_ratio']]
axes[0, 1].bar(df['month'].dt.strftime('%b'), df['ltv_cac_ratio'], color=colors)
axes[0, 1].axhline(y=3, color='green', linestyle='--', label='Target (3:1)')
axes[0, 1].set_title('LTV:CAC Ratio')
axes[0, 1].legend()
axes[0, 1].tick_params(axis='x', rotation=45)

# Burn Rate & Runway
axes[0, 2].bar(df['month'].dt.strftime('%b'), df['burn_rate']/1000,
               color=['#e74c3c' if x > 0 else '#2ecc71' for x in df['burn_rate']])
axes[0, 2].set_title('Monthly Burn Rate (K$)')
axes[0, 2].tick_params(axis='x', rotation=45)

# Customer Growth
axes[1, 0].plot(df['month'], df['total_customers'], marker='o', color='#3498db')
axes[1, 0].set_title('Total Customers')
axes[1, 0].tick_params(axis='x', rotation=45)

# CAC Trend
axes[1, 1].plot(df['month'], df['cac'], marker='s', color='#e74c3c')
axes[1, 1].set_title('Customer Acquisition Cost ($)')
axes[1, 1].tick_params(axis='x', rotation=45)

# Cohort Heatmap
sns.heatmap(cohort_df.astype(float), annot=True, fmt='.0f', cmap='YlGn', ax=axes[1, 2])
axes[1, 2].set_title('Cohort Retention (%)')

plt.tight_layout()
plt.savefig('financial_kpi_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()

# Export Excel-style summary
df.to_csv('financial_summary.csv', index=False)
print("\nDashboard saved as financial_kpi_dashboard.png")
print("Financial summary exported to financial_summary.csv")
