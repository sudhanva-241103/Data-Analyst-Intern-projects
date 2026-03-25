#!/usr/bin/env python3
"""
Project 1: Retail Business Performance & Profitability Analysis
Uses: Pandas, Seaborn, Matplotlib, SQLite
Dataset: Superstore (place CSV in same directory as 'superstore.csv')
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# 1. LOAD & CLEAN DATA
# ------------------------------------------------------------------
# NOTE: Download Superstore dataset from Kaggle and place as 'superstore.csv'
# For demo, we generate sample data
np.random.seed(42)
n = 2000
categories = ['Furniture', 'Office Supplies', 'Technology']
sub_cats = {
    'Furniture': ['Chairs', 'Tables', 'Bookcases', 'Furnishings'],
    'Office Supplies': ['Binders', 'Paper', 'Storage', 'Supplies', 'Art'],
    'Technology': ['Phones', 'Accessories', 'Copiers', 'Machines']
}
regions = ['East', 'West', 'Central', 'South']

data = []
for i in range(n):
    cat = np.random.choice(categories, p=[0.3, 0.45, 0.25])
    sub = np.random.choice(sub_cats[cat])
    region = np.random.choice(regions)
    month = np.random.randint(1, 13)
    sales = np.random.uniform(10, 2000)
    # Some sub-categories are loss-making
    if sub in ['Tables', 'Machines']:
        profit = sales * np.random.uniform(-0.3, 0.05)
    elif sub in ['Copiers']:
        profit = sales * np.random.uniform(0.2, 0.5)
    else:
        profit = sales * np.random.uniform(-0.1, 0.35)
    ship_days = np.random.randint(1, 10)
    data.append({
        'Order_ID': f'ORD-{i+1:04d}',
        'Order_Date': pd.Timestamp(2023, month, np.random.randint(1, 28)),
        'Category': cat,
        'Sub_Category': sub,
        'Region': region,
        'Sales': round(sales, 2),
        'Profit': round(profit, 2),
        'Quantity': np.random.randint(1, 15),
        'Ship_Days': ship_days
    })

df = pd.DataFrame(data)
print(f"Dataset shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")

# ------------------------------------------------------------------
# 2. SQL ANALYSIS (using SQLite)
# ------------------------------------------------------------------
conn = sqlite3.connect(':memory:')
df.to_sql('retail', conn, index=False)

# Profit margin by category
query1 = """
SELECT Category,
       ROUND(SUM(Profit), 2) as Total_Profit,
       ROUND(SUM(Sales), 2) as Total_Sales,
       ROUND(SUM(Profit)/SUM(Sales)*100, 2) as Profit_Margin_Pct
FROM retail
GROUP BY Category
ORDER BY Profit_Margin_Pct DESC
"""
margin_by_cat = pd.read_sql(query1, conn)
print("\n--- Profit Margin by Category ---")
print(margin_by_cat)

# Profit margin by sub-category
query2 = """
SELECT Category, Sub_Category,
       ROUND(SUM(Profit), 2) as Total_Profit,
       ROUND(SUM(Sales), 2) as Total_Sales,
       ROUND(SUM(Profit)/SUM(Sales)*100, 2) as Profit_Margin_Pct
FROM retail
GROUP BY Category, Sub_Category
ORDER BY Profit_Margin_Pct ASC
"""
margin_by_sub = pd.read_sql(query2, conn)
print("\n--- Profit Margin by Sub-Category ---")
print(margin_by_sub)

conn.close()

# ------------------------------------------------------------------
# 3. CORRELATION ANALYSIS
# ------------------------------------------------------------------
corr = df[['Sales', 'Profit', 'Quantity', 'Ship_Days']].corr()
print("\n--- Correlation Matrix ---")
print(corr)

# ------------------------------------------------------------------
# 4. VISUALIZATIONS
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Retail Business Performance Dashboard', fontsize=16, fontweight='bold')

# 4a. Profit margin by sub-category
ax1 = axes[0, 0]
margin_by_sub_sorted = margin_by_sub.sort_values('Profit_Margin_Pct')
colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in margin_by_sub_sorted['Profit_Margin_Pct']]
ax1.barh(margin_by_sub_sorted['Sub_Category'], margin_by_sub_sorted['Profit_Margin_Pct'], color=colors)
ax1.set_xlabel('Profit Margin (%)')
ax1.set_title('Profit Margin by Sub-Category')
ax1.axvline(x=0, color='black', linewidth=0.8)

# 4b. Monthly sales trend
ax2 = axes[0, 1]
monthly = df.groupby(df['Order_Date'].dt.month).agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()
ax2.plot(monthly['Order_Date'], monthly['Sales'], marker='o', label='Sales', color='#3498db')
ax2.plot(monthly['Order_Date'], monthly['Profit'], marker='s', label='Profit', color='#2ecc71')
ax2.set_xlabel('Month')
ax2.set_ylabel('Amount ($)')
ax2.set_title('Monthly Sales & Profit Trend')
ax2.legend()
ax2.set_xticks(range(1, 13))

# 4c. Correlation heatmap
ax3 = axes[1, 0]
sns.heatmap(corr, annot=True, cmap='RdYlGn', center=0, ax=ax3, fmt='.2f')
ax3.set_title('Correlation Heatmap')

# 4d. Region-wise performance
ax4 = axes[1, 1]
region_perf = df.groupby('Region').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()
x_pos = np.arange(len(region_perf))
width = 0.35
ax4.bar(x_pos - width/2, region_perf['Sales'], width, label='Sales', color='#3498db')
ax4.bar(x_pos + width/2, region_perf['Profit'], width, label='Profit', color='#2ecc71')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(region_perf['Region'])
ax4.set_title('Region-wise Sales & Profit')
ax4.legend()

plt.tight_layout()
plt.savefig('retail_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nDashboard saved as retail_dashboard.png")

# ------------------------------------------------------------------
# 5. STRATEGIC RECOMMENDATIONS
# ------------------------------------------------------------------
print("\n" + "="*60)
print("STRATEGIC RECOMMENDATIONS")
print("="*60)
loss_making = margin_by_sub[margin_by_sub['Profit_Margin_Pct'] < 0]
print(f"\nLoss-making sub-categories: {list(loss_making['Sub_Category'])}")
print("- Review pricing strategy and supplier contracts")
print("- Consider discontinuing or bundling with profitable items")

best = margin_by_sub.nlargest(3, 'Profit_Margin_Pct')
print(f"\nTop profitable sub-categories: {list(best['Sub_Category'])}")
print("- Increase marketing spend on these categories")
print("- Expand inventory for high-margin items")
print("\nProject complete!")
