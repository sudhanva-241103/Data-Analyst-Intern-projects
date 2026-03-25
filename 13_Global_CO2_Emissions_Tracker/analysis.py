#!/usr/bin/env python3
"""
Project 13: Global CO2 Emissions Tracker by Sector
Uses: Pandas, Matplotlib, Seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
countries = ['China', 'USA', 'India', 'Russia', 'Japan', 'Germany', 'UK', 'Brazil', 'Canada', 'Australia']
years = list(range(2010, 2024))
sectors = ['Energy', 'Industry', 'Transport']
populations = {'China': 1400, 'USA': 330, 'India': 1380, 'Russia': 144, 'Japan': 126,
               'Germany': 83, 'UK': 67, 'Brazil': 212, 'Canada': 38, 'Australia': 25}
gdp_T = {'China': 14.7, 'USA': 21.4, 'India': 2.9, 'Russia': 1.5, 'Japan': 5.1,
          'Germany': 3.8, 'UK': 2.8, 'Brazil': 1.4, 'Canada': 1.6, 'Australia': 1.3}

base_emissions = {'China': 10000, 'USA': 5000, 'India': 2600, 'Russia': 1700, 'Japan': 1100,
                  'Germany': 700, 'UK': 400, 'Brazil': 500, 'Canada': 600, 'Australia': 400}
sector_split = {'Energy': 0.73, 'Industry': 0.21, 'Transport': 0.06}
growth_rates = {'China': 0.03, 'USA': -0.01, 'India': 0.05, 'Russia': 0.005,
                'Japan': -0.015, 'Germany': -0.025, 'UK': -0.03, 'Brazil': 0.02,
                'Canada': -0.005, 'Australia': 0.01}

data = []
for country in countries:
    for i, year in enumerate(years):
        total = base_emissions[country] * (1 + growth_rates[country]) ** i
        for sector in sectors:
            emissions = total * sector_split[sector] * (1 + np.random.uniform(-0.05, 0.05))
            data.append({
                'country': country, 'year': year, 'sector': sector,
                'emissions_mt': round(emissions, 1),
                'population_m': populations[country],
                'gdp_trillion': gdp_T[country],
            })

df = pd.DataFrame(data)
df['per_capita'] = (df['emissions_mt'] / df['population_m']).round(2)
df['per_gdp'] = (df['emissions_mt'] / df['gdp_trillion']).round(1)

print(f"Dataset: {df.shape}")

# Latest year summary
latest = df[df['year'] == 2023].groupby('country').agg({'emissions_mt': 'sum', 'per_capita': 'sum'}).sort_values('emissions_mt', ascending=False)
print("\n--- 2023 Total Emissions by Country (MT CO2) ---")
print(latest.round(1))

# Sector breakdown
sector_total = df[df['year'] == 2023].groupby('sector')['emissions_mt'].sum()
print(f"\n--- Global Sector Breakdown (2023) ---")
print(sector_total.round(0))

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Global CO2 Emissions Tracker', fontsize=16, fontweight='bold')

# Top emitters bar
top_emit = df[df['year'] == 2023].groupby('country')['emissions_mt'].sum().sort_values(ascending=True)
top_emit.plot(kind='barh', ax=axes[0, 0], color='#e74c3c')
axes[0, 0].set_title('Total CO2 Emissions by Country (2023, MT)')

# Sector pie
axes[0, 1].pie(sector_total, labels=sector_total.index, autopct='%1.1f%%',
               colors=['#e74c3c', '#3498db', '#f39c12'])
axes[0, 1].set_title('Global Emissions by Sector (2023)')

# Trends for top 5
for country in ['China', 'USA', 'India', 'Russia', 'Japan']:
    country_data = df[df['country'] == country].groupby('year')['emissions_mt'].sum()
    axes[1, 0].plot(country_data.index, country_data.values, marker='.', label=country)
axes[1, 0].set_title('Emissions Trend (Top 5)')
axes[1, 0].legend(fontsize=8)
axes[1, 0].set_xlabel('Year')

# Per capita comparison
pc = df[df['year'] == 2023].groupby('country')['per_capita'].sum().sort_values(ascending=True)
pc.plot(kind='barh', ax=axes[1, 1], color='#2ecc71')
axes[1, 1].set_title('Per Capita Emissions (2023, MT per Million)')

plt.tight_layout()
plt.savefig('co2_emissions_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nDashboard saved as co2_emissions_dashboard.png")
