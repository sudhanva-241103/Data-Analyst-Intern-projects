#!/usr/bin/env python3
"""
Project 12: Electric Vehicle Charging Demand Forecasting
Uses: Pandas, Matplotlib, Seaborn, Statsmodels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
# Generate 6 months of hourly data
dates = pd.date_range('2024-01-01', periods=180*24, freq='h')

df = pd.DataFrame({'timestamp': dates})
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Weather simulation
df['temperature'] = 15 + 10 * np.sin(2 * np.pi * (df['month'] - 1) / 12) + np.random.normal(0, 3, len(df))
df['precipitation'] = np.random.exponential(0.5, len(df)).clip(0, 10)

# Demand pattern (sessions per hour)
peak_pattern = np.zeros(24)
peak_pattern[7:10] = [3, 5, 4]  # Morning peak
peak_pattern[11:14] = [4, 5, 4]  # Lunch peak
peak_pattern[17:21] = [6, 8, 7, 5]  # Evening peak
peak_pattern[21:24] = [3, 2, 1]
peak_pattern[0:7] = [0.5, 0.3, 0.2, 0.2, 0.3, 0.5, 1]

df['base_demand'] = df['hour'].map(lambda h: peak_pattern[h])
df['demand'] = (
    df['base_demand']
    * (1 - df['is_weekend'] * 0.3)
    * (1 + (df['temperature'] < 10).astype(float) * 0.18)
    * (1 - df['precipitation'] * 0.02)
    + np.random.normal(0, 0.5, len(df))
).clip(0).round(1)

print(f"Dataset: {df.shape}")
print(f"Avg daily demand: {df.groupby(df['timestamp'].dt.date)['demand'].sum().mean():.1f} sessions")

# Time-series decomposition (simplified)
daily = df.groupby(df['timestamp'].dt.date).agg({'demand': 'sum', 'temperature': 'mean'}).reset_index()
daily.columns = ['date', 'daily_demand', 'avg_temp']
daily['trend'] = daily['daily_demand'].rolling(7).mean()

# Forecasting model
features = ['hour', 'day_of_week', 'month', 'is_weekend', 'temperature', 'precipitation']
X = df[features]
y = df['demand']

train_size = int(len(df) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
mape = mean_absolute_percentage_error(y_test[y_test > 0], pred[y_test > 0]) * 100
print(f"\nForecast MAE: {mae:.2f}")
print(f"Forecast MAPE: {mape:.1f}%")

# Demand heatmap data
heatmap_data = df.groupby(['day_of_week', 'hour'])['demand'].mean().unstack()
day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('EV Charging Demand Forecasting Dashboard', fontsize=16, fontweight='bold')

# Demand heatmap
sns.heatmap(heatmap_data, ax=axes[0, 0], cmap='YlOrRd', yticklabels=day_labels)
axes[0, 0].set_title('Avg Charging Demand (Hour x Day)')
axes[0, 0].set_xlabel('Hour of Day')

# Daily trend
axes[0, 1].plot(range(len(daily)), daily['daily_demand'], alpha=0.3, color='#3498db')
axes[0, 1].plot(range(len(daily)), daily['trend'], color='#e74c3c', linewidth=2, label='7-day MA')
axes[0, 1].set_title('Daily Demand Trend')
axes[0, 1].set_xlabel('Day')
axes[0, 1].legend()

# Hourly demand pattern
hourly_avg = df.groupby('hour')['demand'].mean()
axes[1, 0].fill_between(hourly_avg.index, hourly_avg.values, alpha=0.3, color='#2ecc71')
axes[1, 0].plot(hourly_avg.index, hourly_avg.values, color='#2ecc71', linewidth=2)
axes[1, 0].set_title('Average Hourly Demand Pattern')
axes[1, 0].set_xlabel('Hour')
axes[1, 0].set_ylabel('Avg Sessions')

# Temperature impact
temp_bins = pd.cut(df['temperature'], bins=5)
temp_demand = df.groupby(temp_bins)['demand'].mean()
temp_demand.plot(kind='bar', ax=axes[1, 1], color='#f39c12')
axes[1, 1].set_title('Demand vs Temperature')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('ev_demand_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nDashboard saved as ev_demand_dashboard.png")

print("\n=== CHARGING OPTIMIZATION STRATEGY ===")
print("1. Peak demand: 5-8 PM weekdays - ensure max charger availability")
print("2. Cold weather increases demand 18% - pre-warm charger infrastructure")
print("3. Weekend demand 30% lower - schedule maintenance on Sat/Sun mornings")
print("4. Consider dynamic pricing: higher rates 5-8 PM, lower rates midnight-6 AM")
