#!/usr/bin/env python3
"""
Project 9: Airbnb Dynamic Pricing Recommendation Engine
Uses: Pandas, Scikit-learn, Matplotlib, Seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
n = 2000
neighborhoods = ['Downtown', 'Midtown', 'Suburb', 'Beach', 'Historic', 'Airport', 'University']
prop_types = ['Entire home', 'Private room', 'Shared room']
seasons = ['Spring', 'Summer', 'Fall', 'Winter']

df = pd.DataFrame({
    'listing_id': range(1, n+1),
    'neighborhood': np.random.choice(neighborhoods, n),
    'property_type': np.random.choice(prop_types, n, p=[0.5, 0.4, 0.1]),
    'bedrooms': np.random.choice([0, 1, 2, 3, 4], n, p=[0.1, 0.35, 0.3, 0.15, 0.1]),
    'bathrooms': np.random.choice([1, 1.5, 2, 2.5, 3], n),
    'amenity_count': np.random.randint(3, 25, n),
    'review_score': np.random.uniform(3, 5, n).round(1),
    'num_reviews': np.random.exponential(30, n).astype(int),
    'season': np.random.choice(seasons, n),
    'distance_to_center': np.random.uniform(0.5, 15, n).round(1),
})

# Price generation
base_prices = {'Downtown': 180, 'Midtown': 150, 'Beach': 200, 'Historic': 160,
               'Suburb': 90, 'Airport': 100, 'University': 80}
season_mult = {'Summer': 1.3, 'Spring': 1.1, 'Fall': 1.0, 'Winter': 0.9}
type_mult = {'Entire home': 1.0, 'Private room': 0.5, 'Shared room': 0.25}

df['price'] = df.apply(lambda r: (
    base_prices[r['neighborhood']]
    * season_mult[r['season']]
    * type_mult[r['property_type']]
    * (1 + r['bedrooms'] * 0.2)
    * (1 + (r['review_score'] - 3) * 0.1)
    + np.random.normal(0, 15)
), axis=1).clip(20).round(2)

print(f"Dataset: {df.shape}")
print(f"Price range: ${df['price'].min():.0f} - ${df['price'].max():.0f}")

# Model
le_n = LabelEncoder()
le_p = LabelEncoder()
le_s = LabelEncoder()
df['neigh_enc'] = le_n.fit_transform(df['neighborhood'])
df['prop_enc'] = le_p.fit_transform(df['property_type'])
df['season_enc'] = le_s.fit_transform(df['season'])

features = ['neigh_enc', 'prop_enc', 'bedrooms', 'bathrooms', 'amenity_count',
            'review_score', 'num_reviews', 'season_enc', 'distance_to_center']
X = df[features]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print(f"\nR-squared: {r2_score(y_test, pred):.4f}")
print(f"MAE: ${mean_absolute_error(y_test, pred):.2f}")

# Pricing recommendation function
def recommend_price(neighborhood, prop_type, bedrooms, review_score, season):
    input_data = pd.DataFrame([{
        'neigh_enc': le_n.transform([neighborhood])[0],
        'prop_enc': le_p.transform([prop_type])[0],
        'bedrooms': bedrooms, 'bathrooms': 1, 'amenity_count': 10,
        'review_score': review_score, 'num_reviews': 20,
        'season_enc': le_s.transform([season])[0], 'distance_to_center': 5
    }])
    return model.predict(input_data)[0]

print("\n--- Sample Pricing Recommendations ---")
for n_, p_, b_, r_, s_ in [
    ('Downtown', 'Entire home', 2, 4.5, 'Summer'),
    ('Suburb', 'Private room', 1, 4.0, 'Winter'),
    ('Beach', 'Entire home', 3, 4.8, 'Summer'),
]:
    price = recommend_price(n_, p_, b_, r_, s_)
    print(f"  {n_}, {p_}, {b_}BR, {r_} stars, {s_}: ${price:.0f}/night")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Airbnb Dynamic Pricing Dashboard', fontsize=16, fontweight='bold')

neigh_price = df.groupby('neighborhood')['price'].mean().sort_values()
neigh_price.plot(kind='barh', ax=axes[0, 0], color='#3498db')
axes[0, 0].set_title('Avg Price by Neighborhood')

season_price = df.groupby('season')['price'].mean()
season_price.plot(kind='bar', ax=axes[0, 1], color='#e74c3c')
axes[0, 1].set_title('Avg Price by Season')
axes[0, 1].tick_params(axis='x', rotation=0)

axes[1, 0].scatter(y_test, pred, alpha=0.4, color='#2ecc71', s=15)
axes[1, 0].plot([0, y_test.max()], [0, y_test.max()], 'r--')
axes[1, 0].set_title('Actual vs Predicted Price')

feat_imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
feat_imp.plot(kind='barh', ax=axes[1, 1], color='#f39c12')
axes[1, 1].set_title('Feature Importance')

plt.tight_layout()
plt.savefig('airbnb_pricing_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nDashboard saved as airbnb_pricing_dashboard.png")
