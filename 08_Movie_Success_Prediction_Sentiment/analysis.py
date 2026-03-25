#!/usr/bin/env python3
"""
Project 8: Movie Success Prediction and Sentiment Study
Uses: Pandas, TextBlob, Scikit-learn, Matplotlib
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

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except:
    HAS_TEXTBLOB = False

np.random.seed(42)
n = 1000
genres = ['Action', 'Comedy', 'Drama', 'Thriller', 'Horror', 'Romance', 'Sci-Fi', 'Animation']

df = pd.DataFrame({
    'movie_id': range(1, n+1),
    'genre': np.random.choice(genres, n, p=[0.2, 0.15, 0.2, 0.12, 0.08, 0.1, 0.08, 0.07]),
    'budget_M': np.random.exponential(50, n).clip(1, 300).round(1),
    'runtime': np.random.normal(120, 20, n).clip(80, 200).astype(int),
    'rating': np.random.uniform(2, 9, n).round(1),
    'vote_count': np.random.exponential(1000, n).astype(int),
    'cast_popularity': np.random.exponential(30, n).round(1),
})

# Sample reviews
reviews = [
    "Absolutely amazing movie, loved every minute!",
    "Terrible waste of time, boring plot.",
    "Decent film with great acting performances.",
    "Not bad but not great either.",
    "Masterpiece of cinema, must watch!",
    "Disappointing sequel, lost the charm.",
    "Fun and entertaining family movie.",
    "Dark and gripping thriller, well made.",
]
df['sample_review'] = np.random.choice(reviews, n)

if HAS_TEXTBLOB:
    df['sentiment'] = df['sample_review'].apply(lambda x: TextBlob(x).sentiment.polarity)
else:
    df['sentiment'] = np.random.uniform(-0.5, 0.8, n)

# Revenue (correlated)
df['revenue_M'] = (
    df['budget_M'] * np.random.uniform(0.5, 4, n)
    + df['cast_popularity'] * 2
    + df['rating'] * 10
    + df['sentiment'] * 50
    + np.random.normal(0, 30, n)
).clip(0.5)

# Model
le = LabelEncoder()
df['genre_enc'] = le.fit_transform(df['genre'])
features = ['budget_M', 'runtime', 'rating', 'vote_count', 'cast_popularity', 'sentiment', 'genre_enc']
X = df[features]
y = df['revenue_M']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print(f"R-squared: {r2_score(y_test, pred):.4f}")
print(f"MAE: ${mean_absolute_error(y_test, pred):.1f}M")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Movie Success Prediction & Sentiment Analysis', fontsize=16, fontweight='bold')

axes[0, 0].scatter(y_test, pred, alpha=0.5, color='#3498db')
axes[0, 0].plot([0, y_test.max()], [0, y_test.max()], 'r--')
axes[0, 0].set_title(f'Actual vs Predicted Revenue (R2={r2_score(y_test, pred):.2f})')
axes[0, 0].set_xlabel('Actual Revenue ($M)')
axes[0, 0].set_ylabel('Predicted Revenue ($M)')

feat_imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
feat_imp.plot(kind='barh', ax=axes[0, 1], color='#2ecc71')
axes[0, 1].set_title('Feature Importance')

genre_sent = df.groupby('genre')['sentiment'].mean().sort_values()
genre_sent.plot(kind='barh', ax=axes[1, 0], color='#f39c12')
axes[1, 0].set_title('Avg Sentiment by Genre')

genre_rev = df.groupby('genre')['revenue_M'].mean().sort_values()
genre_rev.plot(kind='barh', ax=axes[1, 1], color='#9b59b6')
axes[1, 1].set_title('Avg Revenue by Genre ($M)')

plt.tight_layout()
plt.savefig('movie_analysis_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("Dashboard saved as movie_analysis_dashboard.png")
