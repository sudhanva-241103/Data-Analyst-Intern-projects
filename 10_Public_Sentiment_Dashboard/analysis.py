#!/usr/bin/env python3
"""
Project 10: Real-Time Public Sentiment Dashboard (Twitter/X)
Uses: Pandas, NLTK, TextBlob, Matplotlib, Seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except:
    HAS_TEXTBLOB = False

np.random.seed(42)
n = 3000
brands = ['BrandA', 'BrandB']
sample_tweets = [
    "Love the new product from {brand}! Amazing quality!",
    "Terrible customer service from {brand}, never buying again",
    "{brand} just released something cool, check it out",
    "Disappointed with {brand}'s latest update",
    "Great experience with {brand} support team today",
    "{brand} is overpriced for what you get",
    "Just got my order from {brand}, exactly as expected",
    "Why is {brand} so slow at shipping?",
    "Best purchase I made this year from {brand}!",
    "{brand} needs to improve their app",
]

data = []
for i in range(n):
    brand = np.random.choice(brands)
    tweet = np.random.choice(sample_tweets).format(brand=brand)
    date = pd.Timestamp('2024-01-01') + pd.Timedelta(hours=np.random.randint(0, 24*30))
    data.append({'tweet_id': i, 'text': tweet, 'brand': brand, 'created_at': date,
                 'retweet_count': np.random.poisson(5), 'favorite_count': np.random.poisson(20)})

df = pd.DataFrame(data)

# Text cleaning
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip().lower()

df['clean_text'] = df['text'].apply(clean_text)

# Sentiment
if HAS_TEXTBLOB:
    df['polarity'] = df['clean_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['subjectivity'] = df['clean_text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
else:
    df['polarity'] = np.random.uniform(-0.8, 0.8, n)
    df['subjectivity'] = np.random.uniform(0, 1, n)

df['sentiment'] = df['polarity'].apply(
    lambda x: 'Positive' if x > 0.1 else ('Negative' if x < -0.1 else 'Neutral'))

print(f"Dataset: {df.shape}")
print(f"\nSentiment Distribution:\n{df['sentiment'].value_counts()}")
print(f"\nAvg Polarity: {df['polarity'].mean():.3f}")

# Daily sentiment trends
df['date'] = df['created_at'].dt.date
daily = df.groupby(['date', 'brand']).agg({'polarity': 'mean', 'tweet_id': 'count'}).reset_index()
daily.columns = ['date', 'brand', 'avg_polarity', 'tweet_count']

# Word frequency
all_words = ' '.join(df['clean_text']).split()
stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'from', 'for', 'with',
             'at', 'by', 'of', 'in', 'on', 'it', 'i', 'my', 'me', 'so', 'just', 'got'}
word_freq = Counter(w for w in all_words if w not in stopwords and len(w) > 2)

# Batch logging
df[['tweet_id', 'date', 'brand', 'polarity', 'sentiment']].to_csv('sentiment_log.csv', index=False)

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Public Sentiment Dashboard', fontsize=16, fontweight='bold')

# Sentiment distribution
sent_counts = df['sentiment'].value_counts()
axes[0, 0].pie(sent_counts, labels=sent_counts.index, autopct='%1.1f%%',
               colors=['#2ecc71', '#95a5a6', '#e74c3c'])
axes[0, 0].set_title('Overall Sentiment Distribution')

# Polarity over time
for brand in brands:
    brand_data = daily[daily['brand'] == brand]
    axes[0, 1].plot(range(len(brand_data)), brand_data['avg_polarity'], marker='.', label=brand, alpha=0.7)
axes[0, 1].set_title('Daily Sentiment Trend')
axes[0, 1].set_xlabel('Day')
axes[0, 1].set_ylabel('Avg Polarity')
axes[0, 1].legend()
axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# Brand comparison
brand_sent = df.groupby(['brand', 'sentiment']).size().unstack(fill_value=0)
brand_sent.plot(kind='bar', ax=axes[1, 0], color=['#e74c3c', '#95a5a6', '#2ecc71'])
axes[1, 0].set_title('Sentiment by Brand')
axes[1, 0].tick_params(axis='x', rotation=0)

# Top words
top_words = word_freq.most_common(15)
words, counts = zip(*top_words)
axes[1, 1].barh(words, counts, color='#3498db')
axes[1, 1].set_title('Top 15 Words')

plt.tight_layout()
plt.savefig('sentiment_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nDashboard saved as sentiment_dashboard.png")
print("Sentiment log exported to sentiment_log.csv")
