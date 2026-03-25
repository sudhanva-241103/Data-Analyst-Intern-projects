#!/usr/bin/env python3
"""
Project 4: YouTube Trending Video Analytics
Uses: Pandas, Matplotlib, Seaborn, TextBlob, SQLite
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False
    print("TextBlob not installed. Sentiment will use random scores for demo.")

# ------------------------------------------------------------------
# 1. GENERATE SAMPLE DATA
# ------------------------------------------------------------------
np.random.seed(42)
n = 2000
categories = ['Entertainment', 'Music', 'Gaming', 'News & Politics', 'Comedy',
              'Science & Tech', 'Sports', 'Education', 'Howto & Style', 'Film & Animation']
regions = ['US', 'UK', 'IN', 'CA', 'DE']

titles_base = ['Amazing', 'Top 10', 'How to', 'Best', 'New', 'Official', 'Review',
               'Tutorial', 'Unboxing', 'Challenge', 'Viral', 'Breaking', 'Live']

data = []
for i in range(n):
    cat = np.random.choice(categories, p=[0.2, 0.18, 0.15, 0.1, 0.1, 0.08, 0.07, 0.05, 0.04, 0.03])
    region = np.random.choice(regions, p=[0.3, 0.2, 0.25, 0.15, 0.1])
    views = int(np.random.exponential(500000))
    title = f"{np.random.choice(titles_base)} {cat} Video {np.random.randint(1,100)}"
    tags = '|'.join(np.random.choice(['funny', 'viral', 'trending', 'new', 'official',
                                       'music', 'gaming', 'news', 'tech', 'review',
                                       'tutorial', 'vlog', 'comedy'], size=np.random.randint(5, 20)))
    data.append({
        'video_id': f'VID_{i:05d}',
        'title': title,
        'category': cat,
        'region': region,
        'views': views,
        'likes': int(views * np.random.uniform(0.02, 0.08)),
        'dislikes': int(views * np.random.uniform(0.001, 0.01)),
        'comment_count': int(views * np.random.uniform(0.005, 0.03)),
        'tags': tags,
        'trending_days': np.random.randint(1, 8),
        'publish_hour': np.random.randint(0, 24),
        'publish_month': np.random.randint(1, 13),
    })

df = pd.DataFrame(data)
print(f"Dataset: {df.shape}")

# ------------------------------------------------------------------
# 2. SENTIMENT ANALYSIS
# ------------------------------------------------------------------
if HAS_TEXTBLOB:
    df['sentiment'] = df['title'].apply(lambda x: TextBlob(x).sentiment.polarity)
else:
    df['sentiment'] = np.random.uniform(-0.5, 0.8, n)

df['sentiment_label'] = df['sentiment'].apply(
    lambda x: 'Positive' if x > 0.1 else ('Negative' if x < -0.1 else 'Neutral'))

print(f"\nSentiment Distribution:\n{df['sentiment_label'].value_counts()}")

# ------------------------------------------------------------------
# 3. SQL ANALYSIS
# ------------------------------------------------------------------
conn = sqlite3.connect(':memory:')
df.to_sql('youtube', conn, index=False)

rank_query = """
SELECT category,
       COUNT(*) as num_videos,
       ROUND(AVG(views)) as avg_views,
       ROUND(AVG(likes)) as avg_likes,
       ROUND(AVG(trending_days), 1) as avg_trending_days
FROM youtube
GROUP BY category
ORDER BY avg_views DESC
"""
cat_ranking = pd.read_sql(rank_query, conn)
print("\n--- Category Ranking by Avg Views ---")
print(cat_ranking)
conn.close()

# ------------------------------------------------------------------
# 4. VISUALIZATIONS
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('YouTube Trending Video Analytics Dashboard', fontsize=16, fontweight='bold')

# Category by avg views
ax1 = axes[0, 0]
cat_ranking_sorted = cat_ranking.sort_values('avg_views')
ax1.barh(cat_ranking_sorted['category'], cat_ranking_sorted['avg_views'], color='#e74c3c')
ax1.set_title('Average Views by Category')
ax1.set_xlabel('Average Views')

# Sentiment distribution
ax2 = axes[0, 1]
sent_counts = df['sentiment_label'].value_counts()
ax2.pie(sent_counts, labels=sent_counts.index, autopct='%1.1f%%',
        colors=['#2ecc71', '#95a5a6', '#e74c3c'])
ax2.set_title('Title Sentiment Distribution')

# Region-wise category heatmap
ax3 = axes[1, 0]
pivot = df.pivot_table(values='views', index='category', columns='region', aggfunc='mean')
sns.heatmap(pivot, ax=ax3, cmap='YlOrRd', fmt='.0f')
ax3.set_title('Avg Views: Category x Region')

# Publish hour distribution
ax4 = axes[1, 1]
hour_views = df.groupby('publish_hour')['views'].mean()
ax4.plot(hour_views.index, hour_views.values, marker='o', color='#3498db')
ax4.fill_between(hour_views.index, hour_views.values, alpha=0.2, color='#3498db')
ax4.set_title('Average Views by Publish Hour')
ax4.set_xlabel('Hour of Day')
ax4.set_ylabel('Avg Views')

plt.tight_layout()
plt.savefig('youtube_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nDashboard saved as youtube_dashboard.png")
