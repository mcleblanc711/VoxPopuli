#!/usr/bin/env python3
"""Create synthetic Reddit data for testing."""

import pandas as pd
import random
from pathlib import Path

Path('data/reddit').mkdir(parents=True, exist_ok=True)

dates = pd.date_range('2023-01-01', '2023-12-31', freq='4h')
tickers = ['AAPL', 'TSLA', 'GME', 'AMD', 'NVDA']
data = []

for d in dates:
    for _ in range(random.randint(1, 3)):
        t = random.choice(tickers)
        bullish = random.random() > 0.4
        data.append({
            'post_id': f'p{len(data)}',
            'title': f'${t} looking great! Buy now!' if bullish else f'${t} is crashing hard',
            'selftext': 'This is financial advice' if random.random() > 0.7 else '',
            'score': random.randint(10, 500),
            'num_comments': random.randint(5, 200),
            'created_utc': d,
            'subreddit': random.choice(['wallstreetbets', 'stocks']),
        })

df = pd.DataFrame(data)
df.to_parquet('data/reddit/synthetic_2023.parquet')
print(f'Created {len(df)} synthetic posts in data/reddit/synthetic_2023.parquet')
