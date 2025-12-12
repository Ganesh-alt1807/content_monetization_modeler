"""Train and save a simple dummy model for the app.
Creates app/yt_revenue_model.pkl
"""
import random
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

R = 10000
np.random.seed(42)

views = np.random.randint(100, 1_000_000, size=R)
likes = (views * np.random.uniform(0.01, 0.1, size=R)).astype(int)
comments = (likes * np.random.uniform(0.01, 0.2, size=R)).astype(int)
watch_time_minutes = np.random.uniform(1, 20, size=R)
video_length_minutes = np.random.uniform(1, 30, size=R)
subscribers = np.random.randint(0, 200_000, size=R)
category = np.random.choice([0,1,2,3,4,5], size=R)
device = np.random.choice([0,1,2,3], size=R)
country = np.random.choice([0,1,2,3,4,5], size=R)

engagement_rate = (likes + comments) / np.where(views==0,1,views)
watch_efficiency = watch_time_minutes / np.where(video_length_minutes==0,1,video_length_minutes)

# Construct features
X = pd.DataFrame({
    'views': views,
    'likes': likes,
    'comments': comments,
    'watch_time_minutes': watch_time_minutes,
    'video_length_minutes': video_length_minutes,
    'subscribers': subscribers,
    'category': category,
    'device': device,
    'country': country,
    'engagement_rate': engagement_rate,
    'watch_efficiency': watch_efficiency,
})

# Very simple target function with noise
y = 0.00015 * views + 0.5 * engagement_rate + 0.0004 * subscribers + 0.8 * watch_efficiency + np.random.normal(0, 2.0, size=R)

model = LinearRegression()
model.fit(X, y)

out_path = Path(__file__).parent.parent / 'app' / 'yt_revenue_model.pkl'
joblib.dump(model, out_path)
print(f"Saved dummy model to {out_path}")
