from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('data.csv')
df['english'] = df['english'].astype(str).str.strip()
df['spanish'] = df['spanish'].astype(str).str.strip()

# Drop duplicates to avoid data leakage
df = df.drop_duplicates(subset=['english', 'spanish'])

# Compute lengths
df['english_len'] = df['english'].apply(lambda x: len(x.split()))
df['spanish_len'] = df['spanish'].apply(lambda x: len(x.split()))

# Filter by length
min_len = 5
max_len = 10
filtered_df = df[(df['english_len'] >= min_len) & (df['english_len'] <= max_len)]

# Split into train/val before oversampling
train_df, val_df = train_test_split(filtered_df, test_size=0.1, random_state=42)

# Oversample training set by  length
groups = [train_df[train_df['english_len'] == i] for i in range(min_len, max_len + 1)]
max_size = max(len(g) for g in groups)
oversampled = [resample(g, replace=True, n_samples=max_size, random_state=42) for g in groups]
balanced_train_df = pd.concat(oversampled).sample(frac=1, random_state=42).reset_index(drop=True)

# Save 
balanced_train_df[['english', 'spanish']].to_csv("balanced_train.csv", index=False)
val_df[['english', 'spanish']].to_csv("val.csv", index=False)

# Plotting...................................................................................
print("Balanced train shape:", balanced_train_df.shape)
print("Validation shape:", val_df.shape)

plt.figure(figsize=(10, 6))
plt.hist(balanced_train_df['english_len'], bins=np.arange(min_len - 0.5, max_len + 1.5, 1), alpha=0.6, label='Train')
plt.hist(val_df['english_len'], bins=np.arange(min_len - 0.5, max_len + 1.5, 1), alpha=0.6, label='Validation')
plt.xlabel("Sentence Length")
plt.ylabel("Count")
plt.legend()
plt.title("Length Distribution (Train vs Validation)")
plt.show()
