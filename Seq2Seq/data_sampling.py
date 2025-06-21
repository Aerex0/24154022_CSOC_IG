from sklearn.utils import resample
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('data.csv')




df['english'] = df['english'].astype(str).str.strip()
df['spanish'] = df['spanish'].astype(str).str.strip()


df['english_len'] = df['english'].apply(lambda x: len(x.split()))
df['spanish_len'] = df['spanish'].apply(lambda x: len(x.split()))

min_len = 5
max_len = 10


filtered_df = df[(df['english_len'] >= min_len) & (df['english_len'] <= max_len)]


groups = [filtered_df[filtered_df['english_len'] == i] for i in range(min_len, max_len + 1)]


max_size = max(len(g) for g in groups)

# Oversample each group to the max size
oversampled = [resample(g, replace=True, n_samples=max_size, random_state=42) for g in groups]

# Concatenate all oversampled groups
balanced_df = pd.concat(oversampled).sample(frac=1, random_state=42).reset_index(drop=True)


balanced_df[['english', 'spanish']].to_csv("balanced_data.csv", index=False)

# new shape and show length distribution
print("Balanced shape:", balanced_df.shape)

plt.figure(figsize=(10, 6))
plt.hist(balanced_df['english_len'], bins=np.arange(min_len - 0.5, max_len + 1.5, 1), rwidth=0.8)
plt.xlabel("Sentence Length")
plt.ylabel("Count")
plt.show()
