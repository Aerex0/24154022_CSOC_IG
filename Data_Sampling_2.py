import pandas as pd

train_df = pd.read_csv("train.csv", header=None, names=['polarity','title','review'])
test_df = pd.read_csv("test.csv", header=None, names=['polarity','title','review'])


train_df = train_df.drop('review', axis=1)
test_df = test_df.drop('review', axis=1)

# Shuffling the training data
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Saving the modified data
train_df.to_csv("Train_without_review.csv", index=True)
test_df.to_csv("Test_without_review.csv", index=True)
