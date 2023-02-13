from mumin import MuminDataset
import numpy as np
import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from read_dataset import read_dataset

with open("bearer_token.txt", "r") as f:
    bearer_token = f.read()

tweet_dataset_df = read_dataset(bearer_token)
train, val_test = train_test_split(tweet_dataset_df, test_size=0.3, random_state=42, shuffle=True)
test, val = train_test_split(val_test, test_size=1/3, random_state=42, shuffle=True)


# Extract the tweet embeddings
X_train = np.stack(train.text_emb)
X_val = np.stack(val.text_emb)
X_test = np.stack(test.text_emb)

# Extract the labels
y_train = pd.get_dummies(train.label)["misinformation"]
y_val = pd.get_dummies(val.label)["misinformation"]
y_test = pd.get_dummies(test.label)["misinformation"]


# Initialise the model
model = XGBClassifier()

# Train the model
model.fit(X_train, y_train)

# Get predictions
train_preds = model.predict(X_train)
val_preds = model.predict(X_val)
test_preds = model.predict(X_test)

# Compute macro-average F1-score
train_scores = f1_score(y_train, train_preds, average=None)
val_scores = f1_score(y_val, val_preds, average=None)
test_scores = f1_score(y_test, test_preds, average=None)

print('*** Training scores ***')
print(f'Misinformation F1: {100 * train_scores[1]:.2f}%')
print(f'Factual F1: {100 * train_scores[0]:.2f}%')
print(f'Macro-average F1: {100 * train_scores.mean():.2f}%\n')

print('*** Validation scores ***')
print(f'Misinformation F1: {100 * val_scores[1]:.2f}%')
print(f'Factual F1: {100 * val_scores[0]:.2f}%')
print(f'Macro-average F1: {100 * val_scores.mean():.2f}%\n')

print('*** Test scores ***')
print(f'Misinformation F1: {100 * test_scores[1]:.2f}%')
print(f'Factual F1: {100 * test_scores[0]:.2f}%')
print(f'Macro-average F1: {100 * test_scores.mean():.2f}%')