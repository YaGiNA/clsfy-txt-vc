from mumin import MuminDataset
import numpy as np
import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_curve
from sklearn.preprocessing import normalize
from xgboost import XGBClassifier
from statistics import mean

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.optim import Adam

from tqdm import tqdm
from read_dataset import read_dataset


class TweetEmbeddingClassifyModel(nn.Module):
    def __init__(self, in_channels, drop_rate) :
        super(TweetEmbeddingClassifyModel, self).__init__()

        self.drp1 = nn.Dropout(p=drop_rate)
        self.line = nn.Linear(in_channels, 2)
        self.drp2 = nn.Dropout(p=drop_rate)
        self.smax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.drp1(x)
        x = self.line(x)
        x = self.drp2(x)
        x = self.smax(x)
        return x


with open("bearer_token.txt", "r") as f:
    bearer_token = f.read()

def set_dataset():
    target_df = pd.read_csv("./mumin-large-equal.csv", index_col=0)
    all_tweet_dataset_df = read_dataset(bearer_token, en_only=False)
    tweet_dataset_df = all_tweet_dataset_df[all_tweet_dataset_df["tweet_id"].isin(target_df["tweet_id"])]
    train, val_test = train_test_split(tweet_dataset_df, test_size=0.3, random_state=42, shuffle=True)
    test, val = train_test_split(val_test, test_size=1/3, random_state=42, shuffle=True)
    test_idxs = test["tweet_id"].tolist()
    # Extract the tweet embeddings
    X_train = np.stack(train.text_emb)
    X_val = np.stack(val.text_emb)
    X_test = np.stack(test.text_emb)
    Xs = (X_train, X_val, X_test)

    # Extract the labels
    y_train = np.array(pd.get_dummies(train.label)["misinformation"])
    y_val = np.array(pd.get_dummies(val.label)["misinformation"])
    y_test = np.array(pd.get_dummies(test.label)["misinformation"])
    ys = (y_train, y_val, y_test)
    return Xs, ys, test_idxs
# Initialise the model

def pred_xgb(Xs, ys):
    X_train, X_val, X_test = Xs
    y_train, y_val, y_test = ys
    model = XGBClassifier()

    # Train the model
    model.fit(X_train, y_train)

    # Get predictions
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)
    """
    # Compute macro-average F1-score
    train_scores = f1_score(y_train, train_preds, average=None)
    val_scores = f1_score(y_val, val_preds, average=None)
    test_scores = f1_score(y_test, test_preds, average=None)
    """
    """
    train_scores = calc_EER(y_train, train_preds)
    val_scores = calc_EER(y_val, val_preds)
    test_scores = calc_EER(y_test, test_preds)
    return (train_scores, val_scores, test_scores)
    """
    return (train_preds, val_preds, test_preds)
    

def pred_emb_dense(Xs, ys):
    BATCH_SIZE = 64 # 適宜変更
    EPOCHS = 100 # 適宜変更
    X_train, X_val, X_test = Xs
    y_train, y_val, y_test = ys
    train_set = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_set = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_set = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_dataloader = DataLoader(
        train_set, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=2, 
        drop_last=True,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_set, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=2, 
        drop_last=True,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_set, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=2, 
        drop_last=False,
        pin_memory=True
    )


    model = TweetEmbeddingClassifyModel(in_channels=X_train.shape[1], drop_rate=0.2)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(DEVICE)
    optimizer = Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    for epoch in tqdm(range(EPOCHS)):
        # 学習
        model.train()
        train_loss = 0
        for batch, label in train_dataloader:

            for param in model.parameters():
                param.grad = None

            batch = batch.float()

            batch = batch.to(DEVICE)
            label = label.to(DEVICE)

            preds = model(batch)
            loss = criterion(preds, label)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        valid_loss = 0
        with torch.inference_mode():

            for batch, label in val_dataloader:

                batch = batch.float()

                batch = batch.to(DEVICE)
                label = label.to(DEVICE)

                preds = model(batch)

                loss = criterion(preds, label)
                valid_loss += loss.item()

        valid_loss /= len(val_dataloader)
        if not epoch % 10:
            print(epoch, "Loss:", valid_loss)

    model.eval()
    preds_list = []

    with torch.no_grad():
        for news in tqdm(test_dataloader):

            for param in model.parameters():
                param.grad = None

            news = news[0]
            news = news.to(DEVICE)

            preds = model(news)[:, 1]
            preds_list = preds_list + preds.tolist()
    
    return preds_list

def calc_EER(y_target, normalized_vals):
    fpr, tpr, thresholds = roc_curve(y_target, normalized_vals, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return EER#, eer_threshold


if __name__ == '__main__':
    # TODO Add other classifier
    got_Xs, got_ys, test_indexes = set_dataset()
    emb_dense_preds = pred_emb_dense(got_Xs, got_ys)
    """
    xgb_preds = pred_xgb(got_Xs, got_ys)
    xgb_test_preds = xgb_preds[2]
    """

    rawnet2_preds = {}
    with open("./pre_trained_eval_CM_scores-mumin.txt") as f:
        for line in f:
            idx, val = line.split()
            rawnet2_preds[int(idx)] = float(val)
    rawnet2_test_vals = []
    for test_idx in test_indexes:
        try:
            rawnet2_test_vals.append(rawnet2_preds[int(test_idx)])
        except KeyError:
            print(test_idx, "not in rawnet2 results")
    normalized_rawnet2_test_preds = np.array([(x-min(rawnet2_test_vals))/(max(rawnet2_test_vals)-min(rawnet2_test_vals)) for x in rawnet2_test_vals])
    combined_vals = np.array([(rawnet2 + emb) / 2 for rawnet2, emb in zip(normalized_rawnet2_test_preds, emb_dense_preds)])
    # combined_vals = np.array([(rawnet2 + xgb) / 2 for rawnet2, xgb in zip(normalized_rawnet2_test_preds, xgb_test_preds)])
    test_scores = calc_EER(got_ys[2], combined_vals)
    print('*** Test scores ***')
    print(f'EER: {100 * test_scores:.2f}%')
    # print(f'EER: {100 * xgb_scores[2]:.2f}%')
"""
    print('*** Training scores ***')
    print(f'Misinformation F1: {100 * xgb_scores[0][1]:.2f}%')
    print(f'Factual F1: {100 * xgb_scores[0][0]:.2f}%')
    print(f'Macro-average F1: {100 * xgb_scores[0].mean():.2f}%\n')

    print('*** Validation scores ***')
    print(f'Misinformation F1: {100 * xgb_scores[1][1]:.2f}%')
    print(f'Factual F1: {100 * xgb_scores[1][0]:.2f}%')
    print(f'Macro-average F1: {100 * xgb_scores[1].mean():.2f}%\n')

    print('*** Test scores ***')
    print(f'Misinformation F1: {100 * xgb_scores[2][1]:.2f}%')
    print(f'Factual F1: {100 * xgb_scores[2][0]:.2f}%')
    print(f'Macro-average F1: {100 * xgb_scores[2].mean():.2f}%')
"""