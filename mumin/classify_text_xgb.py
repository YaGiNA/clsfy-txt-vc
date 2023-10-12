from mumin import MuminDataset
import numpy as np
import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_curve, accuracy_score
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
from read_dataset import read_dataset, read_vctk_tsv


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
    Xs = [X_train, X_val, X_test]

    # Extract the labels
    y_train = np.array(pd.get_dummies(train.label)["misinformation"])
    y_val = np.array(pd.get_dummies(val.label)["misinformation"])
    y_test = np.array(pd.get_dummies(test.label)["misinformation"])
    ys = [y_train, y_val, y_test]
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
    test_preds_proba = model.predict_proba(X_test)[:, 1]
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
    return (train_preds, val_preds, test_preds, test_preds_proba)
    

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc


def calc_EER(y_target, normalized_vals):
    fpr, tpr, thresholds = roc_curve(y_target, normalized_vals, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return EER, eer_threshold


if __name__ == '__main__':
    # TODO Add other classifier
    got_Xs, got_ys, test_indexes = set_dataset()
    # used for vctk test set
    vctk_Xs = np.stack(read_vctk_tsv())
    """
    got_Xs[2] = vctk_Xs
    got_ys[2] = np.zeros(vctk_Xs.shape[0], dtype="uint8")

    got_Xs[2] = np.concatenate([got_Xs[2], vctk_Xs])
    got_ys[2] = np.concatenate([got_ys[2], np.zeros(vctk_Xs.shape[0], dtype="uint8")])
    """
    print('*** content only ***')
    xgb_preds = pred_xgb(got_Xs, got_ys)
    test_scores, threshold_EER = calc_EER(got_ys[2], xgb_preds[2])
    print(f'EER: {100 * test_scores:.2f}%')
    print("accuracy:", accuracy_score(got_ys[2], xgb_preds[2]))
    # """
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
    print('*** waveform only ***')
    test_scores, threshold_EER = calc_EER(got_ys[2], normalized_rawnet2_test_preds)
    print(f'EER: {100 * test_scores:.2f}%')
    preds = [1 if val > threshold_EER else 0 for val in normalized_rawnet2_test_preds]
    print("accuracy:", accuracy_score(got_ys[2], preds))
    
    combined_vals = np.array([(rawnet2 + xgb) / 2 for rawnet2, xgb in zip(normalized_rawnet2_test_preds, xgb_preds[-1])])
    test_scores, threshold_EER = calc_EER(got_ys[2], combined_vals)
    preds = [1 if val > threshold_EER else 0 for val in combined_vals]
    print('*** combined ***')
    print(f'EER: {100 * test_scores:.2f}%')
    print("accuracy:", accuracy_score(got_ys[2], preds))
    # """
    """
    print(f'EER: {100 * xgb_scores[2]:.2f}%')
    val_fake = [val for val, y in zip(combined_vals, got_ys[2]) if y > 0]
    print(val_fake[:10])
    print(len([val for val in val_fake if val > threshold_EER])/len(val_fake))
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