from mumin import MuminDataset
import numpy as np
import pandas as pd

with open("bearer_token.txt", "r") as f:
    bearer_token = f.read()
dataset = MuminDataset(bearer_token, size='large')
dataset.compile()

tweet_df = dataset.nodes['tweet']
tweet_df.dropna(inplace=True)
tweet_en_df = tweet_df[tweet_df["lang"] == "en"]
claim_df = dataset.nodes['claim']
discusses_df = dataset.rels[('tweet', 'discusses', 'claim')]
tweet_claim_df = (tweet_en_df.merge(discusses_df, left_index=True, right_on='src')
                          .merge(claim_df, left_on='tgt', right_index=True)
                          .reset_index(drop=True))

tweet_claim_misinfo = tweet_claim_df[tweet_claim_df["label"] == "misinformation"]
tweet_claim_factural = tweet_claim_df[tweet_claim_df["label"] == "factual"]
tweet_claim_sampled_misinfo = tweet_claim_misinfo.sample(n=tweet_claim_factural.shape[0])
tweet_claim_balanced = pd.concat([tweet_claim_factural, tweet_claim_sampled_misinfo])
tweet_dataset_df = tweet_claim_balanced[["text", "label"]]