from mumin import MuminDataset
import numpy as np
import pandas as pd
import re

with open("bearer_token.txt", "r") as f:
    bearer_token = f.read()

def read_dataset(token, en_only=False):
    dataset = MuminDataset(token, size='large')
    dataset.compile()

    tweet_df = dataset.nodes['tweet']
    tweet_df.dropna(inplace=True)
    tweet_en_df = tweet_df[tweet_df["lang"] == "en"]
    claim_df = dataset.nodes['claim']
    discusses_df = dataset.rels[('tweet', 'discusses', 'claim')]
    tweet_claim_df = (tweet_en_df.merge(discusses_df, left_index=True, right_on='src')
                            .merge(claim_df, left_on='tgt', right_index=True))
    if en_only:
        tweet_claim_misinfo = tweet_claim_df[tweet_claim_df["label"] == "misinformation"]
        tweet_claim_factural = tweet_claim_df[tweet_claim_df["label"] == "factual"]
        tweet_claim_sampled_misinfo = tweet_claim_misinfo.sample(n=tweet_claim_factural.shape[0])
        tweet_claim_balanced = pd.concat([tweet_claim_factural, tweet_claim_sampled_misinfo])
        return tweet_claim_balanced
        # tweet_dataset_df = tweet_claim_balanced[["text", "label"]].replace(r'\n',' ', regex=True)
        # tweet_dataset_df = tweet_claim_balanced.replace(r'\n',' ', regex=True)
        # return tweet_dataset_df
    return tweet_claim_df

def parse_tsv_line(raw_line):
    str_emb = raw_line[1:-4]
    emb = np.fromstring(str_emb, sep=",")
    return emb


def read_vctk_tsv():
    with open("./vctk_spoken_emb.tsv", "r") as f:
        vctk_embs = [parse_tsv_line(line) for line in f]
    return vctk_embs


if __name__ == '__main__':
    tweet_dataset_df = read_dataset(bearer_token, en_only=True)
    tweet_dataset_df["text"] = tweet_dataset_df["text"].apply(lambda x: re.split('https:\/\/.*', str(x))[0]).replace(r'\n',' ', regex=True)
    tweet_dataset_df_nodup = tweet_dataset_df.drop_duplicates(subset=['text'])
    tweet_dataset_df_nodup[["tweet_id", "text", "label"]].to_csv("mumin-large-equal.csv")