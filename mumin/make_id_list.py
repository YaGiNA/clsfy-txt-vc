import pandas as pd

df = pd.read_csv("./mumin-large-equal.csv", index_col=0)
idxs = df["tweet_id"]
for i, val in idxs.items():
    with open("./ASVspoof2021.DF.cm.eval.trl.txt", "a") as f:
        f.write(str(val)+"\n")