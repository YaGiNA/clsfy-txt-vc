import numpy as np
import pandas as pd
from sklearn import metrics

vals = []
with open("/home/y-yanagi/ASVspoof/2021/DF/Baseline-RawNet2/pre_trained_eval_CM_scores-mumin.txt", "r") as f:
    for line in f.readlines():
        vals.append(float(line.split()[1]))

normalized_vals = np.array([(x-min(vals))/(max(vals)-min(vals)) for x in vals])

df = pd.read_csv("~/data/trial_tts/vits/mumin-large-equal.csv", index_col=0)
y_true = pd.get_dummies(df["label"])["misinformation"]
y_ordered = []
with open ("/home/y-yanagi/ASVspoof/2021/DF/Baseline-RawNet2/database/ASVspoof_DF_cm_protocols/ASVspoof2021.DF.cm.eval.trl.txt", "r") as f:
    for line in f:
        y_ordered.append(y_true.get(int(line.rstrip())))

y_target = np.array(y_ordered)
fpr, tpr, thresholds = metrics.roc_curve(y_target, normalized_vals, pos_label=1)
fnr = 1 - tpr
eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
print(EER)