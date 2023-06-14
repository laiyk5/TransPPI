import numpy as np
from sklearn import metrics

def calculate_max_f1_precision_recall(y_true, y_score):
    precisions, recalls, thredsholds = metrics.precision_recall_curve(y_true, y_score)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-20)
    idx = np.argmax(f1_scores)
    max_f1, max_pre, max_re =  f1_scores[idx], precisions[idx], recalls[idx]
    assert max_f1 == np.max(f1_scores)
    return max_f1, max_pre, max_re

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('input')
args = parser.parse_args()

data = np.loadtxt(args.input)
y_pred, y_true = data[:,0], data[:,1]

f1_score, precision, recall = calculate_max_f1_precision_recall(y_true, y_pred)

print(f1_score)
print(precision)
print(recall)
