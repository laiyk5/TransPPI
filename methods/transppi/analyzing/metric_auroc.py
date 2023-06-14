'''
Input:
    : Prediction of a model and the ground true value on a training/test set.
Output:
    : print the metric to stdout
'''

from sklearn import metrics
import numpy as np
import argparse

def calculate_auroc(y_true, y_pred):
    fpr, tpr, thredsholds = metrics.roc_curve(y_true, y_pred)
    auroc = metrics.auc(fpr, tpr)
    return auroc

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
args = parser.parse_args()

data = np.loadtxt(args.input)
y_pred = data[:,0]
y_true = data[:,1]

auroc = calculate_auroc(y_true, y_pred)

print(auroc)
