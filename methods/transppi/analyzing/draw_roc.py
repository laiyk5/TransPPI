'''
Input: A txt file with multiple lines, each line correspond to one example in the training/test dataset. the first column is prediction of the model, the second column is the ground true value of the example.
'''

import os
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

def draw_roc(y_pred, y_true, outpath):
    
    precision, recall, _ = metrics.roc_curve(y_true, y_pred)
    
    fig, ax = plt.subplots(layout='constrained')
    
    ax.set_title('Precision Recall Curve')
    ax.set_xlabel('Precision')
    ax.set_ylabel('Recall')
    
    ax.plot(precision, recall)
    
    fig.savefig(outpath)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
parser.add_argument('output', type=str)
args = parser.parse_args()

data = np.loadtxt(args.input)
y_pred = data[:,0]
y_true = data[:,1]

draw_roc(y_pred, y_true, args.output)
