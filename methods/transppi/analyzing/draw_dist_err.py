import matplotlib.pyplot as plt
import torch
from torch.nn import BCELoss
import numpy as np
from utils.misc import split_by_label
import seaborn as sns

def draw_kde_loss(y_pred, y_true, outpath):
    '''
    draw distribution of the loss
    '''    

    loss_fn = BCELoss(reduction='none')
    losses = loss_fn(torch.tensor(y_pred), torch.tensor(y_true)).tolist()

    losses_pos, losses_neg = split_by_label(losses, y_true)

    fig, ax = plt.subplots()
    ax.set_title("loss distribution")
    ax.set_xlabel("loss")
    ax.set_ylabel("density")

    sns.kdeplot(losses, label='all', ax=ax)
    sns.kdeplot(losses_pos, label='positive', ax=ax)
    sns.kdeplot(losses_neg, label='negative', ax=ax)

    ax.legend()

    fig.savefig(outpath)
    plt.close(fig)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
args = parser.parse_args()

data = np.loadtxt(args.input)
y_pred, y_true = data[:,0], data[:,1]

draw_kde_loss(y_pred, y_true, args.output)
