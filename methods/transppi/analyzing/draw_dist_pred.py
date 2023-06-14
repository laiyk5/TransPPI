import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import argparse

from utils.misc import split_by_label

parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
args = parser.parse_args()

data = np.loadtxt(args.input)
y_pred, y_true = data[:,0], data[:,1]

y_pred_pos, y_pred_neg = split_by_label(y_pred, y_true)

fig, ax = plt.subplots()
ax.set_xlabel("value")
ax.set_ylabel("density")

sns.kdeplot(y_pred, label='all', ax=ax)
sns.kdeplot(y_pred_pos, label='positive', ax=ax)
sns.kdeplot(y_pred_neg, label='negative', ax=ax)

ax.legend()

fig.savefig(args.output)
plt.close(fig)
