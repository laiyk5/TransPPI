import torch
from torch.nn import BCELoss
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("input")
args = parser.parse_args()

data = np.loadtxt(args.input)
y_pred, y_true = data[:,0], data[:,1]

loss_fn = BCELoss(reduction='mean')
loss = loss_fn(torch.tensor(y_pred), torch.tensor(y_true)).item()

print(loss)
