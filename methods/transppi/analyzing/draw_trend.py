import numpy as np

import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
args = parser.parse_args()

fig, ax = plt.subplots(layout='constrained')
data = np.loadtxt(args.input)
ax.plot(data)
ax.set_xlabel('epoch')
ax.set_ylabel('value')

fig.savefig(args.output)
