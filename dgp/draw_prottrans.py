import seaborn as sns
import matplotlib.pyplot as plt
import argparse

import h5py

import numpy as np

import random

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input')
    parser.add_argument('-o', '--output')
    return parser

def main(args):
    random.seed(0)

    dataset = h5py.File(args.input)
    
    to_draw = []
    
    items = random.sample(list(dataset.items()), 16)
    ylabels = []

    for key, value in items:
        ylabels.append(key)
        to_draw.append(  np.array(random.sample(list(np.average(value, axis=0)), 32) ))
    
    to_draw = np.stack(to_draw)
    # print(to_draw)

    fig, ax = plt.subplots()
    sns.heatmap(data=to_draw, vmin=0, vmax=1, ax=ax)
    ax.set_xlabel('features')
    ax.set_ylabel('proteins')
    print(ylabels)
    print(len(ylabels))
    ax.set_yticks(ticks=list(range(0, len(ylabels))), labels=ylabels, rotation=45)
    # ax.set_xticklabels(ylabels)
    fig.savefig(args.output, dpi=200)

if __name__ == '__main__':
    parser = create_argparser()
    args = parser.parse_args()
    main(args)