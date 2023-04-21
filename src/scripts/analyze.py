import matplotlib as mpl
from matplotlib import pyplot as plt
import h5py
import argparse
import re
import numpy as np
import os
from sklearn import metrics

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('resfile', type=str)
    parser.add_argument('out_dir', type=str)
    return parser


def draw_loss(file, fig, batch, epochs):
    fig, ax = plt.subplots()
    print(file.keys())
    for epoch in range(epochs):
        loss = file[f'{batch}_{epoch}_train_loss']
        loss_x = np.arange(len(loss.size[0]))
        y_true = file[f'{batch}_{epoch}_train_y_true']
        ax = fig.add_subplot()
        ax.scatter(x=loss_x, y=loss, edgecolors=y_true)
        fig.show()
        os.system('pause')


def get_num_epochs(file, fig):
    pattern = re.compile(r'(?P<batch>[0-9]+)_(?P<epoch>[0-9]+)_validate_y_score')
    num_epochs = 0
    for key in file.keys():
        m = pattern.match(key)
        if not m or m['batch'] != '0':
            continue
        num_epochs += 1
    return num_epochs


def get_metrics_one_epoch(y_true, y_score):
    precisions, recalls, _ = metrics.precision_recall_curve(y_true=y_true, probas_pred=y_score)
    fpr, tpr, _ = metrics.roc_curve(y_true=y_true, y_score=y_score)
    auprc = metrics.auc(recalls, precisions)
    auroc = metrics.auc(fpr, tpr)

    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-20)
    idx = np.argmax(f1_scores)
    f1_score = f1_scores[idx]
    precision = precisions[idx]
    recall = recalls[idx]

    return auroc, auprc, precision, recall, f1_score


def get_metrics_one_batch(y_trues, y_scores):
    aurocs, auprcs, precisions, recalls, f1_scores = [], [], [], [], []
    for y_true, y_score in zip(y_trues, y_scores):
        auroc, auprc, precision, recall, f1_score = get_metrics_one_epoch(y_true, y_score)
        aurocs.append(auroc)
        auprcs.append(auprc)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
    return aurocs, auprcs, precisions, recalls, f1_scores


def draw_metrics(file, epochs, out_dir):
    aurocs, auprcs, precisions, recalls, f1_scores = [], [], [], [], []
    for epoch in range(epochs):
        validate_y_true = file[f'0_{epoch}_y_true']
        validate_y_score = file[f'0_{epoch}_y_score']
        auroc, auprc, precision, recall, f1_score = get_metrics_one_epoch(validate_y_true, validate_y_score)
        aurocs.append(auroc)
        auprcs.append(auprc)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
    
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(np.arange(len(aurocs)), aurocs)
    fig.savefig(os.path.join(out_dir, 'auroc.png'))
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(np.arange(len(auprcs)), auprcs)
    fig.savefig(os.path.join(out_dir, 'auprc.png'))
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(np.arange(len(precisions)), precisions)
    fig.savefig(os.path.join(out_dir, 'precision.png'))
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(np.arange(len(recalls)), recalls)
    fig.savefig(os.path.join(out_dir, 'recall.png'))
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(np.arange(len(f1_scores)), f1_scores)
    fig.savefig(os.path.join(out_dir, 'f1_score.png'))
    plt.close(fig)


if __name__ == '__main__':
    parser = create_argparser()
    args = parser.parse_args()
    os.makedirs(args.out_dir)
    file = h5py.File(args.resfile, 'r')
    draw_metrics(file, 30, args.out_dir)
    