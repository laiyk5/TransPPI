from sklearn import metrics
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import h5py

import os

def draw_roc(y_true, y_score, ax):
    fpr, tpr, thredsholds = metrics.roc_curve(y_true, y_score)
    ax.plot(fpr, tpr)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(-0.1, 1.1)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    return

def draw_prc(y_true, y_score, ax):
    precisions, recalls, thredsholds = metrics.precision_recall_curve(y_true, y_score)
    ax.plot(recalls, precisions)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(-0.1, 1.1)
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    return

def draw_distribution(y_true, y_score, label, ax):
    y_score_neg = [y_score[i] for i in 
                    filter(lambda i : y_true[i] == label, range(len(y_score)))]
    sns.histplot(y_score_neg, kde=True, ax=ax)
    ax.set_xlim(0, 1)
    ax.set_ylabel('count')
    ax.set_xlabel('score')

def draw_density(y_true, y_score, ax):
    y_score_neg = [y_score[i] for i in 
                    filter(lambda i : y_true[i] == 0, range(len(y_score)))]
    y_score_pos = [y_score[i] for i in 
                    filter(lambda i : y_true[i] == 1, range(len(y_score)))]
    sns.kdeplot(y_score_neg, color='red', ax=ax)
    sns.kdeplot(y_score_pos, color='green', ax=ax)
    ax.set_xlim(0, 1)
    ax.set_ylabel('density')
    ax.set_xlabel('score')

def cal_auroc(y_true, y_score):
    fpr, tpr, thredsholds = metrics.roc_curve(y_true, y_score)
    auroc = metrics.auc(fpr, tpr)
    return auroc

def cal_auprc(y_true, y_score):
    precisions, recalls, thredsholds = metrics.precision_recall_curve(y_true, y_score)
    auprc = metrics.auc(recalls, precisions)
    return auprc

def cal_max_f1_precision_recall(y_true, y_score):
    precisions, recalls, thredsholds = metrics.precision_recall_curve(y_true, y_score)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-20)
    idx = np.argmax(f1_scores)
    max_f1, max_pre, max_re =  f1_scores[idx], precisions[idx], recalls[idx]
    assert max_f1 == np.max(f1_scores)
    return max_f1, max_pre, max_re


def draw_and_save_kde(y_true, y_score, out_dir):
    fig, ax = plt.subplots(constrained_layout=True)
    draw_density(y_true, y_score, ax=ax)
    fig.savefig(os.path.join(out_dir, 'kde.png'), dpi=200)
    plt.close(fig)

def draw_and_save_hist(y_true, y_score, out_dir):
    fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
    draw_distribution(y_true, y_score, 0, ax[0])
    ax[0].set_title('Label = 0')
    draw_distribution(y_true, y_score, 1, ax[1])
    ax[1].set_title('Label = 1')
    fig.savefig(os.path.join(out_dir, 'dist.png'), dpi=200)
    plt.close(fig)

def draw_and_save_trend(y, x_label, y_label, out_dir, ylim=True):
    fig, ax = plt.subplots(constrained_layout=True)
    sns.lineplot(y, ax=ax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if ylim:
        ax.set_ylim(bottom=-0.1)
    fig.savefig(os.path.join(out_dir, y_label + '.png'))
    plt.close(fig)

def draw_and_save_roc_prc(y_true, y_score, out_dir):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), constrained_layout=True)
    draw_roc(y_true, y_score, ax=axes[0])
    draw_prc(y_true, y_score, ax=axes[1])
    fig.savefig(os.path.join(out_dir, 'prc_roc.png'), dpi=200)
    plt.close(fig)

class Logger:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        
        self.metric_logfile = open(os.path.join(self.out_dir, 'metrics.log'), 'w')
        self.resfile = h5py.File(os.path.join(self.out_dir, 'resfile.hdf5'), 'w')

        self.auroc = []
        self.auprc = []
        self.precision = []
        self.recall = []
        self.f1_score = []
        self.loss = []
        self.loss_avg = []
        self.lr = []



    def step_metrics(self, epoch, y_true, y_score):
        out_dir = os.path.join(self.out_dir, str(epoch))
        os.makedirs(out_dir, exist_ok=True)

        epoch_prefix = f'{epoch}_'
        self.resfile.create_dataset(epoch_prefix + 'y_true', data=y_true)
        self.resfile.create_dataset(epoch_prefix + 'y_score', data=y_score)

        draw_and_save_roc_prc(y_true, y_score, out_dir)
        draw_and_save_kde(y_true, y_score, out_dir)
        draw_and_save_hist(y_true, y_score, out_dir)

        # save log auroc, auprc, precision, recall, f1_score
        auroc = cal_auroc(y_true, y_score)
        auprc = cal_auprc(y_true, y_score)
        f1_score, precision, recall = cal_max_f1_precision_recall(y_true, y_score)

        self.metric_logfile.write('[EPOCH_' + epoch_prefix + ']' + f"auroc={auroc}, auprc={auprc}, precision={precision}, recall={recall}, f1_scores={f1_score}\n")
        self.metric_logfile.flush()

        # Update Trend
        self.auroc.append(auroc)
        self.auprc.append(auprc)
        self.precision.append(precision)
        self.recall.append(recall)
        self.f1_score.append(f1_score)

        # auroc, auprc
        draw_and_save_trend(self.auroc, 'epoch', 'auroc', self.out_dir)
        draw_and_save_trend(self.auprc, 'epoch', 'auprc', self.out_dir)
        draw_and_save_trend(self.precision, 'epoch', 'precision', self.out_dir)
        draw_and_save_trend(self.recall, 'epoch', 'recall', self.out_dir)
        draw_and_save_trend(self.f1_score, 'epoch', 'f1_score', self.out_dir)

    def append_loss(self, loss):
        self.loss += loss
        draw_and_save_trend(self.loss, 'batch', 'loss', self.out_dir)

    def append_loss_avg(self, loss_avg):
        self.loss_avg.append(loss_avg)
        draw_and_save_trend(self.loss_avg, 'epoch', 'loss_avg', self.out_dir)

    def append_lr(self, lr):
        self.lr += lr
        draw_and_save_trend(self.lr, 'batch', 'lr', self.out_dir, ylim=False)