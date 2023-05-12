
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import numpy as np

from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

import h5py

import os
import random

class Logger:
    def __init__(self, out_dir, fold, prefix):
        self.out_dir = os.path.join(out_dir, f'{fold}')
        os.makedirs(self.out_dir, exist_ok=True)
        self.prefix = prefix
        
        self.metric_logfile = open(os.path.join(self.out_dir, prefix + 'metrics.log'), 'w')
        self.resfile = h5py.File(os.path.join(self.out_dir, prefix + 'resfile.hdf5'), 'w')

        self.auroc = []
        self.auprc = []
        self.precision = []
        self.recall = []
        self.f1_score = []
        self.loss = []

    # draw destribution
    def _draw_distribution(self, y_true, y_score, ax):
        y_score_neg = [y_score[i] for i in 
                        filter(lambda i : y_true[i] == 0, range(len(y_score)))]
        y_score_pos = [y_score[i] for i in 
                        filter(lambda i : y_true[i] == 1, range(len(y_score)))]
        sns.histplot(y_score_neg, color='red', kde=True, ax=ax)
        sns.histplot(y_score_pos, color='green', kde=True, ax=ax)
        ax.set_ylabel('count')
        ax.set_xlabel('score')
    
    def _draw_density(self, y_true, y_score, ax):
        y_score_neg = [y_score[i] for i in 
                        filter(lambda i : y_true[i] == 0, range(len(y_score)))]
        y_score_pos = [y_score[i] for i in 
                        filter(lambda i : y_true[i] == 1, range(len(y_score)))]
        sns.kdeplot(y_score_neg, color='red', ax=ax)
        sns.kdeplot(y_score_pos, color='green', ax=ax)
        ax.set_ylabel('density')
        ax.set_xlabel('score')

    def step_metrics(self, epoch, y_true, y_score):
        prefix_epoch = self.prefix + f'{epoch}_'

        self.resfile.create_dataset(prefix_epoch + 'y_true', data=y_true)
        self.resfile.create_dataset(prefix_epoch + 'y_score', data=y_score)

        fpr, tpr, thredsholds = metrics.roc_curve(y_true, y_score)
        precisions, recalls, thredsholds = metrics.precision_recall_curve(y_true, y_score)

        # draw prc roc
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), constrained_layout=True)
        axes[0].plot(recalls, precisions)
        axes[0].set_xlabel('recall')
        axes[0].set_ylabel('precision')
        axes[1].plot(fpr, tpr)
        axes[1].set_xlabel('FPR')
        axes[1].set_ylabel('TPR')
        fig.savefig(os.path.join(self.out_dir, prefix_epoch + 'prc_roc.png'), dpi=200)
        plt.close(fig)

        # draw kde
        fig, ax = plt.subplots(figsize=(5,5), constrained_layout=True)
        self._draw_density(y_true, y_score, ax=ax)
        fig.savefig(os.path.join(self.out_dir, prefix_epoch + 'kde.png'), dpi=200)
        plt.close(fig)

        # draw dist
        fig, ax = plt.subplots(figsize=(5,5), constrained_layout=True)
        self._draw_distribution(y_true, y_score, ax=ax)
        fig.savefig(os.path.join(self.out_dir, prefix_epoch + 'dist.png'), dpi=200)
        plt.close(fig)

        # save log auroc, auprc, precision, recall, f1_score
        auroc = metrics.auc(fpr, tpr)
        auprc = metrics.auc(recalls, precisions)
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-20)
        idx = np.argmax(f1_scores)
        precision, recall, f1_scores = precisions[idx], recalls[idx], f1_scores[idx]

        self.metric_logfile.write('[' + prefix_epoch + ']' + f"auroc={auroc}, auprc={auprc}, precision={precision}, recall={recall}, f1_scores={f1_scores}\n")
        self.metric_logfile.flush()

        # Update Trend
        self.auroc.append(auroc)
        self.auprc.append(auprc)
        self.precision.append(precision)
        self.recall.append(recall)
        self.f1_score.append(f1_scores)

        # auroc, auprc
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), constrained_layout=True)
        axes[0].set_xlabel('epoch')
        axes[0].set_ylabel('auroc')
        sns.lineplot(self.auroc, ax=axes[0])
        axes[1].set_xlabel('epoch')
        axes[1].set_ylabel('auprc')
        sns.lineplot(self.auprc, ax=axes[1])
        fig.savefig(os.path.join(self.out_dir, self.prefix + 'auroc_auprc.png'), dpi=200)
        plt.close(fig)

        # f1_score, precision, recall
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), constrained_layout=True)

        axes[0].set_xlabel('epoch')
        axes[1].set_xlabel('epoch')
        axes[2].set_xlabel('epoch')
        
        axes[0].set_ylabel('f1 score')
        axes[1].set_ylabel('precision')
        axes[2].set_ylabel('recall')

        sns.lineplot(self.f1_score, ax=axes[0])
        sns.lineplot(self.precision, ax=axes[1])
        sns.lineplot(self.recall, ax=axes[2])

        fig.savefig(os.path.join(self.out_dir, self.prefix + 'f1-precision-recall.png'), dpi=200)
        plt.close(fig)

    def step_loss(self, epoch, loss):
        prefix_epoch = self.prefix + f'{epoch}_'
        self.resfile.create_dataset(prefix_epoch + 'loss', data=loss)

        self.loss += loss
        fig, ax = plt.subplots()
        loss_mean = np.mean(loss)
        loss_var = np.var(loss)
        sns.lineplot(self.loss, ax=ax)
        ax.text(x=10, y=10, s=f"mean={loss_mean}, var={loss_var}")
        ax.set_ylabel('loss')
        ax.set_xlabel('batch')
        fig.savefig(os.path.join(self.out_dir, self.prefix + f'loss.png'), dpi=200)
        plt.close(fig)


class Factory:
    def new_model_scheduler_optimizer(self):
        raise NotImplementedError

    def new_dataloader(self):
        raise NotImplementedError
    
    def new_loss_func(self):
        raise NotImplementedError


def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    return

class Trainer:
    def __init__(self, out_dir, epochs, folds, factory, device):
        self.out_dir = out_dir
        self.epochs = epochs
        self.folds = folds
        self.factory = factory
        self.device = device

    def train_kfold(self, dataset):
        random.shuffle(dataset)
        self.ppi_dataset_y = [label for _, _, label in dataset]
        skf = StratifiedKFold(n_splits=10, shuffle=False)
        for fold, (train_index, test_index) in enumerate(skf.split(np.zeros(len(self.ppi_dataset_y)), self.ppi_dataset_y)):
            train_ppi_dataset = [dataset[index] for index in train_index]
            random.shuffle(train_ppi_dataset)
            validate_ppi_dataset = [dataset[index] for index in test_index]
            
            train_dataloader = self.factory.new_dataloader(train_ppi_dataset)
            validate_dataloader = self.factory.new_dataloader(validate_ppi_dataset)

            self.__train_fold(fold, train_dataloader, validate_dataloader)


    def __train_fold(self, fold, train_dataloader, validate_dataloader):
        model, optimizer, scheduler = self.factory.new_model_scheduler_optimizer()
        model = model.to(self.device)
    
        train_logger = Logger(self.out_dir, fold=fold, prefix='train_')
        validate_logger = Logger(self.out_dir, fold=fold, prefix='validate_')

        for epoch in range(self.epochs):
            self.__train_and_log(model, train_dataloader, optimizer, scheduler, train_logger, epoch)
            self.__validate_and_log(model, validate_dataloader, validate_logger, epoch)
            if scheduler is not None:
                scheduler.step()
            torch.save(model, os.path.join(self.out_dir, f'{fold}_{epoch}_model.pth'))


    def __train_and_log(self, model, dataloader, optimizer, scheduler, logger:Logger, epoch):
        y_true, y_score, loss = self.__train(model, dataloader, optimizer, scheduler)
        logger.step_metrics(epoch, y_true, y_score)
        logger.step_loss(epoch, loss)

    def __validate_and_log(self, model, dataloader, logger, epoch):
        y_true, y_score = self.__validate(model, dataloader)
        logger.step_metrics(epoch, y_true, y_score)
        auroc = metrics.roc_auc_score(y_true, y_score)
        return auroc

    def __train(self, model: torch.nn.Module, dataloader: DataLoader, optimizer:torch.optim.Optimizer, scheduler):
        model.train()
        loss_func = self.factory.new_loss_func().to(self.device)

        progress_bar = tqdm(dataloader)
        
        all_y_true, all_y_score, all_loss = [], [], []
        loss_avg = 0
        for i, data in enumerate(progress_bar):
            torch.cuda.empty_cache()

            data_gpu = map(lambda x : x.to(self.device), data[:-1])
            y_true = data[-1]

            y_true_gpu = y_true.unsqueeze(-1).to(self.device)

            optimizer.zero_grad()
            y_score_gpu = model(*data_gpu)
            loss = loss_func(y_score_gpu, y_true_gpu)
            loss.backward()
            optimizer.step()
            
            # store the raw data.
            y_score = y_score_gpu.squeeze(-1).detach().cpu()
            all_y_true += list(y_true.numpy())
            all_y_score += list(y_score.numpy())
            all_loss.append(float(loss.cpu()))

            # update description of progress bar.
            loss_avg = (loss_avg * i + loss) / (i+1)
            if scheduler is not None:
                progress_bar.set_description(f"loss: {loss}, loss_avg: {loss_avg}, lr:{scheduler.get_last_lr()}")
            else:
                progress_bar.set_description(f"loss: {loss}, loss_avg: {loss_avg}")
            
        return all_y_true, all_y_score, all_loss


    @ torch.no_grad()
    def __validate(self, model: torch.nn.Module, dataloader: DataLoader):
        model.eval()

        all_y_true, all_y_score = [], []
        progress_bar = tqdm(dataloader)
        for i, data in enumerate(progress_bar):
            y_true = data[-1]
            data_gpu = map(lambda x : x.to(self.device), data[:-1])
            
            y_score_gpu = model(*data_gpu)
            
            y_score = y_score_gpu.squeeze(-1).detach().cpu()
            all_y_true += list(y_true.numpy())
            all_y_score += list(y_score.numpy())
        
        return all_y_true, all_y_score