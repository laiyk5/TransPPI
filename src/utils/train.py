
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import numpy as np

from tqdm import tqdm

import os
import random

from .logger import Logger

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
        for fold, (train_index, validation_index) in enumerate(skf.split(np.zeros(len(self.ppi_dataset_y)), self.ppi_dataset_y)):
            def up_sample_pos(idx):
                pos_idx = [i for i in idx if dataset[i][2] == 1]
                neg_idx = [i for i in idx if dataset[i][2] == 0]
                len_pos = len(pos_idx)
                len_neg = len(neg_idx)
                pos_sampled_idx = random.choices(pos_idx, k=len_neg - len_pos)
                sampled_idx = list(idx) + pos_sampled_idx
                return sampled_idx
            
            train_index = up_sample_pos(train_index)
            # train_index = random.sample(train_index, 50)
            # validation_index = random.sample(list(validation_index), 300)

            random.shuffle(train_index)
            random.shuffle(validation_index)
            train_ppi_dataset = [dataset[index] for index in train_index]
            validate_ppi_dataset = [dataset[index] for index in validation_index]
            
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
            # if scheduler is not None:
            #     scheduler.step()
            model_path = os.path.join(self.out_dir, 'model')
            os.makedirs(model_path, exist_ok=True)
            torch.save(model, os.path.join(model_path, f'{fold}_{epoch}_model.pth'))


    def __train_and_log(self, model, dataloader, optimizer, scheduler:torch.optim.lr_scheduler.LRScheduler, logger:Logger, epoch):
        model.train()
        loss_func = self.factory.new_loss_func().to(self.device)

        progress_bar = tqdm(dataloader)
        
        all_y_true, all_y_score, all_loss, all_lr = [], [], [], []
        loss_avg = 0
        
        batch_size = len(progress_bar)
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
            if scheduler is not None:
                scheduler.step(epoch * batch_size + i)
                all_lr += scheduler.get_last_lr()
                # scheduler.step(epoch + i/batch_size)

            # update description of progress bar.
            loss_avg = (loss_avg * i + loss) / (i+1)
            progress_bar.set_description(f"loss: {loss}, loss_avg: {loss_avg}")

            if i % 10 == 9:
                logger.append_loss(all_loss)
                all_loss.clear()
                if scheduler is not None:
                    logger.append_lr(all_lr)
                    all_lr.clear()

        logger.step_metrics(epoch, all_y_true, all_y_score)
        logger.append_loss(all_loss)
        logger.append_lr(all_lr)

    @torch.no_grad()
    def __validate_and_log(self, model, dataloader, logger, epoch):
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
        logger.step_metrics(epoch, all_y_true, all_y_score)
        auroc = metrics.roc_auc_score(all_y_true, all_y_score)
        return auroc