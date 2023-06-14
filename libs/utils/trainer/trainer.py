
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import numpy as np

from tqdm import tqdm

import os
import sys
import random

from .logger import Logger

class Factory:
    def new_model_scheduler_optimizer(self, device):
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
        self.ppi_dataset_y = [label for _, _, label in dataset]
        skf = StratifiedKFold(n_splits=10, shuffle=True)
        for fold, (train_index, validation_index) in enumerate(skf.split(np.zeros(len(self.ppi_dataset_y)), self.ppi_dataset_y)):
            def up_sample_pos(idx):
                pos_idx = [i for i in idx if dataset[i][-1] == 1]
                neg_idx = [i for i in idx if dataset[i][-1] == 0]
                len_pos = len(pos_idx)
                len_neg = len(neg_idx)
                pos_sampled_idx = random.choices(pos_idx, k=len_neg - len_pos)
                pos_idx = pos_idx + pos_sampled_idx
                assert len(pos_idx) == len(neg_idx), "len(pos_idx) != len(neg_idx) in train dataset."
                return pos_idx + neg_idx
            
            
            # train_index = up_sample_pos(train_index)
            # train_index = random.sample(train_index, 30000)
            # validation_index = random.sample(list(validation_index), 300)

            def down_sample_neg(idx):
                pos_idx = [i for i in idx if dataset[i][-1] == 1]
                neg_idx = [i for i in idx if dataset[i][-1] == 0]
                neg_idx = random.sample(neg_idx, k=len(pos_idx))
                assert len(neg_idx) == len(pos_idx), "len(pos_idx) != len(neg_idx) in train dataset."
                return pos_idx + neg_idx

            train_index = down_sample_neg(train_index)
            # validation_index = random.sample(list(validation_index), k=len(train_index))

            random.shuffle(train_index)
            random.shuffle(validation_index)
            train_ppi_dataset = [dataset[index] for index in train_index]
            validate_ppi_dataset = [dataset[index] for index in validation_index]
            
            train_dataloader = self.factory.new_dataloader(train_ppi_dataset)
            validate_dataloader = self.factory.new_dataloader(validate_ppi_dataset)

            self.__train_fold(fold, train_dataloader, validate_dataloader)


    def __train_fold(self, fold, train_dataloader, validate_dataloader):
        model, optimizer, scheduler = self.factory.new_model_scheduler_optimizer(self.device)
    
        train_logger = Logger(os.path.join(self.out_dir, str(fold), 'train'))
        validate_logger = Logger(os.path.join(self.out_dir, str(fold), 'validation'))

        for epoch in range(self.epochs):
            self.__train_and_log(model, train_dataloader, optimizer, scheduler, train_logger, epoch)
            if epoch % 10 == 0:
                self.__validate_and_log(model, validate_dataloader, validate_logger, epoch)
            if scheduler is not None:
                scheduler.step()
            # save model
            model_path = os.path.join(self.out_dir, 'model')
            os.makedirs(model_path, exist_ok=True)
            torch.save(model, os.path.join(model_path, f'{fold}_{epoch}_model.pth'))
        self.__validate_and_log(model, validate_dataloader, validate_logger, self.epochs)

    def __train_and_log(self, model, dataloader, optimizer, scheduler, logger:Logger, epoch):
        model.train()
        loss_func = self.factory.new_loss_func()

        progress_bar = tqdm(dataloader)
        
        all_y_true, all_y_score, all_loss, all_lr = [], [], [], []
        loss_avg = 0
        
        batch_size = len(progress_bar)
        for i, data in enumerate(progress_bar):
            torch.cuda.empty_cache()

            data_gpu = map(lambda x : x.to(self.device), data[:-1])
            y_true_cpu = data[-1]

            y_true_gpu = y_true_cpu.unsqueeze(-1).to(self.device)

            optimizer.zero_grad()
            y_score_gpu = model(*data_gpu)
            loss_cpu = loss_func(y_score_gpu, y_true_gpu)
            loss_cpu.backward()
            optimizer.step()

            # logging

            loss = loss_cpu.item()
            y_true = y_true_cpu.tolist()
            y_score = torch.tensor(y_score_gpu.squeeze(-1).tolist()).sigmoid().tolist()
            
            # store the raw data.
            all_y_true += y_true
            all_y_score += y_score
            all_loss.append(loss)

            # all_lr.append(optimizer.rate()) # only for noam_opt

            if scheduler is not None:
                # scheduler.step(epoch * batch_size + i)
                # scheduler.step(epoch + i/batch_size)
                all_lr += scheduler.get_last_lr()

            # update description of progress bar.
            loss_avg = (loss_avg * i + loss) / (i+1)
            progress_bar.set_description(f"loss: {loss}, loss_avg: {loss_avg}")

            if i % 50 == 49:
                logger.append_loss(all_loss)
                all_loss.clear()
                if scheduler is not None:
                    logger.append_lr(all_lr)
                    all_lr.clear()

        logger.step_metrics(epoch, all_y_true, all_y_score)
        logger.append_loss(all_loss)
        logger.append_loss_avg(loss_avg)
        logger.append_lr(all_lr)

    @torch.no_grad()
    def __validate_and_log(self, model, dataloader, logger:Logger, epoch):
        model.eval()


        loss_func = torch.nn.BCEWithLogitsLoss(reduction='none')

        all_y_true, all_y_score, all_loss = [], [], []
        loss_avg = 0
        progress_bar = tqdm(dataloader)
        for i, data in enumerate(progress_bar):
            data_gpu = map(lambda x : x.to(self.device), data[:-1])
            y_true_cpu = data[-1]
            y_true_gpu = y_true_cpu.unsqueeze(-1).to(self.device)

            y_score_gpu = model(*data_gpu)
            loss_cpu = loss_func(y_score_gpu, y_true_gpu)

            y_true = y_true_cpu.tolist()
            y_score = torch.tensor(y_score_gpu.squeeze(-1).tolist()).sigmoid().tolist()
            loss = loss_cpu.squeeze(-1).tolist()

            all_y_true += y_true
            all_y_score += y_score
            all_loss += loss
            loss_avg = (loss_avg * i + np.mean(loss)) / (i+1)
            if i % 100 == 99:
                logger.append_loss(all_loss)
                all_loss.clear()
        logger.step_metrics(epoch, all_y_true, all_y_score)
        logger.append_loss(all_loss)
        logger.append_loss_avg(loss_avg)
        auroc = metrics.roc_auc_score(all_y_true, all_y_score)
        return auroc