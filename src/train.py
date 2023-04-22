from transppi.ppi_transformer import PPITransformer
from utils import check_data_integrity
from utils.load_data import get_ppi_dataset

import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import numpy as np

from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

import h5py

import argparse
import sys
import os
from datetime import datetime
import random


class TaskDataset(Dataset):
    def __init__(self, ppi_dataset, coord_dataset, prottrans_dataset):
        super(TaskDataset).__init__()
        self.ppi_dataset = ppi_dataset
        self.coord_dataset = coord_dataset
        self.prottrans_dataset = prottrans_dataset
 
    def __len__(self):
        return len(self.ppi_dataset)
    
    def __getitem__(self, key):
        id1, id2, label = self.ppi_dataset[key]
        ids = [id1, id2]
        protein_coord = [torch.tensor(np.array(self.coord_dataset[id])) for id in ids]
        protein_prottrans = [torch.tensor(np.array(self.prottrans_dataset[id])) for id in ids]
        # [protein, vertex, dim_coord], [protein, vertex, dim_feat], 1
        return ids, protein_coord, protein_prottrans, label
    
    def collate_fn(self, batch):
        ids, coord_batch_protein, prottrans_batch_protein, label_batch = map(list, zip(*batch))

        def _pad(data:torch.Tensor, target_dim:int):
            '''
            [dim,...] -> [target_dim, ...]
            '''
            shape = [target_dim] + list(data.shape)[1:]
            padded_data = torch.zeros(shape)
            padded_data[:data.shape[0]] = data[:data.shape[0]]
            return padded_data
        

        def _pad_batch_protein(data:list[list[torch.Tensor]], target_dim:int):
            '''
            [batch,protein,dim] -> tensor[batch,protein,target_dim]
            '''
            batch_size = len(data)
            protein_size = len(data[0])
            padded_data = torch.stack([
                torch.stack([
                    _pad(data[batch][protein], target_dim)
                    for protein
                    in range(protein_size)
                ])
                for batch
                in range(batch_size)
            ])
            return padded_data

        # [batch, protein]
        length_batch_protein = torch.tensor([[len(coord_protein[0]), len(coord_protein[1])] for coord_protein in coord_batch_protein])
        # length_batch_protein_2 = torch.tensor([[len(node_feat_protein[0]), len(node_feat_protein[1])] for node_feat_protein in node_feat_batch_protein])
        # assert length_batch_protein.equal(length_batch_protein_2), f"length_not_equal, {ids} {length_batch_protein}, {length_batch_protein_2}"

        maxlen = max(1500, int(torch.max(length_batch_protein))) # note: we cut off 1500 length to avoid Out-Of-Memory
        padded_coord_batch_protein = _pad_batch_protein(coord_batch_protein, maxlen)
        padded_prottrans_batch_protein = _pad_batch_protein(prottrans_batch_protein, maxlen)
    
        return padded_coord_batch_protein, padded_prottrans_batch_protein, length_batch_protein.unsqueeze(-1), torch.tensor(label_batch, dtype=torch.float32)



class Logger:
    def __init__(self, out_dir, fold, prefix):
        self.out_dir = os.path.join(args.out_dir, f'{fold}')
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
        # sns.histplot(y_score_neg, color='red', kde=True, ax=ax)
        # sns.histplot(y_score_pos, color='green', kde=True, ax=ax)
        sns.kdeplot(y_score_neg, color='red', ax=ax)
        sns.kdeplot(y_score_pos, color='green', ax=ax)
        ax.set_ylabel('score')
        ax.set_xlabel('batch')

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
        self._draw_distribution(y_true, y_score, ax=ax)
        fig.savefig(os.path.join(self.out_dir, prefix_epoch + 'kde.png'), dpi=200)
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

    

class Trainer:
    def __init__(self, args):
        seed = random.randint(0, 10000)
        # comment this line out to use the seed specified in command line.
        args.random_state = seed
        self.seed_everything(args.random_state)
        self.args = args
        self.out_dir = args.out_dir

        # prepare the device to run on.
        assert torch.cuda.is_available(), "Cuda is not available."
        self.device = torch.device(f'cuda:{args.gpu}')

        # prepare the output dir.
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, 'args.txt'), 'w') as file:
            file.write(str(args))

        # prepare the datasets.
        self.ppi_dataset = get_ppi_dataset(args.ppi_dir)
        self.coord_dataset = h5py.File(args.coord, 'r')
        self.prottrans_dataset = h5py.File(args.prottrans, 'r')

    def seed_everything(self, seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        return

    def main(self):
        self.train_kfold(args)


    def train_kfold(self, args):
        random.shuffle(self.ppi_dataset)
        self.ppi_dataset_y = [label for _, _, label in self.ppi_dataset]
        skf = StratifiedKFold(n_splits=10, shuffle=False)
        for fold, (train_index, test_index) in enumerate(skf.split(np.zeros(len(self.ppi_dataset_y)), self.ppi_dataset_y)):
            train_ppi_dataset = [self.ppi_dataset[index] for index in train_index]
            test_ppi_dataset = [self.ppi_dataset[index] for index in test_index]
            
            train_task_dataset = TaskDataset(train_ppi_dataset, self.coord_dataset, self.prottrans_dataset)
            train_dataloader = DataLoader(train_task_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=train_task_dataset.collate_fn)

            validate_task_dataset = TaskDataset(test_ppi_dataset, self.coord_dataset, self.prottrans_dataset) 
            validate_dataloader = DataLoader(validate_task_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=train_task_dataset.collate_fn)

            model = PPITransformer(args.dim_edge_feat, args.dim_vertex_feat, args.dim_hidden).to(self.device)
            
            # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
            train_logger = Logger(args.out_dir, fold=fold, prefix='train_')
            validate_logger = Logger(args.out_dir, fold=fold, prefix='validate_')
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10, T_mult=2)

            for epoch in range(args.epochs):
                self.train_and_log(model, train_dataloader, optimizer, scheduler, train_logger, epoch)
                self.validate_and_log(model, validate_dataloader, validate_logger, epoch)
                scheduler.step()

            torch.save(model, os.path.join(args.out_dir, f'{fold}_model.pth'))

    def train_and_log(self, model, dataloader, optimizer, scheduler, logger:Logger, epoch):
        y_true, y_score, loss = self.train(model, dataloader, optimizer, scheduler)
        logger.step_metrics(epoch, y_true, y_score)
        logger.step_loss(epoch, loss)

    def validate_and_log(self, model, dataloader, logger, epoch):
        y_true, y_score = self.validate(model, dataloader)
        logger.step_metrics(epoch, y_true, y_score)
        auroc = metrics.roc_auc_score(y_true, y_score)
        return auroc

    def train(self, model: torch.nn.Module, dataloader: DataLoader, optimizer:torch.optim.Optimizer, scheduler:torch.optim.lr_scheduler.LRScheduler):
        model.train()
        loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0], dtype=torch.float32)).to(self.device)

        progress_bar = tqdm(dataloader)
        
        all_y_true, all_y_score, all_loss = [], [], []
        loss_avg = 0
        for i, (vertex_coord, vertex_feat, protein_length, y_true) in enumerate(progress_bar):
            vertex_coord_gpu = vertex_coord.to(self.device)
            vertex_feat_gpu = vertex_feat.to(self.device)
            protein_length_gpu = protein_length.to(self.device)
            y_true_gpu = y_true.unsqueeze(-1).to(self.device)

            optimizer.zero_grad()
            y_score_gpu = model(vertex_coord_gpu, vertex_feat_gpu, protein_length_gpu)
            loss = loss_func(y_score_gpu, y_true_gpu)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # store the raw data.
            y_score = y_score_gpu.squeeze(-1).detach().cpu()
            all_y_true += list(y_true.numpy())
            all_y_score += list(y_score.numpy())
            all_loss.append(float(loss.cpu()))

            # update description of progress bar.
            loss_avg = (loss_avg * i + loss) / (i+1)
            progress_bar.set_description(f"loss: {loss}, loss_avg: {loss_avg}, lr:{scheduler.get_last_lr()}")
            
        return all_y_true, all_y_score, all_loss


    @ torch.no_grad()
    def validate(self, model: torch.nn.Module, dataloader: DataLoader):
        model.eval()

        all_y_true, all_y_score = [], []
        progress_bar = tqdm(dataloader)
        for i, (vertex_coord, vertex_feat, protein_length, y_true) in enumerate(progress_bar):
            vertex_coord_gpu = vertex_coord.to(self.device)
            vertex_feat_gpu = vertex_feat.to(self.device)
            protein_length_gpu = protein_length.to(self.device)
            
            y_score_gpu = model(vertex_coord_gpu, vertex_feat_gpu, protein_length_gpu)
            
            y_score = y_score_gpu.squeeze(-1).detach().cpu()
            all_y_true += list(y_true.numpy())
            all_y_score += list(y_score.numpy())
        
        return all_y_true, all_y_score



def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dim_vertex_feat', type=int, default=1024)
    parser.add_argument('--dim_edge_feat', type=int, default=64)
    parser.add_argument('--dim_hidden', type=int, default=128)

    parser.add_argument('--ppi_dir', default='data/ppi/Profppikernel')
    parser.add_argument('--coord', default='data/coord.hdf5')
    parser.add_argument('--prottrans', default='data/prottrans.hdf5')
    
    parser.add_argument('--random_state', default=2023)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-4)

    parser.add_argument('--out_dir', default=os.path.join('out', 'train', datetime.now().strftime("%y-%m-%d-%H-%M") ))

    parser.add_argument('--gpu', type=int, default=0)

    return parser  

if __name__ == '__main__':
    parser = create_arg_parser()
    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.main()
