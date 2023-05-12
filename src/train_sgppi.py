from transppi.ppi_transformer import PPITransformer
from utils import check_data_integrity
from utils.load_data import get_ppi_dataset

import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import numpy as np

import h5py

import argparse
import sys
import os
from datetime import datetime
import random
from utils.train import Trainer, Factory, seed_everything

import dgl.nn.pytorch as dglnn
import dgl
import torch.nn.functional as F
import torch.nn as nn
from scipy import sparse as sp

from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

class TaskDataset(Dataset):
    def __init__(self, ppi_dataset, coord_dataset, prottrans_dataset):
        super().__init__()
        self.ppi_dataset = ppi_dataset
        self.coord_dataset = coord_dataset
        self.prottrans_dataset = prottrans_dataset

    def __len__(self):
        return len(self.ppi_dataset)
    
    def __getitem__(self, key):
        def get_graph(coord):
            coord = torch.tensor(np.array(coord))
            coord = coord.unsqueeze(1) - coord.unsqueeze(0)
            distance = torch.sqrt(torch.sum(torch.pow(coord, 2), dim=-1))
            adj = (distance < 10)
            adj = sp.coo_matrix(adj)

            missing = adj.T > adj
            adj = adj + adj.T * missing - adj * missing
            g = dgl.from_scipy(adj)
            return g
        
        def get_item(id, coords, feats):
            g = get_graph(coords[id])
            feat = feats[id]
            g.ndata['fea'] = torch.tensor(np.array(feat), dtype=torch.float32)
            return g
        
        id1, id2, label = self.ppi_dataset[key]
        g1 = get_item(id1, self.coord_dataset, self.prottrans_dataset)
        g2 = get_item(id2, self.coord_dataset, self.prottrans_dataset)
        return g1, g2, label

    def collate_fn(self, batch):
        g1s, g2s, labels = map(list, zip(*batch))
        return dgl.batch(g1s), dgl.batch(g2s), torch.tensor(labels, dtype=torch.float32)


class MyGCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(MyGCN, self).__init__()
        self.out1 = dglnn.GraphConv(nfeat, nhid)
        self.out2 = dglnn.GraphConv(nhid, nhid)
        self.l1 = nn.Linear(nhid, 512)
        self.l2 = nn.Linear(512, 128)
        self.l3 = nn.Linear(128, 1)

    def forward(self, x1, x2):
        fea1 = x1.ndata['fea']
        fea2 = x2.ndata['fea']
        fea1 = F.relu(self.out1(x1, fea1))
        fea2 = F.relu(self.out1(x2, fea2))
        fea1 = F.relu(self.out2(x1, fea1))
        fea2 = F.relu(self.out2(x2, fea2))
        x1.ndata['fea'] = fea1
        x2.ndata['fea'] = fea2
        hg1 = dgl.mean_nodes(x1, 'fea')
        hg2 = dgl.mean_nodes(x2, 'fea')
        hg = torch.mul(hg1, hg2)
        l1 = self.l1(hg)
        l2 = self.l2(l1)
        l3 = self.l3(l2)
        return l3


class SGPPIFactory(Factory):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.coord_dataset = h5py.File(args.coord)
        self.prottrans_dataset = h5py.File(args.prottrans)

    def new_model_scheduler_optimizer(self):
        model = MyGCN(self.args.nfeat, self.args.nhid, self.args.dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        scheduler = None
        return model, optimizer, scheduler
    
    def new_dataloader(self, ppi_dataset):
        task_dataset = TaskDataset(ppi_dataset, self.coord_dataset, self.prottrans_dataset)
        dataloader = DataLoader(task_dataset, shuffle=False, batch_size=self.args.batch_size, collate_fn=task_dataset.collate_fn, num_workers=4)
        return dataloader
    
    def new_loss_func(self):
        return torch.nn.BCEWithLogitsLoss()


def create_arg_parser():
    parser = argparse.ArgumentParser(prog='train_sgppi', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--nfeat', type=int, default=1024, help='number of the dimentions of the features.')
    parser.add_argument('--nhid', type=int, default=512, help='number of the hidden dimension of the modoel.')
    parser.add_argument('--dropout', type=int, default=0.1, help='rate of dropout.')

    parser.add_argument('--ppi_dir', default='data/ppi/Pans', help='the path of the ppi directory.')
    parser.add_argument('--coord', default='data/coord.hdf5', help='the path of the coord.hdf5 file.')
    parser.add_argument('--prottrans', default='data/prottrans.hdf5', help='the path of the prottrans.hdf5 file')
    
    parser.add_argument('--random_state', default=2023, help='random state to regenerate the result.')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--epochs', type=int, default=50, help='epochs for each fold.')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate.')

    parser.add_argument('--out_dir', default=os.path.join('out', 'train_sgppi', datetime.now().strftime("%y-%m-%d-%H-%M") ), help='the directory to save outputs.')

    parser.add_argument('--gpu', type=int, default=0, help='the id of gpu to run.')

    return parser


def draw_adj(ppi_dataset, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    coord_dataset = h5py.File(args.coord)
    ids = []
    for ppi in ppi_dataset:
        ids.append(ppi[0])
        ids.append(ppi[1])

    for id in tqdm(ids, desc="draw adj"):
        coord = torch.tensor(np.array(coord_dataset[id]))
        coord = coord.unsqueeze(1) - coord.unsqueeze(0)
        distance = torch.sqrt(torch.sum(torch.pow(coord, 2), dim=-1))
        adj = (distance < 10)
        fig, ax = plt.subplots()
        sns.heatmap(data=adj, ax=ax)
        fig.savefig(os.path.join(out_dir, f'{id}.png'))
    

def main(args):
    seed_everything(args.random_state)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'args.txt'), 'wt') as outfile:
        outfile.write('SGPPI\n')
        outfile.write(str(args))

    ppi_dataset = get_ppi_dataset(args.ppi_dir)
    ppi_dataset_pos = [item for item in tqdm(ppi_dataset) if item[2] == 1]
    ppi_dataset += ppi_dataset_pos * 99
    random.shuffle(ppi_dataset)
    ppi_dataset = ppi_dataset[:5000]

    draw_adj(ppi_dataset, os.path.join(args.out_dir, 'adj'))

    # coord_dataset = h5py.File(args.coord)

    # ppi_dataset_new = []
    # for ppi in tqdm(ppi_dataset):
    #     id1, id2, label = ppi
    #     if np.array(coord_dataset[id1]).shape[0] > 500:
    #         continue
    #     if np.array(coord_dataset[id2]).shape[0] > 500:
    #         continue
    #     ppi_dataset_new.append((id1, id2, label))
    # ppi_dataset = ppi_dataset_new
        
    # idx_pos = []
    # idx_neg = []
    # for i in range(len(ppi_dataset)):
    #     if ppi_dataset[i][2] == 1:
    #         idx_pos.append(i)
    #     else:
    #         idx_neg.append(i)
    # ppi_dataset_new = []
    # idx = idx_pos[:10] + idx_neg[:1000]
    # for i in idx:
    #     ppi_dataset_new.append(ppi_dataset[i])
    # ppi_dataset = ppi_dataset_new

    factory = SGPPIFactory(args)

    assert torch.cuda.is_available()
    device = torch.device(f'cuda:{args.gpu}')
    trainer = Trainer(args.out_dir, args.epochs, 10, factory, device)
    
    trainer.train_kfold(ppi_dataset)


if __name__ == '__main__':
    parser = create_arg_parser()
    args = parser.parse_args()
    main(args)
