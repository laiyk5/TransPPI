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


def log_metrics(y_true, y_score, out_dir, out_prefix=''):
    log_file = open(os.path.join(out_dir, out_prefix+"metrics.log"), 'wt')

    fpr, tpr, thredsholds = metrics.roc_curve(y_true, y_score)
    roc_data = {"TPR": tpr, "FPR": fpr}
    fig, ax = plt.subplots(figsize=(5,5))
    sns.lineplot(data=roc_data, x="FPR", y="TPR", ax=ax)
    plt.savefig(os.path.join(out_dir, out_prefix + "roc.png"), dpi=150)

    precisions, recalls, thredsholds = metrics.precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5,5))
    prc_data = {"Precision": precisions, "Recall": recalls}
    sns.lineplot(data=prc_data, x="Recall", y="Precision", ax=ax)
    plt.savefig(os.path.join(out_dir, out_prefix + "prc.png"), dpi=150)

    auroc = metrics.auc(fpr, tpr)
    auprc = metrics.auc(recalls, precisions)
    log_file.write(f"auroc={auroc}, auprc={auprc}\n")
    
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-20)
    idx = np.argmax(f1_scores)
    precision, recall, f1_scores = precisions[idx], recalls[idx], f1_scores[idx]
    log_file.write(f"precision={precision}, recall={recall}, f1_scores={f1_scores}\n")

    log_file.close()


def train(model: torch.nn.Module, dataloader: DataLoader, optimizer:torch.optim.Optimizer, device):
    model.train()
    loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10])).to(device)

    progress_bar = tqdm(dataloader)
    
    all_y_true, all_y_score = [], []
    loss_avg = 0
    for i, (vertex_coord, vertex_feat, protein_length, y_true) in enumerate(progress_bar):
        #print(vertex_coord.shape, vertex_feat.shape, protein_length.shape, y_true.shape)
        #sys.exit()
        vertex_coord_gpu = vertex_coord.to(device)
        vertex_feat_gpu = vertex_feat.to(device)
        protein_length_gpu = protein_length.to(device)
        y_true_gpu = y_true.unsqueeze(-1).to(device)
        
        y_score_gpu = model(vertex_coord_gpu, vertex_feat_gpu, protein_length_gpu)

        loss = loss_func(y_score_gpu, y_true_gpu)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        y_score = y_score_gpu.squeeze(-1).detach().cpu()
        all_y_true += list(y_true.numpy())
        all_y_score += list(y_score.numpy())

        loss_avg = (loss_avg * i + loss) / (i+1)
        progress_bar.set_description(f"loss: {loss}, loss_avg: {loss_avg}")

    return all_y_true, all_y_score
        

def validate(model: torch.nn.Module, dataloader: DataLoader, device):
    model.train()

    all_y_true, all_y_score = [], []
    progress_bar = tqdm(dataloader)
    for i, (vertex_coord, vertex_feat, protein_length, y_true) in enumerate(progress_bar):
        vertex_coord_gpu = vertex_coord.to(device)
        vertex_feat_gpu = vertex_feat.to(device)
        protein_length_gpu = protein_length.to(device)
        
        y_score_gpu = model(vertex_coord_gpu, vertex_feat_gpu, protein_length_gpu)
        
        y_score = y_score_gpu.detach().cpu()
        all_y_true += list(y_true.numpy())
        all_y_score += list(y_score.numpy())
    
    return all_y_true, all_y_score



def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'args.txt'), 'w') as file:
        file.write(str(args))

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # load dataset
    ppi_dataset = get_ppi_dataset(args.ppi_dir)
    coord_dataset = h5py.File(args.coord, 'r')
    prottrans_dataset = h5py.File(args.prottrans, 'r')
    check_data_integrity(ppi_dataset, coord_dataset, prottrans_dataset)
    
    ppi_dataset_y = [label for _, _, label in ppi_dataset]
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.random_state)
    #skf = StratifiedKFold(n_splits=5)
    for fold, (train_index, test_index) in enumerate(skf.split(np.zeros(len(ppi_dataset_y)), ppi_dataset_y)):
        print(f"[FOLD{fold}]")

        res_file = h5py.File(os.path.join(args.out_dir, 'res.hdf5'), 'w')
        
        # Build dataloaders.
        train_ppi_dataset = [ppi_dataset[index] for index in train_index]
        test_ppi_dataset = [ppi_dataset[index] for index in test_index]
        train_task_dataset = TaskDataset(train_ppi_dataset, coord_dataset, prottrans_dataset)
        test_task_dataset = TaskDataset(test_ppi_dataset, coord_dataset, prottrans_dataset) 
        train_dataloader = DataLoader(train_task_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=train_task_dataset.collate_fn)
        test_dataloader = DataLoader(test_task_dataset, shuffle=True, batch_size=1, collate_fn=train_task_dataset.collate_fn)
        
        model = PPITransformer(args.dim_edge_feat, args.dim_vertex_feat, args.dim_hidden, device=device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=3, verbose=True)
        

        for epoch in range(args.epochs):
            print(f"[FOLD{fold}][EPOCH{epoch}]")
            
            y_true, y_score = train(model, train_dataloader,
                  optimizer=optimizer,
                  device=device)
            res_file.create_dataset(f"{fold}_{epoch}_train_score", data=y_score)
            res_file.create_dataset(f"{fold}_{epoch}_train_true", data=y_true)
            log_metrics(y_true, y_score, out_dir=args.out_dir, out_prefix=f'{fold}_{epoch}_train_')

            y_true, y_score = validate(model, test_dataloader, device)
            res_file.create_dataset(f"{fold}_{epoch}_test_score", data=y_score)
            res_file.create_dataset(f"{fold}_{epoch}_test_true", data=y_true)
            log_metrics(y_true, y_score, out_dir=args.out_dir, out_prefix=f'{fold}_{epoch}_test_')

            auroc = metrics.roc_auc_score(y_true, y_score)
            scheduler.step(auroc)

        torch.save(model, os.path.join(args.out_dir, f'{fold}_model.pth'))


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dim_vertex_feat', default=1024)
    parser.add_argument('--dim_edge_feat', default=64)
    parser.add_argument('--dim_hidden', default=512)

    parser.add_argument('--ppi_dir', default='data/ppi/Profppikernel')
    parser.add_argument('--coord', default='data/coord.hdf5')
    parser.add_argument('--prottrans', default='data/normalized_prottrans-1500.hdf5')
    
    parser.add_argument('--random_state', default=2023)
    parser.add_argument('--batch_size', default=2)
    parser.add_argument('--epochs', default=50)
    parser.add_argument('--lr', default=5e-4)

    parser.add_argument('--out_dir', default=os.path.join('out', 'train', datetime.now().strftime("%y-%m-%d-%H-%M") ))

    parser.add_argument('--gpu', default=0)

    return parser  

if __name__ == '__main__':
    parser = create_arg_parser()
    args = parser.parse_args()
    main(args)
