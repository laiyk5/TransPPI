from transppi.ppi_transformer import PPITransformer
from utils.load_data import get_ppi_dataset

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

import h5py

import argparse
import os
from datetime import datetime
from utils.train import Trainer, Factory, seed_everything


class TaskDataset(Dataset):
    def __init__(self, ppi_dataset, coord_dataset, prottrans_dataset):
        super().__init__()
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
        

        def _pad_batch_protein(data, target_dim:int):
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

class TransPPIFactory(Factory):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.coord_dataset = h5py.File(args.coord)
        self.prottrans_dataset = h5py.File(args.prottrans)

    def new_model_scheduler_optimizer(self, device):
        model = PPITransformer(self.args.dim_edge_feat, self.args.dim_vertex_feat, self.args.dim_hidden).to(device)
        # optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        # scheduler = None
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10, T_mult=2)
        return model, optimizer, scheduler
    
    def new_dataloader(self, ppi_dataset):
        task_dataset = TaskDataset(ppi_dataset, self.coord_dataset, self.prottrans_dataset)
        dataloader = DataLoader(task_dataset, shuffle=False, batch_size=self.args.batch_size, collate_fn=task_dataset.collate_fn)
        return dataloader
    
    def new_loss_func(self):
        return torch.nn.BCEWithLogitsLoss()


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dim_vertex_feat', type=int, default=1024)
    parser.add_argument('--dim_edge_feat', type=int, default=64)
    parser.add_argument('--dim_hidden', type=int, default=64)

    parser.add_argument('--ppi_dir', default='data/ppi/Pans')
    parser.add_argument('--coord', default='data/coord.hdf5')
    parser.add_argument('--prottrans', default='dgp/out/prottrans_normed.hdf5')
    
    parser.add_argument('--random_state', default=2023)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-4)

    parser.add_argument('--out_dir', default=os.path.join('out', 'train_transppi', datetime.now().strftime("%y-%m-%d-%H-%M") ))

    parser.add_argument('--gpu', type=int, default=0)

    return parser

def main(args):
    seed_everything(args.random_state)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'args.txt'), 'wt') as outfile:
        outfile.write('TransPPI\n')
        outfile.write(str(args))

    ppi_dataset = get_ppi_dataset(args.ppi_dir)
    factory = TransPPIFactory(args)

    assert torch.cuda.is_available()
    device = torch.device(f'cuda:{args.gpu}')
    trainer = Trainer(args.out_dir, args.epochs, 10, factory, device)
    
    trainer.train_kfold(ppi_dataset)


if __name__ == '__main__':
    parser = create_arg_parser()
    args = parser.parse_args()
    main(args)
