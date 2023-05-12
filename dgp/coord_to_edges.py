import numpy as np
import h5py
from tqdm import tqdm
import torch
from scipy import sparse as sp

def coord_to_adj(coord):
    coord = torch.tensor(np.array(coord))
    coord = coord.unsqueeze(1) - coord.unsqueeze(0)
    distance = torch.sqrt(torch.sum(torch.pow(coord, 2), dim=-1))
    adj = (distance < 10)
    sp.coo_matrix(adj)
    return adj

def main():
    coords_h5py = h5py.File('data/coord.hdf5', 'r')
    out_h5py = h5py.File('data/adj.hdf5', 'w')
    for key in tqdm(coords_h5py.keys()):
        coord = coords_h5py[key]
        adj = coord_to_adj(coord)
        out_h5py.create_dataset(key, data=adj)

if __name__ == '__main__':
    main()