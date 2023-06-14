'''
A testing script that test a model on a given testing set and given features.
The model should be loaded from a checked point.
This script save the result of the prediction and the ground true value.
Save the model and predictions of the training model for each epoch.
'''

from nn import PPITransformer

import torch
from torch.utils.data import DataLoader

import argparse
import os
import sys
import random

from data import TaskDataset, collate_fn

import json
import h5py

def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, required=True)

    # testing data sets
    parser.add_argument('--ppi', type=str, default='../../preprocessing/out/ppi/Pans/test_0.json')
    parser.add_argument('--coord', type=str, default='../../preprocessing/out/coord.hdf5')
    parser.add_argument('--prottrans', type=str, default='../../preprocessing/out/prottrans_normed.hdf5')
    
    # testing settings
    parser.add_argument('--random_state', type=int, default=2023)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)

    # output
    parser.add_argument('--output', '-o', type=str, help='The output file.')

    return parser


parser = create_arg_parser()
args = parser.parse_args()

if not torch.cuda.is_available():
    print("CUDA is not available.",file=sys.stderr)
    exit(-1)
device = torch.device(f'cuda:{args.gpu}')

model_config = {}
with open('model_config.json', 'rt') as file:
    model_config = json.load(file)

model = PPITransformer(dim_edge_feat=model_config["dim_edge_feat"], dim_vertex_feat=model_config["dim_vertex_feat"], dim_hidden=model_config["dim_hidden"]).to(device)

checkpoint = torch.load(args.checkpoint)
model.load_state_dict(checkpoint['model_state_dict'])
last_epoch = checkpoint['epoch']

ppi_file = open(args.ppi, 'rt')
ppi_dataset = json.load(ppi_file)
ppi_file.close()

def up_sample_pos(dataset):
    pos = [data for data in dataset if data[-1] == 1]
    neg = [data for data in dataset if data[-1] == 0]
    len_pos = len(pos)
    len_neg = len(neg)
    pos_sampled = random.choices(pos, k=len_neg - len_pos)
    pos = pos + pos_sampled
    return pos + neg

# ppi_dataset = up_sample_pos(ppi_dataset)
#random.shuffle(ppi_dataset)
#ppi_dataset = ppi_dataset[:10000]

coord_dataset = h5py.File(args.coord)
prottrans_dataset = h5py.File(args.prottrans)

training_task_dataset = TaskDataset(ppi_dataset, coord_dataset, prottrans_dataset)
dataloader = DataLoader(training_task_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=collate_fn)

loss_func = torch.nn.BCEWithLogitsLoss().to(device)
model.eval()

from tqdm import tqdm

y_pred, y_true = [], []
# iterate the training set
for data_row in tqdm(dataloader):
    coord, prottrans, len, label = [data.to(device) for data in data_row ]
    logits = model(coord, prottrans, len).squeeze()
    y_true += label.tolist()
    y_pred += torch.sigmoid(logits).tolist()

os.makedirs(os.path.dirname(args.output), exist_ok=True)
outfile = open(args.output, 'wt')
for y_pred, y_true in zip(y_pred, y_true):
    outfile.write(str(y_pred) + ' ' + str(y_true) + ' ' + '\n')
outfile.close()
