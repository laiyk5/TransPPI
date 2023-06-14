from sklearn.model_selection import StratifiedKFold
from utils.load_data import get_ppi_dataset
import numpy as np

import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
args = parser.parse_args()

ppi_dataset = get_ppi_dataset(args.input)
ppi_dataset_y = [label for _, _, label in ppi_dataset]

os.makedirs(args.output, exist_ok=True)

skf = StratifiedKFold(n_splits=10)
for fold, (train_index, test_index) in enumerate(skf.split(np.zeros(len(ppi_dataset_y)), ppi_dataset_y)):
    train_ppi_dataset = [ppi_dataset[index] for index in train_index]
    test_ppi_dataset = [ppi_dataset[index] for index in test_index]

    with open(os.path.join(args.output, f'train_{fold}.json'), 'wt') as f:
        f.write(json.dumps(train_ppi_dataset))
    
    with open(os.path.join(args.output, f'test_{fold}.json'), 'wt') as f:
        f.write(json.dumps(test_ppi_dataset))

