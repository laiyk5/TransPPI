from sklearn import svm
from sklearn.model_selection import StratifiedKFold

from utils.load_data import get_ppi_dataset

import numpy as np
from argparse import ArgumentParser
import h5py

from utils.train import Logger

from tqdm import tqdm

from datetime import datetime

import os
import sys

import numpy as np

import random

def create_parser():
    parser = ArgumentParser()
    parser.add_argument('--ppi', default='data/ppi/Pans')
    parser.add_argument('--feat', default='data/prottrans.hdf5')
    parser.add_argument('--out', default=os.path.join('out', 'train_svm', datetime.now().strftime("%y-%m-%d-%H-%M") ))
    return parser


def main(args):
    os.makedirs(args.out, exist_ok=True)
    ppi_dataset = get_ppi_dataset(args.ppi)

    ppi_dataset_pos = [item for item in tqdm(ppi_dataset) if item[2] == 1]
    ppi_dataset += ppi_dataset_pos * 99
    random.shuffle(ppi_dataset)
    ppi_dataset = ppi_dataset[:5000]
    
    ppi_dataset_label = list(map(lambda x : x[2], ppi_dataset))

    skf = StratifiedKFold(10, shuffle=True, random_state=2023)
    
    feat_dataset = h5py.File(args.feat, 'r')
    for fold, (train_idx, validation_idx) in enumerate(skf.split(X=np.zeros(len(ppi_dataset_label)), y=ppi_dataset_label)):
        def item_to_feat(item):
            feat0 = np.sum(feat_dataset[item[0]], axis=0)
            feat1 = np.sum(feat_dataset[item[1]], axis=0)
            return feat0 + feat1
        train_X = np.array(list(map(item_to_feat, tqdm([ppi_dataset[i] for i in train_idx]))))
        train_y = np.array(list(map(lambda x : x[2], tqdm([ppi_dataset[i] for i in train_idx]))))

        sys.stdout.flush()

        clf = svm.SVC(verbose=True)

        train_logger = Logger(args.out, fold, 'train_')
        clf.fit(train_X, train_y)
        prediction = clf.predict(train_X)
        train_logger.step_metrics(0, train_y, prediction)

        validation_logger = Logger(args.out, fold, 'validation_')
        validation_X = np.array(list(map(item_to_feat, tqdm([ppi_dataset[i] for i in validation_idx]))))
        validation_y = np.array(list(map(lambda x : x[2], tqdm([ppi_dataset[i] for i in validation_idx]))))
        prediction = clf.predict(validation_X)
        validation_logger.step_metrics(0, validation_y, prediction)
        
if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)