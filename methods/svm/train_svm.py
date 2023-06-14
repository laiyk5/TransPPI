from sklearn import svm
from sklearn.model_selection import StratifiedKFold

from utils.load_data import get_ppi_dataset

import numpy as np
from argparse import ArgumentParser
import h5py

from utils.logger import Logger

from tqdm import tqdm

from datetime import datetime

import os
import sys

import numpy as np

import random

def create_parser():
    parser = ArgumentParser()
    parser.add_argument('--ppi', default='data/ppi/Pans')
    parser.add_argument('--feat', default='dgp/out/prottrans_normed.hdf5')
    parser.add_argument('--out', default=os.path.join('out', 'train_svm', datetime.now().strftime("%y-%m-%d-%H-%M") ))
    return parser


def main(args):
    os.makedirs(args.out, exist_ok=True)
    ppi_dataset = get_ppi_dataset(args.ppi)
    ppi_dataset_label = np.array(list(map(lambda x : x[2], ppi_dataset)))

    skf = StratifiedKFold(10, shuffle=True, random_state=2023)
    
    feat_dataset = h5py.File(args.feat, 'r')
    for fold, (train_idx, validation_idx) in enumerate(skf.split(X=np.zeros(len(ppi_dataset_label)), y=ppi_dataset_label)):
        def item_to_feat(item):
            feat0 = np.sum(feat_dataset[item[0]], axis=0)
            feat1 = np.sum(feat_dataset[item[1]], axis=0)
            return feat0 + feat1
        
        def up_sample_pos(idx):
            pos_idx = [i for i in idx if ppi_dataset[i][2] == 1]
            neg_idx = [i for i in idx if ppi_dataset[i][2] == 0]
            len_pos = len(pos_idx)
            len_neg = len(neg_idx)
            pos_sampled_idx = random.choices(pos_idx, k=len_neg - len_pos)
            sampled_idx = list(idx) + pos_sampled_idx
            return sampled_idx

        def down_sample_neg(idx):
            pos_idx = [i for i in idx if ppi_dataset[i][2] == 1]
            neg_idx = [i for i in idx if ppi_dataset[i][2] == 0]
            neg_sampled_idx = random.sample(neg_idx, k=len(pos_idx))
            return pos_idx + neg_sampled_idx
    
        train_idx = up_sample_pos(train_idx)
        train_idx = random.sample(train_idx, 30000)

        train_X = np.array(list(map(item_to_feat, tqdm([ppi_dataset[i] for i in train_idx]))))
        train_y = np.array(list(map(lambda x : x[2], tqdm([ppi_dataset[i] for i in train_idx]))))

        sys.stdout.flush()

        clf = svm.SVC(verbose=True, probability=True)

        train_logger = Logger(args.out, fold, 'train_')
        start = datetime.now()
        clf.fit(train_X, train_y)
        end = datetime.now()
        print(end-start)
        prediction = clf.predict_proba(train_X)
        prediction = prediction[:, 1]
        print(prediction)
        train_logger.step_metrics(0, train_y, prediction)

        validation_logger = Logger(args.out, fold, 'validation_')
        validation_X = np.array(list(map(item_to_feat, tqdm([ppi_dataset[i] for i in validation_idx]))))
        validation_y = np.array(list(map(lambda x : x[2], tqdm([ppi_dataset[i] for i in validation_idx]))))
        prediction = clf.predict_proba(validation_X)
        prediction = prediction[:, 1]
        validation_logger.step_metrics(0, validation_y, prediction)
        
if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)