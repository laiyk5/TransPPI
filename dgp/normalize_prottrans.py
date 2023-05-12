import h5py
import numpy as np
from tqdm import tqdm

def normalize(args):
    prottrans_file = h5py.File(args.input)

    max_feature, min_feature = [], []
    for key, feature in tqdm(prottrans_file.items()):
        max_feature.append(np.max(feature, axis=0))
        min_feature.append(np.min(feature, axis=0))
    max_feature = np.max(max_feature, axis=0)
    min_feature = np.min(min_feature, axis=0)

    print(max_feature)
    print(min_feature)

    normalized_prottrans_file = h5py.File(args.output, 'w')

    for key, feature in tqdm(prottrans_file.items()):
        normalized_feature = (feature - min_feature) / (max_feature - min_feature)
        normalized_prottrans_file.create_dataset(key, data=normalized_feature)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    args = parser.parse_args()
    normalize(args)