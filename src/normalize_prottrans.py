import h5py
import numpy as np
from tqdm import tqdm

prottrans_file = h5py.File('out/new-prottrans-1500.hdf5')

max_feature, min_feature = [], []
for key, feature in tqdm(prottrans_file.items()):
    max_feature.append(np.max(feature))
    min_feature.append(np.min(feature))
max_feature = np.max(max_feature)
min_feature = np.min(min_feature)

print(max_feature)
print(min_feature)

normalized_prottrans_file = h5py.File('out/normalized_prottrans-1500.hdf5', 'w')

for key, feature in tqdm(prottrans_file.items()):
    normalized_feature = (feature - min_feature) / (max_feature - min_feature)
    normalized_prottrans_file.create_dataset(key, data=normalized_feature)
