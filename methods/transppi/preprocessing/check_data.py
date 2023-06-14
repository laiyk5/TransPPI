import sys

from utils.load_data import *
from utils.logs import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ppi_dir', default='data/Profppikernel')
parser.add_argument('--seq_path', default='out/seq-1500.fasta')
parser.add_argument('--pdb_dir', default='out/pdb')
parser.add_argument('--prottrans_path', default='out/prottrans-1500.hdf5')
args = parser.parse_args()

def short_repr(string: str, target_length=10):
    if (len(string) <= target_length):
        return string
    return f'{string[:target_length]}...({len(string) - target_length} left)'

print(f"# Constructing PPI dataset from {args.ppi_dir}")
ppi_dataset = get_ppi_dataset(args.ppi_dir)
print(len(ppi_dataset))
print(ppi_dataset[:3])

# Get id dataset from ppi_dataset.
id_dataset = ppi_dataset_to_id_dataset(ppi_dataset)
print(len(id_dataset))
print(list(id_dataset)[:3])

print(f"## Constructing sequence data from {args.seq_path}...")
seq_path = args.seq_path
seq_dataset = get_seq_dataset(seq_path)
print(len(seq_dataset))
print([(key, short_repr(value)) for key, value in list(seq_dataset.items())[:3]])

print(f"## Constructing coordinate dataset from {args.pdb_dir}...")
pdb_dir = args.pdb_dir
coord_dataset = CoordDataset(pdb_dir)
print(len(coord_dataset.keys()))
print([(key, short_repr(str(coord_dataset.get(key)), 20)) for key in list(id_dataset)[:3]])

print(f'## Constructing porttrans dataset from {args.prottrans_path}')
prottrans_dataset = ProtTransDataset(args.prottrans_path)
print(len(prottrans_dataset.keys()))
print([ (key, prottrans_dataset.get(key)) for key in list(prottrans_dataset.keys())[:3] ])


print('# Checking Data Integrity...')

print('## Checking PDB Dataset Integrity...')
pdb_id_dataset = id_dataset.intersection(coord_dataset.keys())
assert id_dataset == pdb_id_dataset, f"Missing pdb data: {id_dataset.difference(pdb_id_dataset)}"
print("OK.")

print('## Checking Seq Dataset Integrity...')
seq_id_dataset = id_dataset.intersection(seq_dataset.keys())
assert id_dataset == seq_id_dataset, f"Missing seq data: {id_dataset.difference(seq_id_dataset)}"
print("OK.")

print('## Checking ProtTrans Dataset Integrity...')
prottrans_id_dataset = id_dataset.intersection(prottrans_dataset.keys())
assert id_dataset == prottrans_dataset, f"Missing seq data: {id_dataset.difference(prottrans_id_dataset)}"
print("OK.")