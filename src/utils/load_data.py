import re
import os
from collections import UserDict
import h5py
import torch
import numpy as np

# low-level file reading methods.

def read_ppi_file(file, label):
    entries = []
    pattern = re.compile(r'(\w+)\W+(\w+)')
    for line in file:
        m = pattern.search(line)
        if m is not None:
            entries.append((m[1], m[2], label))
    return entries


def read_fasta_file(file):
    seq_dataset = dict()
    sequence_id = ''
    sequence = ''
    for line in file:
        if line.startswith(">"):
            m = re.search(r'>(?P<id>[A-Z0-9]+)', line)
            if m:
                sequence_id = m['id']
                seq_dataset[sequence_id] = ''
        else:
            sequence = re.sub(r"\s+", "", line)
            seq_dataset[sequence_id] += sequence
    return seq_dataset


def read_coordinates_from_pdb_file(file):
    pattern = re.compile(r'''^(ATOM)\s+
                             ([1-9][0-9]*)\s+   # atom pos
                             (?P<atom_name>\w+)\s+           # atom name
                             (\w+)\s+           # residue name
                             (\w+)\s*           # \s* for data like A1117 in Q5T6F2
                             (?P<residue_pos>[1-9][0-9]*)\s+   # residue pos
                             (?P<x>-?[0-9]+\.[0-9]+)\s*  # residue coord x
                             (?P<y>-?[0-9]+\.[0-9]+)\s*  # residue coord y
                             (?P<z>-?[0-9]+\.[0-9]+)\s+  # residue coord z
                             ([0-9]+\.[0-9]+)\s+
                             ([0-9]+\.[0-9]+)\s+
                             (\w+)\s*$''', re.X)
    coords = []
    for line in file:
        match = pattern.search(line)
        if not match:
            continue
        if len(coords) == 1500:
            break
        if match['atom_name'] == 'CA':
            coords.append([float(match['x']), float(match['y']), float(match['z'])])
    return coords

# id

def ppi_dataset_to_id_dataset(ppi_dataset):
    id_dataset = set()
    for item in ppi_dataset:
        id_dataset.add(item[0])
        id_dataset.add(item[1])
    return id_dataset


# High level abstractions

def get_ppi_dataset(ppi_dir):
    ppi_dataset = []
    with open(os.path.join(ppi_dir, "pos.txt")) as file:
        ppi_dataset += read_ppi_file(file, label=1)
    with open(os.path.join(ppi_dir, "neg.txt")) as file:
        ppi_dataset += read_ppi_file(file, label=0)
    return ppi_dataset
    

def get_seq_dataset(seq_file_path):
    file = open(seq_file_path)
    seq_dataset = read_fasta_file(file)
    file.close()
    return seq_dataset


class CoordDataset:
    '''
    We don't load the whole dataset into the memory
    because It's both time and memory consuming.
    '''
    def __init__(self, pdb_dir):
        self.pdb_dir = pdb_dir
        
    def get(self, id):
        path = os.path.join(self.pdb_dir, f'{id}.pdb')
        with open(path) as file:
            coords = read_coordinates_from_pdb_file(file)
        return torch.tensor(coords)
     
    def keys(self):
        filenames = os.listdir(self.pdb_dir)
        id_pattern = re.compile(r'^(?P<id>[A-Z0-9]+)\.pdb')
        id_dataset = set()
        for filename in filenames:
            m = id_pattern.search(filename)
            if m is not None:
                id_dataset.add(m['id'])
        return id_dataset


class ProtTransDataset:
    def __init__(self, prottrans_path: str):
        self.f = h5py.File(prottrans_path, 'r')
    
    def get(self, key):
        return torch.tensor(np.array(self.f[key]))

    def keys(self):
        return set(self.f.keys())
