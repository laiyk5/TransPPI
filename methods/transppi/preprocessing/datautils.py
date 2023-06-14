#!/usr/bin/env python3

import argparse
from urllib import request
import multiprocessing as mp
import tqdm
import sys
import os
import torch

import concurrent.futures

# =================== main ppi ======================

def read_ppi(infile_path):
    infile = open(infile_path)
    for line in infile:
        yield line.strip().split()
    return

def main_ppi(args):
    ids = set()
    for infile_path in args.infile:
        for line in read_ppi(infile_path):
            ids.add(line[0])
            ids.add(line[1])

    for id in ids:
        print(id)

# ==================== main download =================

def fetch_pdb(id):
    with request.urlopen(f'https://alphafold.ebi.ac.uk/files/AF-{id}-F1-model_v4.pdb', timeout=20) as pdbfile:
        pdb_content = pdbfile.read()
        return pdb_content

def fetch_and_save_pdb(id, outdir):
    pdb_content = fetch_pdb(id)
    outfilepath = os.path.join(outdir, f'{id}.pdb')
    with open(outfilepath, 'wb') as outfile:
        outfile.write(pdb_content)

def fetch_fasta(id):
    url = f'https://rest.uniprot.org/uniprotkb/{id}.fasta'
    with request.urlopen(url, timeout=20) as f:
        res = str(f.read().decode('utf-8'))
        head, seq = res.split('\n', maxsplit=1)
        seq_singleline = ''.join([part.strip() for part in seq.split('\n')])
        return f'>{id}\n{seq_singleline}\n'

def fetch_and_save_fasta(id, outdir):
    content = fetch_fasta(id)

    outfilepath = os.path.join(outdir, f'{id}.fasta')
    with open(outfilepath, 'wt') as outfile:
        outfile.write(content)


def download_id(method, ids, outdir):
    id_exception_dir = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_id = {executor.submit(method, id, outdir):id for id in ids}
        progress_bar = tqdm.tqdm(concurrent.futures.as_completed(future_to_id), total=len(ids))
        miss_count = 0
        progress_bar.set_description(f'miss: {miss_count}')
        for future in progress_bar:
            id = future_to_id[future]
            try:
                future.result()
            except Exception as exc:
                id_exception_dir[id] = exc
                miss_count += 1
                progress_bar.set_description(f'miss: {miss_count}')

    return id_exception_dir


def main_download(args):
    ids = []
    with open(args.infile) as infile:
        for line in infile:
            id = line.strip()
            ids.append(id)
    
    os.makedirs(args.outdir, exist_ok=True)

    method = None

    if args.task == 'pdb':
        method = fetch_and_save_pdb
    if args.task == 'fasta':
        method = fetch_and_save_fasta
    
    print(f"Downloading {len(ids)} {args.task} data...", file=sys.stderr)

    id_exception_dir = download_id(method, ids, args.outdir)

    retry_cnt = 5
    while (len(id_exception_dir) != 0 and retry_cnt > 0):
        retry_cnt -= 1
        print(f"Retrying {len(id_exception_dir.keys())} {args.task} data...", file=sys.stderr)
        id_exception_dir = download_id(method, id_exception_dir.keys(), args.outdir)
    for errid, exception in id_exception_dir.items():
        print(f'{errid} {exception}')
        
    

# ================== main fasta ====================


def item_generator(filepath):
    '''
    A lazy parser of a fasta file
    return an item in the following format:
    header = str (a single line without '>' indicator)
    body = str (a single line)
    '''
    with open(filepath) as infile:
        header = None
        body = []
        for l in infile:
            if (l.startswith('>')):
                new_header = l
                if header is not None:    # has item
                    yield header, ''.join(body)
                header = new_header.strip()[1:]
                body = []
            else:
                if header is not None:      # has item
                    body.append(l.strip())
                else:
                    continue
        yield header, ''.join(body)
        return

def main_fasta(args):
    fasta_dict = dict()
    for header, seq in item_generator(args.infile):
        fasta_dict[header] = seq

    id_set = set()
    with open(args.idfile, 'rt') as idfile:
        for line in idfile:
            id_set.add(line.strip())
    
    with open(args.outfile, 'w') as outfile:
        for id in tqdm.tqdm(id_set):
            outfile.write(f'>{id}\n')
            outfile.write(f'{fasta_dict[id]}\n')


# ===================== main distribution =====================

def select(filepath, ids):
    '''
    select proteins in file `filepath`
    '''
    fastas = list()
    for header, body in item_generator(filepath):
        id = header
        if id in ids:
            fastas.append((header, body))
    return fastas

def main_distribution(args):
    import matplotlib.pyplot as plt
    import seaborn as sns
    ids = []
    with open(args.infile) as infile:
        lines = infile.readlines()
        for line in tqdm.tqdm(lines):
            id = line.split()[0]
            ids.append(id)

    lens = []
    for head, body in select(args.seqfile, ids):
        lens.append(len(body))

    print(len(lens))

    sns.set_theme()
    axis = sns.displot(lens)
    axis.set_ylabels(f'count')
    axis.set_xlabels(f'length max_length={max(lens)}')
    plt.show()

# ================== main_coord

import numpy as np
import pickle

def get_pdb_xyz(pdb_file):
    '''
        extract CA coordinates of every residues of a protein
    '''
    current_pos = -1000
    X = []
    current_aa = {} # 'N', 'CA', 'C', 'O'
    for line in pdb_file:
        if (line[0:4].strip() == "ATOM" and int(line[22:26].strip()) != current_pos) or line[0:4].strip() == "TER":
            if current_aa != {}:
                X.append(current_aa["CA"]) # X.append([current_aa["N"], current_aa["CA"], current_aa["C"], current_aa["O"]]) #暂时只考虑CA原子
                current_aa = {}
            if line[0:4].strip() != "TER":
                current_pos = int(line[22:26].strip())

        if line[0:4].strip() == "ATOM":
            atom = line[13:16].strip()
            if atom in ['N', 'CA', 'C', 'O']:
                xyz = np.array([line[30:38].strip(), line[38:46].strip(), line[46:54].strip()]).astype(np.float32)
                current_aa[atom] = xyz
    return np.array(X)

def main_coord(args):
    
    coords = dict()
    for filename in tqdm.tqdm(os.listdir(args.datadir)):
        if not filename.endswith(".pdb"):
            continue
        id = filename.split('.')[0]
        pdb_file = open(os.path.join(args.datadir, filename))
        coord = get_pdb_xyz(pdb_file)
        coords[id] = coord
    
    with open(args.outpath, 'wb') as file:
        pickle.dump(coords, file)


# ================= main feature

def main_protrans(args):
    feats = dict()
    for filename in tqdm.tqdm(os.listdir(args.datadir)):
        if not filename.endswith(".npy"):
            continue
        id = filename.split('.')[0]
        feat_file = open(os.path.join(args.datadir, filename), 'rb')
        feat = np.load(feat_file)
        feats[id] = feat
    
    with open(args.outpath, 'wb') as file:
        pickle.dump(feats, file)


parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers()

ppi_parser = subparsers.add_parser('ppi', help='read id from ppi files')
ppi_parser.add_argument('infile', nargs='+')
ppi_parser.set_defaults(func=main_ppi)

download_parser = subparsers.add_parser('download', help='download data')
download_parser.add_argument('--task', choices=['pdb', 'fasta'], required=True)
download_parser.add_argument('--infile', type=str, required=True)
download_parser.add_argument('--outdir', type=str, required=True)
download_parser.set_defaults(func=main_download)

fasta_parser = subparsers.add_parser('fasta', help='filter and uniq fasta file')
fasta_parser.add_argument('--infile', type=str, required=True)
fasta_parser.add_argument('--outfile', type=str, required=True)
fasta_parser.add_argument('--idfile', type=str, required=True)
fasta_parser.set_defaults(func=main_fasta)

distribution_parser = subparsers.add_parser('distribution', help='plot the length distribution.')
distribution_parser.add_argument('--infile', type=str, required=True)
distribution_parser.add_argument('--seqfile', type=str, required=True)
distribution_parser.set_defaults(func=main_distribution)

coord_parser = subparsers.add_parser('coord', help='Prepare pdb data.')
coord_parser.add_argument('--datadir', type=str, required=True)
coord_parser.add_argument('--outpath', type=str, required=True)
coord_parser.set_defaults(func=main_coord)

protrans_parser = subparsers.add_parser('protrans', help='Prepare protrans data')
protrans_parser.add_argument('--datadir', type=str, required=True)
protrans_parser.add_argument('--outpath', type=str, required=True)
protrans_parser.set_defaults(func=main_protrans)

if __name__ == '__main__':
    args = parser.parse_args()
    args.func(args)