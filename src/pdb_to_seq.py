import re
from utils.load_data import *
from tqdm import tqdm

def read_seq_from_pdb_file(file):    
    aa_codes = {
     'ALA':'A', 'CYS':'C', 'ASP':'D', 'GLU':'E',
     'PHE':'F', 'GLY':'G', 'HIS':'H', 'LYS':'K',
     'ILE':'I', 'LEU':'L', 'MET':'M', 'ASN':'N',
     'PRO':'P', 'GLN':'Q', 'ARG':'R', 'SER':'S',
     'THR':'T', 'VAL':'V', 'TYR':'Y', 'TRP':'W',
     'Sec':'U', 'Pyl':'O'}
    pattern = re.compile(r'''^(ATOM)\s+
                            ([1-9][0-9]*)\s+   # atom pos
                            (?P<atom_name>\w+)\s+           # atom name
                            (?P<aa_name>\w+)\s+             # aa name
                            (\w+)\s*           # \s* for data like A1117 in Q5T6F2
                            (?P<residue_pos>[1-9][0-9]*)\s+   # residue pos
                            (?P<x>-?[0-9]+\.[0-9]+)\s*  # residue coord x
                            (?P<y>-?[0-9]+\.[0-9]+)\s*  # residue coord y
                            (?P<z>-?[0-9]+\.[0-9]+)\s+  # residue coord z
                            ([0-9]+\.[0-9]+)\s+
                            ([0-9]+\.[0-9]+)\s+
                            (\w+)\s*$''', re.X)
    
    seq = ""
    for line in file:
        m = pattern.search(line)
        if m is None or m["atom_name"] != "CA":
            continue
        if len(seq) == 1500:
            break
        aa_name = m["aa_name"]
        aa_code = aa_codes[aa_name]
        seq += aa_code
    return seq


def seq_id_pdb_seq():
    fasta_file = open("out/seq-1500.fasta", "rt")
    seq_dict = read_fasta_file(fasta_file)
    ids = seq_dict.keys()
    new_seq_file = open("out/new-seq-1500.fasta", "wt")
    for id in tqdm(ids):
        pdb_file = open(f"out/pdb/{id}.pdb", "rt")
        seq = read_seq_from_pdb_file(pdb_file)
        new_seq_file.write(f'>{id}\n')
        new_seq_file.write(f'{seq}\n')


if __name__ == '__main__':
    seq_id_pdb_seq()