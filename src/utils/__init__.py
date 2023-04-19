from .load_data import *
from .logs import *

def check_data_integrity(ppi_dataset, coord_dataset, node_feat_dataset):
    id_dataset = ppi_dataset_to_id_dataset(ppi_dataset)
    
    # check coord_dataset
    pdb_id_dataset = id_dataset.intersection(coord_dataset.keys())
    assert id_dataset == pdb_id_dataset, f"Missing pdb data: {id_dataset.difference(pdb_id_dataset)}"

    # check node_feat_dataset
    prottrans_id_dataset = id_dataset.intersection(node_feat_dataset.keys())
    assert id_dataset == prottrans_id_dataset, f"Missing prottrans data: {id_dataset.difference(prottrans_id_dataset)}"