import torch
from torch import nn
from .edge_encoder import EdgeEncoder
from .graph_transformer_encoder import GraphTransformerEncoder
from .misc import gather_nodes

class ProteinEncoder(nn.Module):
    '''
    Encode proteins represented by (vertex_coord, vertex_feat, len)
    '''
    def __init__(self, dim_edge_feat, dim_vertex_feat, dim_hidden, device=torch.device('cpu')):
        super().__init__()
        self.edge_encoder = EdgeEncoder(dim_edge_feat, device=device)
        self.edge_linear = nn.Linear(dim_edge_feat, dim_hidden)
        self.vertex_linear = nn.Linear(dim_vertex_feat, dim_hidden)
        self.graph_transforemer_encoder = GraphTransformerEncoder(dim_hidden, device=device)
        self.device = device

    def forward(self, vertex_coord, vertex_feat, protein_length):
        '''
        Args:
            vertex_coord: coordinates of vertecies. [..., dim_vertex, 3]
            vertex_feat: features of vertecies. [..., dim_vertex, dim_vertex_feat]
            length: length of vertecies. [..., 1]
        Returns:
            the representation of the protein. [..., dim_hidden]
        '''
        # Create mask by length
        dim_vertex = vertex_feat.shape[-2]
        dim_vertex_2 = vertex_coord.shape[-2]
        assert dim_vertex == dim_vertex_2, "dim_vertex inconsistent"
        index = torch.arange(dim_vertex).to(self.device)
        mask = (index < protein_length) # [..., dim_vertex], boolean matrix

        edge_feat, neighbor_idx = self.edge_encoder(vertex_coord, mask.float())

        hid_edge_feat = self.edge_linear(edge_feat)
        hid_vertex_feat = self.vertex_linear(vertex_feat)

        # Create mask with neighbor information by neighbor_idx
        mask_attend = gather_nodes(mask.unsqueeze(-1), neighbor_idx).squeeze(-1) # [... , dim_vertex, dim_neighbor, 1], masks padding neighbors.
        mask_attend = mask_attend * mask.unsqueeze(-1) # [..., dim_vertex, dim_neighbor], masks padding vertex.

        code = self.graph_transforemer_encoder(hid_vertex_feat, hid_edge_feat, neighbor_idx, mask_attend) # [..., dim_vertex, dim_neighbor, dim_vertex_feat]

        return code
    

class PPITransformer(nn.Module):
    def __init__(self, dim_edge_feat, dim_vertex_feat, dim_hidden, device=torch.device('cpu')):
        super().__init__()
        self.protein_encoder = ProteinEncoder(dim_edge_feat=dim_edge_feat, dim_vertex_feat=dim_vertex_feat, dim_hidden=dim_hidden, device=device)
        dim_mid = int(dim_hidden/2) + 1
        self.out_layer1 = nn.Linear(in_features=dim_hidden, out_features=dim_mid)
        self.out_layer2 = nn.Linear(in_features=dim_mid, out_features=1)
        self.device = device
    
    def forward(self, vertex_coords, vertex_feats, protein_length):
        '''
        Args:
            vertex_coords: [..., protein, dim_vertex, 3]
            vertex_feats: [..., protein, dim_vertex, dim_vertex_feat]
            protein_length: [..., protein, 1]
        Return:
            Prediction of PPI. [..., 2]
        '''

        protein_code = self.protein_encoder(vertex_coords, vertex_feats, protein_length)    
        
        fuse_code = protein_code.prod(dim=-2) # [..., dim_hidden]

        hidden_1 = self.out_layer1(fuse_code)
        out = self.out_layer2(hidden_1)
        return out
