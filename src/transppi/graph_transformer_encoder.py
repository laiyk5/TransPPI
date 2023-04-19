import torch
from torch import nn
import numpy as np
from .misc import Normalize
from .misc import gather_nodes

class NeighborAttention(nn.Module):
    def __init__(self, dim_hidden, dim_in, num_heads=4, device=torch.device('cpu')):
        super(NeighborAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_hidden = dim_hidden
        self.device = device

        # Self-attention layers: {queries, keys, values, output}
        self.W_Q = nn.Linear(dim_hidden, dim_hidden, bias=False)
        self.W_K = nn.Linear(dim_in, dim_hidden, bias=False)
        self.W_V = nn.Linear(dim_in, dim_hidden, bias=False)
        self.W_O = nn.Linear(dim_hidden, dim_hidden, bias=False)

    def _masked_softmax(self, attend_logits, mask_attend, dim=-1):
        """ Numerically stable masked softmax """
        negative_inf = np.finfo(np.float32).min
        attend_logits = torch.where(mask_attend > 0, attend_logits, torch.tensor(negative_inf).to(self.device))
        attend = nn.functional.softmax(attend_logits, dim) # TODO softmax before mask?
        attend = mask_attend * attend
        return attend

    def forward(self, h_V, h_E, mask_attend=None):
        """
        Args:
            h_V: vertex features, has shape [..., dim_vertex, dim_hidden]
            h_E: neighbor features, has shape [..., dim_vertex, dim_neighbor, dim_hidden]
            mask_attend: has shape [..., dim_vertex, dim_neighbor], 1 for viable value and 0 for padding value.
        Returns:
            Encoded vertex features. [..., dim_vertex, dim_hidden]
        """

        # Queries, Keys, Values
        n_nodes, n_neighbors, _ = h_E.shape[-3:]
        num_high_dim = len(h_V.shape)-2
        high_dim = list(h_V.shape[:num_high_dim])

        n_heads = self.num_heads

        d = int(self.dim_hidden / n_heads)
        Q = self.W_Q(h_V).view(high_dim + [n_nodes, 1, n_heads, 1, d])
        K = self.W_K(h_E).view(high_dim + [n_nodes, n_neighbors, n_heads, d, 1])
        V = self.W_V(h_E).view(high_dim + [n_nodes, n_neighbors, n_heads, d])

        # Attention with scaled inner product
        attend_logits = torch.matmul(Q, K).view(high_dim + [n_nodes, n_neighbors, n_heads]).transpose(-2,-1)
        attend_logits = attend_logits / np.sqrt(d)
        
        if mask_attend is not None:
            # Masked softmax
            mask = mask_attend.unsqueeze(-2).expand([-1]*num_high_dim + [-1,n_heads,-1])
            attend = self._masked_softmax(attend_logits, mask) # [..., L, heads, K]
        else:
            attend = nn.functional.softmax(attend_logits, -1)

        # Attentive reduction
        h_V_update = torch.matmul(attend.unsqueeze(-2), V.transpose(-3,-2)) # [..., L, heads, 1, K] × [..., L, heads, K, d]
        h_V_update = h_V_update.view(high_dim + [n_nodes, self.dim_hidden])
        h_V_update = self.W_O(h_V_update)
        return h_V_update


class PositionWiseFeedForward(nn.Module):
    def __init__(self, dim_hidden, dim_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(dim_hidden, dim_ff, bias=True)
        self.W_out = nn.Linear(dim_ff, dim_hidden, bias=True)

    def forward(self, h_V):
        h = nn.functional.relu(self.W_in(h_V))
        h = self.W_out(h)
        return h


class GraphTransformerEncoderLayer(nn.Module):
    def __init__(self, dim_hidden, dim_in, num_heads=4, dropout=0.2, device=torch.device('cpu')):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([Normalize(dim_hidden) for _ in range(2)])

        self.attention = NeighborAttention(dim_hidden, dim_in, num_heads, device=device)
        self.dense = PositionWiseFeedForward(dim_hidden, dim_hidden * 4)

    def forward(self, h_V, h_E, mask_attend=None):
        '''
        Args:
            h_V: vertex features, has shape [..., dim_pos, dim_neighbor, dim_hidden]
            h_E: neighbor features, has shape [..., dim_pos, dim_neighbor, dim_in]
            mask_attend: has shape [..., dim_pos, dim_neighbor], 1 for viable value and 0 for padding value.
        Returns:
            encoded vertex features, has shape [..., dim_pos, dim_neighbor, dim_hidden]
        '''
        # Neighbor-attention
        dh = self.attention(h_V, h_E, mask_attend)
        h_V = self.norm[0](h_V + self.dropout(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        return h_V


class GraphTransformerEncoder(nn.Module):
    def __init__(self, dim_hidden, num_layer=4, device=torch.device('cpu')):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphTransformerEncoderLayer(dim_hidden, dim_hidden*2, device=device)
            for _ in range(num_layer)
        ])
    
    def forward(self, vertex_feat, edge_feat, neighbor_idx, mask_attend):
        ''' 
        Encode a K-neighbor graph.
        Args:
            vertex_feat: vertex features, has shape [..., dim_vertex, dim_neighbor, dim_hidden]
            edge_feat: neighbor edge features, has shape [..., dim_vertex, dim_neighbor, dim_edge_feat]
            neighbor_idx: neighbor idx, has shape [..., dim_neighbor]
            mask_attend: shape[..., dim_vertex, dim_neighbor]
        Ret:
            encoded graph features, has shape [..., dim_hidden]
        '''
        for layer in self.layers:
            neighbor_vertex_feat = gather_nodes(vertex_feat, neighbor_idx)
            neighbor_feat = torch.cat((edge_feat, neighbor_vertex_feat), dim=-1)
            vertex_feat = layer(vertex_feat, neighbor_feat, mask_attend=mask_attend) # [..., dim_vertex, dim_hidden]
         
        summed_feat = torch.sum(vertex_feat, dim=(-2)) # [..., dim_hidden]
        count_feat = torch.sum(mask_attend, dim=(-1,-2)).unsqueeze(-1) # [..., 1] 
        mean_feat = summed_feat / count_feat # [..., dim_hidden]
        
        return mean_feat
    
