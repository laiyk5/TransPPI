import torch
from torch import nn

def gather_nodes(feature, neighbor_idx):
    '''
        Features [...,N,C] at Neighbor indices [...,N,K] => [...,N,K,C]
    '''
    #shape = [i for i in neighbor_idx.size()] + [feature.size(-1)]
    shape = list(neighbor_idx.shape) + [feature.shape[-1]]
    return torch.gather(feature.unsqueeze(-2).expand(shape),
                        -3,
                        neighbor_idx.unsqueeze(-1).expand(shape))


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


class Normalize(nn.Module): # layernorm https://blog.csdn.net/shanglianlm/article/details/85075706
    def __init__(self, features, epsilon=1e-6):
        super(Normalize, self).__init__()
        self.gain = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x, dim=-1):
        mu = x.mean(dim, keepdim=True)
        sigma = torch.sqrt(x.var(dim, keepdim=True) + self.epsilon)
        gain = self.gain
        bias = self.bias
        # Reshape
        if dim != -1:
            shape = [1] * len(mu.size())
            shape[dim] = self.gain.size()[0]
            gain = gain.view(shape)
            bias = bias.view(shape)
        return gain * (x - mu) / (sigma + self.epsilon) + bias