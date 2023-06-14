'''
This module defines:
An module called EdgeFeature that transforms coordinates of proteins residues
into a code and construct a k nearest neighbor matrix.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .misc import gather_nodes, Normalize


class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings

    def forward(self, E_idx):
        # i-j
        device = E_idx.device
        N_nodes = E_idx.size(-2)
        N_neighbors = E_idx.size(-1)
        ii = torch.arange(N_nodes, dtype=torch.float32).view([1]*(len(E_idx.shape)-2) + [-1, 1]).to(device)
        d = (E_idx.float() - ii).unsqueeze(-1)
        # Original Transformer frequencies
        frequency = torch.exp(
            torch.arange(0, self.num_embeddings, 2, dtype=torch.float32)
            * -(np.log(10000.0) / self.num_embeddings)).to(device)

        angles = d * frequency.view((1,1,1,-1))
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E # [N_batch, N_nodes, N_neighbors, num_embeddings]


class EdgeEncoder(nn.Module):
    def __init__(self, dim_edge_features, num_positional_embeddings=16,
        num_rbf=16, dim_neighbor=30, augment_eps=0.):
        super().__init__()
        self.dim_neighbor = dim_neighbor
        self.augment_eps = augment_eps 
        self.num_rbf = num_rbf

        # Positional encoding
        self.PE = PositionalEncodings(num_positional_embeddings)
        
        # Embedding and normalization
        self.edge_embedding = nn.Linear(num_positional_embeddings + num_rbf + 7, dim_edge_features, bias=True)
        self.norm_edges = Normalize(dim_edge_features)

    def _dist(self, X, mask, eps=1E-6):
        """ Pairwise euclidean distances """
        mask_2D = torch.unsqueeze(mask,-1) * torch.unsqueeze(mask,-2) # mask [..., dim_vertex] => mask_2D [..., dim_vertex, dim_vertex]
        dX = torch.unsqueeze(X,-2) - torch.unsqueeze(X,-3) # X 坐标矩阵 [..., L, 3]   dX 坐标差矩阵 [..., L, L, 3]
        D = mask_2D * torch.sqrt(torch.sum(dX**2, -1) + eps) # 距离矩阵 [..., L, L]

        # setting the masked distances to the max distance
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max

        # Identify k nearest neighbors (including self)
        D_neighbors, E_idx = torch.topk(D_adjust, self.dim_neighbor, dim=-1, largest=False) # [..., L, k]  D_neighbors为具体距离值（从小到大），E_idx为对应邻居节点的编号

        return D_neighbors, E_idx

    def _rbf(self, D):
        '''
        Args:
            D: [..., dim_vertex, dim_neighbor]
        '''
        # Distance radial basis function
        D_min, D_max, D_count = 0., 20., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count).to(D.device) # [D_count]
        D_mu = D_mu.view([1]*(len(D.shape)-1) + [D_count])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)

        return RBF # [B, L, K, self.num_rbf]

    def _quaternions(self, R):
        """ Convert a batch of 3D rotations [R] to quaternions [Q]
            R [...,3,3]
            Q [...,4]
        """
        # Simple Wikipedia version
        # en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        # For other options see math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
              Rxx - Ryy - Rzz, 
            - Rxx + Ryy - Rzz, 
            - Rxx - Ryy + Rzz
        ], -1)))
        #_R = lambda i,j: R[:,:,:,i,j]
        #_R = lambda i,j: R.tensor_split([j,j+1], dim=-1)[1].tensor_split([i,i+1], dim=-2)[1] # R[...,i,j]
        # _R = lambda i,j: R.transpose(0,-1)[j].transpose(0,-1).transpose(1,-2)[i].transpose(1,-2)
        _R = lambda i,j: R[...,i,j]
        signs = torch.sign(torch.stack([
            _R(2,1) - _R(1,2),
            _R(0,2) - _R(2,0),
            _R(1,0) - _R(0,1)
        ], -1))
        xyz = signs * magnitudes
        # The relu enforces a non-negative trace
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)

        return Q

    def _orientations(self, X, E_idx, eps=1e-6):
        # Shifted slices of unit vectors
        #dX = X[:,1:,:] - X[:,:-1,:]
        dX = X.tensor_split([1],dim=-2)[1] - X.tensor_split([-1], dim=-2)[1] # [..., dim_vertex-1, 3]
        U = F.normalize(dX, dim=-1) # 少了第一个（u0）
        #u_2 = U[:,:-2,:] # u 1~n-2
        u_2 = U.tensor_split([-2],dim=-2)[0] # [..., dim_vertex-3, 3]
        #u_1 = U[:,1:-1,:] # u 2~n-1
        u_1 = U.tensor_split([1,-1],dim=-2)[1]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1) # n 1~n-2

        # Build relative orientations
        o_1 = F.normalize(u_2 - u_1, dim=-1) # b 角平分线向量 [..., dim_vertex-3, 3]
        #O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), 2)
        O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), -1) # [..., dim_vertex-3, 9]
        O = O.view(list(O.shape[:-2]) + [9])
        O = F.pad(O, (0,0,1,2), 'constant', 0) # [..., dim_vertex, 9]


        O_neighbors = gather_nodes(O, E_idx) # [..., dim_vertex, dim_neighbor, 9]
        X_neighbors = gather_nodes(X, E_idx) # [..., dim_vertex, dim_neighbor, 3]
        
        # Re-view as rotation matrices
        O = O.view(list(O.shape[:-1]) + [3,3]) # [..., dim_vertex, 3, 3]
        O_neighbors = O_neighbors.view(list(O_neighbors.shape[:-1]) + [3,3]) # [..., dim_vertex, dim_neighbor, 3, 3]

        # Rotate into local reference frames
        dX = X_neighbors - X.unsqueeze(-2) # [..., dim_vertex, dim_neighbor, 3]
        dU = torch.matmul(O.unsqueeze(-3), dX.unsqueeze(-1)).squeeze(-1) # [..., dim_vertex, dim_neighbor, 3]
        dU = F.normalize(dU, dim=-1)
        R = torch.matmul(O.unsqueeze(-3).transpose(-1,-2), O_neighbors) # [..., dim_vertex, dim_neighbor, 3, 3]
        Q = self._quaternions(R) # [..., dim_vertex, dim_neighbor, 4]

        # Orientation features
        O_features = torch.cat((dU,Q), dim=-1) # [..., dim_vertex, dim_neighbor, 7]

        return O_features


    def forward(self, X, mask):
        '''
        Args:
            X: coordinates of each vertex, has shape [..., dim_vertex,3]
            mask: has shape [..., dim_vertex]
        Returns:
            E: edge feature with shape [..., dim_vertex, dim_neighbor, dim_edge_feature]
            E_idx: neighbor index with shape [..., dim_vertex, dim_neighbor]
        '''
        # Data augmentation
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        # Build k-Nearest Neighbors graph
        D_neighbors, E_idx = self._dist(X, mask)

        # Pairwise features
        RBF = self._rbf(D_neighbors)
        O_features = self._orientations(X, E_idx)

        # Pairwise embeddings
        E_positional = self.PE(E_idx)

        E = torch.cat((E_positional, RBF, O_features), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        return E, E_idx
