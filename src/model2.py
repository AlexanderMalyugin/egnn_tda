import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_geometric.utils import scatter
from torch_geometric.typing import Adj, Size, OptTensor, Tensor

class E3GCL(MessagePassing):
    """
    E(3) Graph Convolution Layer (EGCL)
    """
    def __init__(self, node_attrs, edge_dim =3, aggr='add', equivariant=False):
        super().__init__(aggr=aggr)
        self.node_attrs = node_attrs
        self.edge_dim = edge_dim
        self.equivariant = equivariant

        self.mlp_e = nn.Sequential(
            nn.Linear(node_attrs*2+1+edge_dim, node_attrs),
            # self.dropout,
            nn.SiLU(),
            nn.Linear(node_attrs, node_attrs),
            nn.SiLU()
        )

        self.mlp_x = nn.Sequential(
            nn.Linear(node_attrs*2+1+edge_dim, node_attrs),
            # self.dropout,
            nn.SiLU(),
            nn.Linear(node_attrs, node_attrs)
        )

        self.mlp_h = nn.Sequential(
            nn.Linear(node_attrs*2+1+edge_dim, node_attrs),
            # self.dropout,
            nn.SiLU(),
            nn.Linear(node_attrs, node_attrs)
        )

        self.mlp_inf = nn.Sequential(
            nn.Linear(node_attrs*2+1+edge_dim, node_attrs),
            # self.dropout,
            nn.Sigmoid(),
        )

    def message(self, h_i, h_j, x_i, x_j, edge_attr) -> Tensor:

        # Equation 3
        r_ij = torch.linalg.norm(x_j-x_i, dim=-1, keepdim=True)
        m_ij = self.mlp_e(torch.cat([h_i, h_j, r_ij**2, edge_attr], dim=-1))

        #attn = self.mlp_attn_msg_h(m_ij)
        #msg_h = attn * m_ij

        return m_ij

    def propagate(self,
                  edge_index,
                  node_attrs,
                  edge_attr,
                  coords,
                  rel_coords,
                  size = None,
                  **kwargs):

        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._user_args,
                                     edge_index, size, kwargs)
        msg_kwargs = self.inspector.collect_param_data('message', coll_dict)
        aggr_kwargs = self.inspector.collect_param_data('aggregate', coll_dict)
        update_kwargs = self.inspector.collect_param_data('update', coll_dict)

        m_ij = self.message(**msg_kwargs)

        # Equation 4
        if self.equivariant:

            msg_x = rel_coords * self.mlp_msg_x(m_ij)
            coords_out = kwargs["coords"] + self.aggregate(msg_x, **aggr_kwargs)

        else :
            coords_out = kwargs["coords"]

        # Equation 5
        m_i = self.aggregate(m_ij, **aggr_kwargs)

        # Equation 6
        h_out = self.mlp_h(torch.cat([node_attrs, m_i], dim=-1))
        #h_out = h_i + h_out

        return self.update((coords_out, h_out), **update_kwargs)

    def forward(self, node_attr, coords, edge_index, edge_attr):

        rel_coords = coords[edge_index[0]] - coords[edge_index[1]]
        rel_dist = (rel_coodrs ** 2).sum(dim=-1, keepdim=True)

        edge_attr_feats = torch.cat([edge_attr, rel_dist], dim=-1)

        h_out, coords_out = self.propagate(edge_index, node_attr=node_attr, edge_attr=edge_attr_feats,
                                               coords=coords, rel_coors=rel_coords)

        return torch.cat([h_out, coords_out], dim=-1)


