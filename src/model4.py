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
    def __init__(self, hidden_dim, edge_dim=0, aggr='add', equivariant=False):
        super().__init__(aggr=aggr, flow="source_to_target")
        self.node_attr_dim = hidden_dim
        self.edge_dim = edge_dim
        self.equivariant = equivariant

        self.mlp_e = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1 + edge_dim, hidden_dim),
            # self.dropout,
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        self.mlp_x = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            # self.dropout,
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.mlp_h = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            # self.dropout,
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.mlp_inf = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, h, x, edge_index, edge_attr):
        # h: [N, hidden_dim], x: [N, 3], edge_attr: [E, edge_dim] or None

        m_i, dx_i = self.propagate(edge_index, h=h, x=x, edge_attr=edge_attr)

        # Equation 4
        x_new = x + dx_i if self.equivariant else x

        #Equation 6
        h_new = self.mlp_h(torch.cat([h, m_i], dim=-1))
        return h_new, x_new

    def message(self, h_i, h_j, x_i, x_j, edge_attr):

        # Equation 3
        r2 = ((x_j - x_i) ** 2).sum(dim=-1, keepdim=True)  # [E,1]
        m = self.mlp_e(torch.cat([h_i, h_j, r2, edge_attr], dim=-1))  # [E, H]

        gate = self.mlp_inf(m)  # [E,1]
        m = gate * m

        # Equation 4
        if self.equivariant:
            direction = (x_i - x_j)  # [E,3]
            coef = self.mlp_x(m)  # [E,1]
            dx = direction * coef  # [E,3]
        else:
            dx = None

        return m, dx

    def aggregate(self, inputs, index, dim_size=None):
        m, dx = inputs

        # Equation 5
        m_i = scatter(m, index, dim=0, dim_size=dim_size, reduce="sum")
        if dx is None:
            dx_i = None
        else:
            dx_i = scatter(dx, index, dim=0, dim_size=dim_size, reduce="sum")
        return m_i, dx_i


class E3GG(nn.Module):
    """
    E(3) Equivariant Graph Neural Network
    """
    def __init__(self, node_attr_dim, edge_dim, hidden_dim, num_layers=4, aggr='add', equivariant=False):
        super().__init__()

        self.GCL = nn.ModuleList()
        for layer in range(num_layers):
            self.GCL.append(E3GCL(hidden_dim = hidden_dim,
                                  edge_dim = edge_dim,
                                  aggr = aggr,
                                  equivariant = equivariant
                                  )
                            )

        self.pool = global_mean_pool
        self.mlp_lin = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.lin_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.embedding = torch.nn.Linear(node_attr_dim, hidden_dim)

    def forward(self, data):

        node_attr = data.node_attr
        coors = data.pos
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        node_attr = self.embedding(node_attr)
        #edge_attr = self.embedding(edge_attr)

        for conv in self.GCL:
            node_attr_new, coors_new = conv(node_attr, coors, edge_index, edge_attr)

            node_attr = node_attr_new
            coors = coors_new

        node_attr = self.mlp_lin(node_attr)
        h_graph = self.pool(node_attr, data.batch) # (n, d) -> (batch_size, d)
        out = self.lin_pred(h_graph)

        return out.view(-1)