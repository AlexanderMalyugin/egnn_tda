import torch
import torch.nn as nn
from torch.nn import functional as F

from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import scatter


class GCL(MessagePassing):
    """
    E(3) Equivariant Graph Convolution Layer (EGCL)
    """
    def __init__(self, hidden_dim, aggr='add', equivariant=False):
        super().__init__(aggr=aggr, flow="source_to_target")
        self.node_attr_dim = hidden_dim
        self.equivariant = equivariant

        self.mlp_e = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1 + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        self.mlp_x = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.mlp_h = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.mlp_inf = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, h, x, edge_index, edge_attr):

        m_i, dx_i = self.propagate(edge_index, h=h, x=x, edge_attr=edge_attr)

        # Equation 4
        x_new = x + dx_i if self.equivariant else x

        #Equation 6
        h_new = self.mlp_h(torch.cat([h, m_i], dim=-1))
        return h_new, x_new

    def message(self, h_i, h_j, x_i, x_j, edge_attr):

        # Equation 3
        r2 = ((x_j - x_i) ** 2).sum(dim=-1, keepdim=True)
        m = self.mlp_e(torch.cat([h_i, h_j, r2, edge_attr], dim=-1))

        # Equation 4
        if self.equivariant:
            direction = (x_i - x_j)
            coef = self.mlp_x(m)
            dx = direction * coef
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


class EGNN(nn.Module):
    """
    E(3) Equivariant Graph Neural Network (EGNN)
    """
    def __init__(self,
                 node_attr_dim : int,
                 edge_attr_dim : int,
                 hidden_dim : int = 64,
                 num_layers : int = 7,
                 aggr : str = 'add',
                 equivariant : bool = False):

        super().__init__()

        self.gcls = nn.ModuleList()
        for layer in range(num_layers):
            self.gcls.append(GCL(hidden_dim = hidden_dim,
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

        self.mlp_lin_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.node_attr_embedding = torch.nn.Linear(node_attr_dim, hidden_dim)
        self.edge_attr_embedding = torch.nn.Linear(edge_attr_dim, hidden_dim)

    def forward(self, data):

        node_attr = data.node_attr
        coors = data.pos
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        node_attr = self.node_attr_embedding(node_attr)
        edge_attr = self.edge_attr_embedding(edge_attr)

        for conv in self.gcls:
            node_attr_new, coors_new = conv(node_attr, coors, edge_index, edge_attr)

            node_attr = node_attr_new
            coors = coors_new

        node_attr = self.mlp_lin(node_attr)
        h_graph = self.pool(node_attr, data.batch)
        out = self.mlp_lin_pred(h_graph)

        return out.view(-1)

class EGNN_TDA(nn.Module):
    """
    E(3) Equivariant Graph Neural Network + Topological Data Analysis
    """
    def __init__(self,
                 node_attr_dim : int,
                 edge_attr_dim : int,
                 hidden_dim : int = 64,
                 num_layers : int = 7,
                 aggr : str = 'add',
                 equivariant : bool = False):

        super().__init__()

        self.gcls = nn.ModuleList()
        for layer in range(num_layers):
            self.gcls.append(GCL(hidden_dim = hidden_dim,
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
            nn.Linear(hidden_dim + 16, hidden_dim + 16),
            nn.SiLU(),
            nn.Linear(hidden_dim + 16, 1)
        )

        self.node_attr_embedding = torch.nn.Linear(node_attr_dim, hidden_dim)
        self.edge_attr_embedding = torch.nn.Linear(edge_attr_dim, hidden_dim)

        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(),
        )

        self.pi_head = nn.Sequential(
            nn.Linear(16, 16),
        )

        self.pi_drop = nn.Dropout(p=0.3)

    def forward(self, data):

        node_attr = data.node_attr
        coors = data.pos
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        node_attr = self.node_attr_embedding(node_attr)
        edge_attr = self.edge_attr_embedding(edge_attr)

        for conv in self.gcls:
            node_attr_new, coors_new = conv(node_attr, coors, edge_index, edge_attr)

            node_attr = node_attr_new
            coors = coors_new

        node_attr = self.mlp_lin(node_attr)
        h_graph = self.pool(node_attr, data.batch)

        pi = data.pi
        if pi.dim() == 3:
            pi = pi.unsqueeze(0)

        pi = pi.to(h_graph.device).float()

        z = self.conv1(pi)
        z = F.max_pool2d(z, kernel_size=2, stride=2)

        z = z.mean(dim=(-2, -1))
        z = self.pi_head(z)

        h = torch.cat([h_graph, z], dim=-1)
        out = self.lin_pred(h)

        return out.view(-1)