from __future__ import annotations

from typing import Callable, Optional

import torch
import torch_geometric
from torch_geometric.datasets import QM9
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from rdkit import Chem
import numpy as np

class MaxAtomsFilter(object):

    def __init__(self, max_atoms):
        self.max_atoms = max_atoms

    def __call__(self, data):
        if data.x.shape[0] <= self.max_atoms:
            return True
        else:
            return False

class RemoveFields(object):

    def __init__(self, fields):
        self.fields = fields

    def __call__(self, data):
        for field in self.fields:
            data.__delattr__(field)
        return data

class RemoveX(object):

    def __init__(self, fields):
        self.fields = fields

    def __call__(self, data):
        for field in self.fields:
            data.__delattr__(field)
        return data


class _QM9Dataset(torch_geometric.datasets.QM9):

    NUM_UNIQUE_ATOMS = 5
    PROPERTIES = { "mu" : 0, "alpha" : 1, "homo" : 2, "lumo" : 3, "gap" : 4, "r^2" : 5, "zvpe" : 6, "u_0" : 7, "u" : 8, "h" : 9, "g" : 10, "c_v" : 11, "u_atom_0" : 12, "u_atom" : 13, "h_atom" : 14, "g_atom" : 15, "a" : 16,  "b" : 17,  "c" : 18}

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        target_y: str = None,
    ):
        super().__init__(
            root=root,
            transform=transform,
            pre_filter=pre_filter,
            pre_transform=pre_transform,
        )

        self.node_attr = self.x[:, [1,2,3,4,5,10]]
        self.y = self.y[:, self.PROPERTIES[target_y]]

class _MaxAtomsFilter:
    def __init__(self, max_atoms: int):
        self.max_atoms = int(max_atoms)

    def __call__(self, data: Data) -> bool:
        return int(data.num_nodes) <= self.max_atoms


class _AddNodeAttrAndSelectY:
    def __init__(self, feat_indices: list[int], y_index: int):
        self.feat_indices = feat_indices
        self.y_index = int(y_index)

    def __call__(self, data: Data) -> Data:
        # x expected shape: [N, F]
        x = data.x
        if x is None or x.dim() != 2:
            raise ValueError(f"Expected data.x to be 2D [N,F], got {None if x is None else tuple(x.shape)}")

        F = x.size(1)

        # resolve negative indices like -1
        idx = []
        for i in self.feat_indices:
            ii = i if i >= 0 else F + i
            if ii < 0 or ii >= F:
                raise IndexError(f"Feature index {i} out of bounds for F={F}")
            idx.append(ii)

        idx_t = torch.tensor(idx, dtype=torch.long)
        data.node_attr = x.index_select(1, idx_t)     # [N, len(idx)]

        # QM9: data.y is typically shape [19] (or [1,19] depending on version)
        y = data.y
        if y is None:
            return data

        if y.dim() == 2 and y.size(0) == 1:
            y = y.squeeze(0)  # -> [19]

        if y.dim() != 1:
            raise ValueError(f"Expected data.y to be 1D [T], got {tuple(y.shape)}")

        data.y = y[self.y_index].view(1)             # -> [1]
        return data


class QM9Dataset(QM9):

    PROPERTIES = { "mu" : 0, "alpha" : 1, "homo" : 2, "lumo" : 3, "gap" : 4, "r^2" : 5, "zvpe" : 6, "u_0" : 7, "u" : 8, "h" : 9, "g" : 10, "c_v" : 11, "u_atom_0" : 12, "u_atom" : 13, "h_atom" : 14, "g_atom" : 15, "a" : 16,  "b" : 17,  "c" : 18}

    def __init__(
        self,
        root: str,
        max_atoms: int,
        target_y: str = "gap",
        node_attr_indices: list[int] = [1,2,3,4,5,10],
        transform: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        if target_y not in self.PROPERTIES:
            raise KeyError(f"Unknown target_y='{target_y}'. Available: {list(self.PROPERTIES.keys())}")

        y_index = self.PROPERTIES[target_y]

        super().__init__(
            root=root,
            transform=transform,
            pre_filter=_MaxAtomsFilter(max_atoms),
            pre_transform=_AddNodeAttrAndSelectY(node_attr_indices, y_index),
            force_reload=force_reload,
        )