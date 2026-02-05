from __future__ import annotations

from typing import Callable, Optional

import torch
import torch_geometric
from torch_geometric.datasets import QM9
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
import numpy as np

from ripser import ripser
from scipy.ndimage import gaussian_filter

class MaxAtomsFilter:
    def __init__(self, max_atoms: int):
        self.max_atoms = int(max_atoms)

    def __call__(self, data: Data) -> bool:
        return int(data.num_nodes) <= self.max_atoms


class TDA_transform:
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

        data.PI = self.tda_pi(data)

        return data

    def tda_pi(self, data):

        max_dim = 2
        thresh = 5

        bins = 64
        x_min, x_max = 0.0, thresh
        y_min, y_max = 0.0, thresh

        sigma_px = 1.5

        coors = data.pos.detach().cpu().numpy().astype(np.float64)

        dgms = ripser(coors, maxdim=max_dim, thresh=thresh)["dgms"]
        dgms = [dgms[j][np.isfinite(dgms[j][:, 1])] for j in range(max_dim + 1)]

        pi = np.zeros((max_dim + 1, bins, bins), dtype=np.float32)

        for k in range(max_dim + 1):
            if dgms[k].shape[0] == 0:
                continue

            birth = dgms[k][:, 0]
            death = dgms[k][:, 1]

            x = birth
            y = death - birth

            H, _, _ = np.histogram2d(
                x, y,
                bins=bins,
                range=[[x_min, x_max], [y_min, y_max]],
                weights=None
            )

            img = H.T.astype(np.float32)
            img = gaussian_filter(img, sigma=sigma_px, mode="constant")

            pi[k] = img

        pi = torch.from_numpy(pi)
        pi = pi.unsqueeze(0)

        return pi


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
            pre_filter=MaxAtomsFilter(max_atoms),
            pre_transform=TDA_transform(node_attr_indices, y_index),
            force_reload=force_reload,
        )