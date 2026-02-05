from __future__ import annotations

from typing import Callable, Optional

from torch_geometric.datasets import QM9
from torch_geometric.data import Data
from torch_geometric.transforms import Compose

import numpy as np

from ripser import ripser
from scipy.ndimage import gaussian_filter

import torch
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType as HYB


class DropFields:
    def __init__(self, *keys):
        self.keys = keys
    def __call__(self, data):
        for k in self.keys:
            if hasattr(data, k):
                delattr(data, k)
        return data


class RDKitAromaticSPPreTransform:
    def __call__(self, data):
        N = int(data.num_nodes)

        mol = Chem.RWMol()
        for z in data.z.tolist():
            mol.AddAtom(Chem.Atom(int(z)))

        bt_map = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
                  Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]

        ei = data.edge_index.t().tolist()
        ea = data.edge_attr.tolist() if getattr(data, "edge_attr", None) is not None else None

        for k, (u, v) in enumerate(ei):
            if u >= v:
                continue

            btype = Chem.BondType.SINGLE
            if ea is not None:
                btype = bt_map[int(torch.tensor(ea[k]).argmax().item())]

            mol.AddBond(int(u), int(v), btype)

            if btype == Chem.BondType.AROMATIC:
                mol.GetAtomWithIdx(int(u)).SetIsAromatic(True)
                mol.GetAtomWithIdx(int(v)).SetIsAromatic(True)

        mol = mol.GetMol()
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return data

        aromatic = torch.zeros(N, dtype=torch.float32)
        sp  = torch.zeros(N, dtype=torch.float32)
        sp2 = torch.zeros(N, dtype=torch.float32)
        sp3 = torch.zeros(N, dtype=torch.float32)

        for i, a in enumerate(mol.GetAtoms()):
            aromatic[i] = float(a.GetIsAromatic())
            h = a.GetHybridization()
            if h == HYB.SP: sp[i] = 1.0
            elif h == HYB.SP2: sp2[i] = 1.0
            elif h == HYB.SP3: sp3[i] = 1.0

        data.x[:, 6] = aromatic
        data.x[:, 7] = sp
        data.x[:, 8] = sp2
        data.x[:, 9] = sp3
        return data


class MaxAtomsFilter:
    def __init__(self, max_atoms: int):
        self.max_atoms = int(max_atoms)

    def __call__(self, data: Data) -> bool:
        return int(data.num_nodes) <= self.max_atoms


class Add_node_attrs:

    PROPERTIES = { "mu" : 0, "alpha" : 1, "homo" : 2, "lumo" : 3, "gap" : 4, "r^2" : 5, "zvpe" : 6, "u_0" : 7, "u" : 8, "h" : 9, "g" : 10, "c_v" : 11, "u_atom_0" : 12, "u_atom" : 13, "h_atom" : 14, "g_atom" : 15, "a" : 16,  "b" : 17,  "c" : 18}

    def __init__(self,
                 node_attr_indices: list[int] = [1,2,3,4,6,7,8,9,10],
                 target_y: str = "gap"
                 ):

        self.node_attr_indices = node_attr_indices

        if target_y not in self.PROPERTIES:
            raise KeyError(f"Unknown target_y='{target_y}'. Available: {list(self.PROPERTIES.keys())}")

        self.y_index = self.PROPERTIES[target_y]

    def __call__(self, data: Data) -> Data:
        # x expected shape: [N, F]
        x = data.x
        if x is None or x.dim() != 2:
            raise ValueError(f"Expected data.x to be 2D [N,F], got {None if x is None else tuple(x.shape)}")

        F = x.size(1)

        # resolve negative indices like -1
        idx = []
        for i in self.node_attr_indices:
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

class TDA_transform:
    def __init__(self, max_dim: int = 1,
                 threshold : float = 5,
                 x_max : float = None,
                 y_max : float = None,
                 bins : int = 64,
                 x_min : float = 0,
                 y_min : float = 0,
                 sigma_px : float = 1,
                 ):

        self.max_dim = max_dim
        self.threshold = threshold

        if x_max is None:
            self.x_max = threshold
        else:
            self.x_max = x_max

        if y_max is None:
            self.y_max = threshold
        else:
            self.y_max = y_max

        self.bins = bins
        self.x_min = x_min
        self.y_min = y_min
        self.sigma_px = sigma_px

    def __call__(self, data: Data) -> Data:

        coors = data.pos.detach().cpu().numpy().astype(np.float64)

        dgms = ripser(coors,
                      maxdim = self.max_dim,
                      thresh = self.threshold
                      )["dgms"]

        dgms = [dgms[j][np.isfinite(dgms[j][:, 1])] for j in range(self.max_dim + 1)]

        pi = np.zeros((self.max_dim + 1, self.bins, self.bins), dtype=np.float32)

        for k in range(self.max_dim + 1):
            if dgms[k].shape[0] == 0:
                continue

            birth = dgms[k][:, 0]
            death = dgms[k][:, 1]

            x = birth
            y = death - birth

            H, _, _ = np.histogram2d(
                x, y,
                bins = self.bins,
                range=[[self.x_min, self.x_max], [self.y_min, self.y_max]],
                weights=None
            )

            img = H.T.astype(np.float32)
            img = gaussian_filter(img, sigma=self.sigma_px, mode="constant")

            pi[k] = img

        pi = torch.from_numpy(pi)
        pi = pi.unsqueeze(0)

        data.pi = pi

        return data


class QM9Dataset(QM9):
    def __init__(
        self,
        root: str,
        transform_list: Optional[list[Callable]] = None,
        pre_transform_list: Optional[list[Callable]] = None,
        pre_filter_list: Optional[list[Callable]] = None,
        force_reload: bool = False,
    ):
        pre_filter = None
        if pre_filter_list:
            if len(pre_filter_list) == 1:
                pre_filter = pre_filter_list[0]
            else:
                def pre_filter(data):
                    return all(f(data) for f in pre_filter_list)

        super().__init__(
            root=root,
            transform=Compose(transform_list or []),
            pre_transform=Compose(pre_transform_list or []),
            pre_filter=pre_filter,
            force_reload=force_reload,
        )