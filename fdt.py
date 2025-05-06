# simple_fuzzy_tree.py
import torch, math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal

_EPS = 1e-9  # numerische Stabilisierung


# ---------------------- MF‑Funktionen -----------------------
def triangular(x: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
    """Dreiecks‑MF."""
    return torch.clamp(torch.minimum((x - a) / (b - a + _EPS),
                                     (c - x) / (c - b + _EPS)), 0.0)


def gaussian(x: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    """Gauß‑MF."""
    return torch.exp(-0.5 * ((x - mu) / (sigma + _EPS)) ** 2)



def fuzzy_entropy(m: torch.Tensor) -> float:
    """m: [N, C] Zugehörigkeiten je Klasse."""
    p = m / (m.sum(0, keepdim=True) + _EPS)
    return float(-(p * torch.log2(p + _EPS)).sum())


@dataclass
class Node:
    feat:    Optional[int]                     = None
    params:  Optional[List[Tuple[float, ...]]] = None  # MF‑Parameter
    child:   Optional[Dict[int, "Node"]]       = None
    label:   Optional[int]                     = None  # nur Blatt



class FuzzyTree:
    """
    Simpler Fuzzy‑Decision‑Tree mit 2 Fuzzy‑Sets (low/high) pro Merkmal
    und hartem Routing (argmax).
    """

    def __init__(self, max_depth=3, min_samples=10,
                 shape: Literal["tri", "gauss"]="tri"):
        self.max_depth  = max_depth
        self.min_samples= min_samples
        self.mf         = triangular if shape == "tri" else gaussian
        self.root: Node = None
        self.C          = 0                      # # Klassen


    def fit(self, X: torch.Tensor, y: torch.Tensor):
        X, y = X.float(), y.long()
        self.C = int(y.max()) + 1
        self.root = self._grow(X, y, 0)
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return torch.stack([self._predict_one(x) for x in X])


    def _grow(self, X, y, depth) -> Node:
        # Stopp‑Kriterien
        if (depth >= self.max_depth
                or len(X) < self.min_samples
                or y.unique().numel() == 1):
            return Node(label=int(torch.bincount(y).argmax()))

        feat, params = self._best_split(X, y)
        if feat is None:                         # Fallback
            return Node(label=int(torch.bincount(y).argmax()))

        # hartes Routing
        m  = torch.vstack([self._mf_col(X[:, feat], p) for p in params])  # [2, N]
        idx = m.argmax(0)                                                # [N]

        children = {
            i: self._grow(X[idx == i], y[idx == i], depth + 1) if (idx == i).any()
               else Node(label=int(torch.bincount(y).argmax()))
            for i in (0, 1)
        }
        return Node(feat, params, children)


    def _mf_col(self, col: torch.Tensor, p: Tuple[float, ...]) -> torch.Tensor:
        return self.mf(col, *p)

    def _best_split(self, X, y):
        best_e, best_feat, best_p = math.inf, None, None
        for f in range(X.size(1)):
            col = X[:, f]
            # je nach MF‑Typ 2 Zonen bauen
            if self.mf is triangular:
                a, b, c = torch.quantile(col, torch.tensor([.1, .5, .9])).tolist()
                params  = [(a, b, c), (c, (c+b)/2, col.max().item())]
            else:
                mu = col.median().item()
                sig= 0.884 * col.std()
                params  = [(mu-sig, sig), (mu+sig, sig)]

            m  = torch.vstack([self._mf_col(col, p) for p in params])
            y_m= torch.zeros(len(col), self.C)
            for c in range(self.C):
                y_m[:, c] = m[0]*(y==c) + m[1]*(y==c)
            ent = fuzzy_entropy(y_m)
            if ent < best_e:
                best_e, best_feat, best_p = ent, f, params
        return best_feat, best_p

    def _predict_one(self, x: torch.Tensor) -> torch.Tensor:
        node = self.root
        while node.feat is not None:
            m0 = self._mf_col(x[node.feat], node.params[0])
            m1 = self._mf_col(x[node.feat], node.params[1])
            node = node.child[int(m1 > m0)]
        return torch.tensor(node.label)


if __name__ == "__main__":
    from data_utils import load_Kp_chess_data
    X = torch.rand(300, 4)
    y = (X[:, 0] > 0.5).long()
    X, y, X_t, y_t,_ = load_Kp_chess_data()
    tree = FuzzyTree(max_depth=8, shape="tri").fit(X, y)
    acc = (tree.predict(X_t) == y_t).float().mean().item()
    print("Train‑Acc:", acc)
