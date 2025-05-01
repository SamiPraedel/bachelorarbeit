from __future__ import annotations
from data_utils import load_iris_data, load_K_chess_data_splitted, load_abalon_data, load_Kp_chess_data

"""PyTorch implementation of a very small Fuzzy Decision Tree (numerical features only).

Differences to the NumPy version:
- Entirely Torch‑tensor based → runs on CPU or GPU.
- Membership function can be switched between *triangular* and *gaussian* via constructor.
- Vectorised entropy calculation for speed.

Intentionally minimal; feel free to extend with more fuzzy sets, soft‑routing, pruning …
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Literal

import torch

_EPS = 1e-9

# ---------------------------------------------------------------------------
# Membership functions
# ---------------------------------------------------------------------------

def _triangular_mf(x: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
    """Triangular membership function (a ≤ b ≤ c)."""
    return torch.clamp(torch.minimum((x - a) / (b - a + _EPS), (c - x) / (c - b + _EPS)), min=0.0)


def _gaussian_mf(x: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    """Gaussian (bell‑shaped) membership function."""
    return torch.exp(-0.5 * ((x - mu) / (sigma + _EPS)) ** 2)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _fuzzy_entropy(y_memberships: torch.Tensor) -> torch.Tensor:
    """Fuzzy entropy (smaller ⇒ purer)."""
    totals = y_memberships.sum(dim=0, keepdim=True) + _EPS
    probs = y_memberships / totals
    return -(probs * torch.log2(probs + _EPS)).sum()


@dataclass
class _FDTNode:
    feature: Optional[int] = None            # None ⇒ leaf
    mf_params: Optional[List[Tuple[float, ...]]] = None  # per fuzzy set
    children: Optional[Dict[int, "_FDTNode"]] = None
    class_label: Optional[int] = None        # only for leaves


class FuzzyDecisionTreeTorch:
    """Very small fuzzy decision tree implemented with PyTorch."""

    def __init__(
        self,
        max_depth: int = 3,
        min_samples_split: int = 10,
        membership_shape: Literal["triangular", "gaussian"] = "triangular",
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.membership_shape = membership_shape
        self.root: Optional[_FDTNode] = None
        self.n_classes_: int = 0

        # pick mf function
        if membership_shape == "triangular":
            self._mf = _triangular_mf
        elif membership_shape == "gaussian":
            self._mf = _gaussian_mf
        else:
            raise ValueError("membership_shape must be 'triangular' or 'gaussian'")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        if not torch.is_floating_point(X):
            X = X.float()
        y = y.long()
        self.n_classes_ = int(torch.max(y).item()) + 1
        self.root = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.predict_proba(X), dim=1)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(X):
            X = X.float()
        proba = torch.zeros((X.size(0), self.n_classes_), dtype=torch.float32, device=X.device)
        for i, x in enumerate(X):
            node = self.root
            while node and node.feature is not None:
                if self.membership_shape == "triangular":
                    mf_vals = [
                        self._mf(x[node.feature], *node.mf_params[0]),
                        self._mf(x[node.feature], *node.mf_params[1]),
                    ]
                else:  # gaussian uses (mu, sigma)
                    mf_vals = [
                        self._mf(x[node.feature], *node.mf_params[0]),
                        self._mf(x[node.feature], *node.mf_params[1]),
                    ]
                child_idx = int(torch.argmax(torch.tensor(mf_vals)))
                node = node.children[child_idx]
            proba[i, node.class_label] = 1.0
        return proba

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _best_split(
        self, X: torch.Tensor, y: torch.Tensor
    ) -> Tuple[Optional[int], Optional[List[Tuple[float, ...]]]]:
        best_feat, best_entropy, best_params = None, math.inf, None
        for feat in range(X.size(1)):
            col = X[:, feat]
            if self.membership_shape == "triangular":
                a, b, c = torch.quantile(col, torch.tensor([0.1, 0.5, 0.9])).tolist()
                memberships = torch.vstack([
                    self._mf(col, a, b, c),
                    self._mf(col, c, (c + b) / 2, torch.max(col).item()),
                ])
            else:  # gaussian
                #mu = torch.median(col).item()
                mu = col.median().item() 
                #sigma = torch.quantile(torch.abs(col - mu), torch.tensor(0.68)).item()  # ≈ std
                #sigma = 1,5 * torch.std(col)  # statt robuster 0.68-Quantil
                sigma = 0.884 * col.std() 
                   
                memberships = torch.vstack([
                    self._mf(col, mu - sigma, sigma),  # left bell
                    self._mf(col, mu + sigma, sigma),  # right bell
                ])

            # Map sample‑level memberships to class memberships
            y_mem = torch.zeros((len(col), self.n_classes_), device=X.device)
            for cls in range(self.n_classes_):
                mask = (y == cls).float()
                y_mem[:, cls] = memberships[0] * mask + memberships[1] * mask
            ent = _fuzzy_entropy(y_mem)
            if ent < best_entropy:
                best_feat, best_entropy = feat, ent.item()
                if self.membership_shape == "triangular":
                    best_params = [(a, b, c), (c, (c + b) / 2, torch.max(col).item())]
                else:
                    best_params = [(mu - sigma, sigma), (mu + sigma, sigma)]
        return best_feat, best_params

    def _build_tree(self, X: torch.Tensor, y: torch.Tensor, depth: int) -> _FDTNode:
        # Stopping criteria
        if (
            depth >= self.max_depth
            or X.size(0) < self.min_samples_split
            or torch.unique(y).numel() == 1
        ):
            class_label = int(torch.bincount(y).argmax().item())
            return _FDTNode(class_label=class_label)        

        feat, mf_params = self._best_split(X, y)
        if feat is None:
            class_label = int(torch.bincount(y).argmax().item())
            return _FDTNode(class_label=class_label)

        # Assign samples to fuzzy child (take max membership index)
        if self.membership_shape == "triangular":
            memberships = torch.vstack(
                [
                    self._mf(X[:, feat], *mf_params[0]),
                    self._mf(X[:, feat], *mf_params[1]),
                ]
            )
        else:
            memberships = torch.vstack(
                [
                    self._mf(X[:, feat], *mf_params[0]),
                    self._mf(X[:, feat], *mf_params[1]),
                ]
            )
        child_idx = torch.argmax(memberships, dim=0)

        children: Dict[int, _FDTNode] = {}
        for idx in (0, 1):
            mask = child_idx == idx
            if not mask.any():
                # Empty child → leaf with parent majority
                class_label = int(torch.bincount(y).argmax().item())
                children[idx] = _FDTNode(class_label=class_label)
            else:
                children[idx] = self._build_tree(X[mask], y[mask], depth + 1)

        return _FDTNode(feature=feat, mf_params=mf_params, children=children)


# ---------------------------------------------------------------------------
# Quick sanity check (only executed when running the file directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # toy data: first feature determines class
    X = torch.rand(300, 4)
    y = (X[:, 0] > 0.5).long()
   # X, y, X_t, y_t = load_iris_data()
   # X, y, X_t, y_t = load_K_chess_data_splitted()
    #X, y, X_t, y_t = load_K_chess_data_splitted()
    #X, y, X_t, y_t,_ = load_abalon_data()
    X, y, X_t, y_t,_ = load_Kp_chess_data()
    

    fdt = FuzzyDecisionTreeTorch(max_depth=2, membership_shape="triangular")
    fdt.fit(X, y)
    acc = (fdt.predict(X_t) == y_t).float().mean().item()
    print(f"Training accuracy (triangular): {acc:.3f}")

    fdt_gauss = FuzzyDecisionTreeTorch(max_depth=1, membership_shape="gaussian")
    fdt_gauss.fit(X, y)
    acc2 = (fdt_gauss.predict(X_t) == y_t).float().mean().item()
    print(f"Training accuracy (gaussian): {acc2:.3f}")
