# fuzzy_mmc_torch.py
from __future__ import annotations
import torch
from torch import Tensor
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt


class FuzzyMMC_Torch:
    """
    Minimal‑Fuzzy‑Min‑Max‑Classifier in PyTorch.
    * harte Expansion zweier Hyperbox‑Eckpunkte (min/max)
    * Overlap‑Kontraktion nur zwischen Klassen
    * reine Torch‑Operationen  →  .to(device) möglich
    """

    def __init__(self, sensitivity: float = 1.0, exp_bound: float = 1.0,
                 device: torch.device | str = "cpu"):
        self.sens   = sensitivity   # γ in Original‑Literatur
        self.bound  = exp_bound     # θ  (max. Kantenlänge × #classes)
        self.device = torch.device(device)

        self.boxes:   Optional[Tensor] = None     # Shape [B, 2, D]  (min,max)
        self.labels:  Optional[Tensor] = None     # Shape [B]


    def _membership(self, pattern: Tensor) -> Tensor:
        """pattern: [D]  →  Membership jedes Hyperbox‐Prototyps  [B]"""
        v, w = self.boxes[:, 0], self.boxes[:, 1]         # [B,D]
        a = torch.clamp(1 - self.sens * torch.clamp(pattern - w, min=0.0), 0.0)
        b = torch.clamp(1 - self.sens * torch.clamp(v - pattern, min=0.0), 0.0)
        return (a + b).sum(1) / (2 * pattern.numel())     # [B]
    
    def _contract(self, idx_new: int) -> None:
        vj, wj = self.boxes[idx_new]                      # [2,D]
        for k in range(len(self.boxes)):
            if self.labels[k] == self.labels[idx_new]:
                continue                                  # gleiche Klasse → ignorieren
            vk, wk = self.boxes[k]

            # find min Überlap
            delta = torch.full((vj.size(0),), 1e9, device=self.device)
            cond1 = (vj < vk) & (vk < wj) & (wj < wk)
            cond2 = (vk < vj) & (vj < wk) & (wk < wj)
            cond3 = (vj < vk) & (vk < wk) & (wk < wj)
            cond4 = (vk < vj) & (vj < wj) & (wj < wk)

            # Delta‐Kandidaten
            delta[cond1] = (wj - vk)[cond1]
            delta[cond2] = (wk - vj)[cond2]
            delta[cond3] = torch.minimum(
                (wj - vk)[cond3], (wk - vj)[cond3])
            delta[cond4] = torch.minimum(
                (wk - vj)[cond4], (wj - vk)[cond4])

            if delta.min() == 1e9:
                continue  # keine Überlappung

            i = int(delta.argmin())       # min. Überlappung
            # 4 case
            if   vj[i] < vk[i] < wj[i] < wk[i]:
                vk[i] = wj[i] = (vk[i] + wj[i]) / 2
            elif vk[i] < vj[i] < wk[i] < wj[i]:
                vj[i] = wk[i] = (vj[i] + wk[i]) / 2
            elif vj[i] < vk[i] < wk[i] < wj[i]:
                if (wj[i]-vk[i]) > (wk[i]-vj[i]): vj[i] = wk[i]
                else:                           wj[i] = vk[i]
            elif vk[i] < vj[i] < wj[i] < wk[i]:
                if (wk[i]-vj[i]) > (wj[i]-vk[i]): vk[i] = wj[i]
                else:                            wk[i] = vj[i]

            # aktualisieren
            self.boxes[idx_new] = torch.stack([vj, wj])
            self.boxes[k]       = torch.stack([vk, wk])


    def _train_single(self, x: Tensor, y: int):
        if self.labels is None or (self.labels == y).sum() == 0:
            # -> neue Hyperbox
            box = x.repeat(2, 1)                        # [2,D]  (min=max=x)
            self.boxes  = torch.cat([self.boxes, box.unsqueeze(0)], dim=0) if self.boxes is not None else box.unsqueeze(0)
            self.labels = torch.cat([self.labels, torch.tensor([y], device=self.device)]) if self.labels is not None else torch.tensor([y], device=self.device)
            return

        # sonst: expand
        mem = self._membership(x)                       # [B]
        mem[self.labels != y] = -1                      # andere Klassen ignorieren
        idx = int(mem.argmax())                         # Index Box mit höchster Membership

        v_old, w_old = self.boxes[idx]
        v_new = torch.minimum(v_old, x)
        w_new = torch.maximum(w_old, x)

        if (w_new - v_new).sum() <= self.bound * self.labels.unique().numel():
            # Expansion zulässig
            self.boxes[idx] = torch.stack([v_new, w_new])
        else:
            # neue Box anlegen
            self.boxes  = torch.cat([self.boxes, x.repeat(2,1).unsqueeze(0)], dim=0)
            self.labels = torch.cat([self.labels, torch.tensor([y], device=self.device)])
            idx = len(self.boxes) - 1

        # Kontraktion gegen andere Klassen
        self._contract(idx)


    def fit(self, X: Tensor, y: Tensor, epochs: int = 1):
        X, y = X.to(self.device), y.to(self.device)
        self.boxes  = None
        self.labels = None
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                self._train_single(xi, int(yi))

    def predict(self, x: Tensor) -> Tuple[float, int]:
        mem = self._membership(x.to(self.device))
        best_idx = int(mem.argmax())
        return float(mem[best_idx]), int(self.labels[best_idx])

    def score(self, X: Tensor, y: Tensor) -> float:
        X, y = X.to(self.device), y.to(self.device)
        correct = sum(self.predict(x)[1] == yi.item() for x, yi in zip(X, y))
        return correct / len(y)


    def plot_boxes(self, X: Optional[Tensor] = None, dims: Tuple[int,int]=(0,1)):
        import matplotlib.pyplot as plt
        i, j = dims
        plt.figure(figsize=(6,5))
        if X is not None:
            X_np = X.cpu().numpy()
            plt.scatter(X_np[:,i], X_np[:,j], s=10, alpha=0.3)

        for (v, w), cls in zip(self.boxes.cpu(), self.labels.cpu()):
            plt.gca().add_patch(plt.Rectangle((v[i], v[j]),
                                              w[i]-v[i], w[j]-v[j],
                                              fill=False))
            plt.text(v[i], v[j], str(int(cls)))
        plt.xlabel(f"f{i}")
        plt.ylabel(f"f{j}")
        plt.title("Fuzzy MMC Hyperboxes")
        plt.tight_layout(); plt.show()


if __name__ == "__main__":
    from data_utils import load_iris_data, load_abalon_data
    Xtr, ytr, Xte, yte,_ = load_abalon_data()
    fmmc = FuzzyMMC_Torch(exp_bound=1.5)
    fmmc.fit(Xtr, ytr, epochs=300)
    print("Test‑Acc:", fmmc.score(Xte, yte))
    fmmc.plot_boxes(Xtr, dims=(0,2))
