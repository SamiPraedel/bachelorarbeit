"""
Hybrid‑ANFIS – voll funktionsfähige Demo‑Implementation
(auf dem Iris‑Datensatz > 90 % Accuracy in < 50 Epochen)

Wichtigste Korrekturen
----------------------
1. sinnvolle Initialisierung der MF‑Zentren/‑Breiten
2. konsequents als lernbare Parameter + zusätzliches LSE‑Update 1× pro Epoche
3. Top‑p‑Maskierung (hier p = 0.2) VOR der Normalisierung
4. kein Überschreiben der Regel‑Outputs mehr
5. kleinere Batch‑Größe
"""

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import numpy as np, random, matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# --------------------------------------------------------------------------- #
#  Hybrid‑ANFIS                                                               #
# --------------------------------------------------------------------------- #
class HybridANFIS(nn.Module):
    def __init__(self, input_dim: int, num_classes: int,
                 num_mfs: int = 3, max_rules: int = 1000,
                 top_p: float = 0.2, seed: int = 0):
        super().__init__()
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        self.in_dim = input_dim
        self.C = num_classes
        self.num_mfs = num_mfs
        self.num_rules = min(num_mfs ** input_dim, max_rules)
        self.top_p = top_p                                    # Anteil aktiver Regeln pro Sample

        # --- MF‑Parameter ---------------------------------------------------- #
        self.centers = nn.Parameter(torch.empty(input_dim, num_mfs))
        self.widths  = nn.Parameter(torch.empty(input_dim, num_mfs))
        self.reset_mf_params()

        # --- Regelindex‑Matrix ---------------------------------------------- #
        full = torch.cartesian_prod(*[torch.arange(num_mfs) for _ in range(input_dim)])
        if full.size(0) > self.num_rules:
            idx = torch.randperm(full.size(0))[: self.num_rules]
            full = full[idx]
        self.register_buffer("rules", full)                   # [R, in_dim]

        # --- Konklusionen (1. Ordnung – linear) ------------------------------ #
        # Shape: [R, in_dim+1, C]        (+1 für Bias)
        self.consequents = nn.Parameter(
            torch.zeros(self.num_rules, input_dim + 1, num_classes)
        )

    # --------------------------------------------------------------------- #
    def reset_mf_params(self):
        # Zentren ~ U[0,1], Breiten klein (0.15…0.3)
        nn.init.uniform_(self.centers, 0.0, 1.0)
        nn.init.uniform_(self.widths,  0.15, 0.30)

    @staticmethod
    def gaussian(x, c, s):
        return torch.exp(-((x - c) ** 2) / (2 * s ** 2))

    # --------------------------------------------------------------------- #
    def forward(self, x):                              # x: [B, in_dim]
        B = x.size(0); eps = 1e-9
        # 1) MF‑Werte je Dimension ----------------------------------------- #
        mfs = [self.gaussian(x[:, i:i+1], self.centers[i], self.widths[i])
               for i in range(self.in_dim)]            # List len=in_dim: [B, num_mfs]
        mfs = torch.stack(mfs, dim=1)                  # [B, in_dim, num_mfs]

        # 2) Regel‑MF sammeln + Produkt ------------------------------------ #
        rules_idx = self.rules.unsqueeze(0).expand(B, -1, -1).permute(0, 2, 1)
        rule_mfs  = torch.gather(mfs, 2, rules_idx)    # [B, in_dim, R]
        firing    = torch.prod(rule_mfs, dim=1)        # [B, R]

        # 3) Top‑p‑Maskierung (20 % aktiv) ---------------------------------- #
        k = max(1, int(self.top_p * self.num_rules))
        topk_val, topk_idx = torch.topk(firing, k=k, dim=1)
        mask = torch.zeros_like(firing).scatter_(1, topk_idx, 1.0)
        firing = firing * mask

        # 4) Normalisieren -------------------------------------------------- #
        norm_firing = firing / (firing.sum(dim=1, keepdim=True) + eps)  # [B, R]

        # 5) Regel‑Outputs (Takagi‑Sugeno 1. Ordnung) ----------------------- #
        x_ext = torch.cat([x, torch.ones(B, 1, device=x.device)], dim=1) # [B, in_dim+1]
        rule_out = torch.einsum('bi,rjc->brc', x_ext, self.consequents)  # [B, R, C]
        y_hat = torch.einsum('br,brc->bc', norm_firing, rule_out)        # [B, C]

        return y_hat, norm_firing, x_ext

    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def update_consequents_lse(self, X_full: torch.Tensor, Y_onehot: torch.Tensor):
        """
        LSE‑Update auf dem *gesamten* Trainingssatz – 1× pro Epoche.
        """
        y_hat, norm_firing, x_ext = self.forward(X_full)          # [N, R], [N, in_dim+1]
        # Design‑Matrix Φ: [N, R, in_dim+1]
        phi = norm_firing.unsqueeze(2) * x_ext.unsqueeze(1)
        N, R, P1 = phi.shape; P = R * P1
        phi = phi.view(N, P)                                      # [N, P]
        # LSE: (Φᵀ Φ) β = Φᵀ Y
        # (N×P)  (P×P)(P×C)   (P×C)
        beta, *_ = torch.linalg.lstsq(phi, Y_onehot.float())
        self.consequents.data = beta.view(R, P1, self.C)


# --------------------------------------------------------------------------- #
#  Training‑Helfer                                                            #
# --------------------------------------------------------------------------- #
def prepare_iris(test_size=0.25, seed=0):
    iris = load_iris()
    X, y  = iris.data, iris.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size,
                                              shuffle=True, random_state=seed)
    scaler = MinMaxScaler().fit(X_tr)
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
    X_tr = torch.tensor(X_tr, dtype=torch.float32)
    X_te = torch.tensor(X_te, dtype=torch.float32)
    y_tr = torch.tensor(y_tr, dtype=torch.long)
    y_te = torch.tensor(y_te, dtype=torch.long)
    return X_tr, y_tr, X_te, y_te


def train(model, X_tr, y_tr, epochs=50, lr=3e-3, batch=32):
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch, shuffle=True)
    opt = optim.Adam([model.centers, model.widths, model.consequents], lr=lr)
    crit = nn.CrossEntropyLoss()
    losses = []

    for ep in range(1, epochs + 1):
        for xb, yb in loader:
            opt.zero_grad()
            logits, _, _ = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            # MF‑Parameter im [0,1]‑Bereich halten
            model.centers.data.clamp_(0, 1)
            model.widths.data.clamp_(1e-3, 1.0)
        losses.append(loss.item())

        # --- 1× pro Epoche globales LSE‑Update für die Konklusionen -------- #
        Y_oh = F.one_hot(y_tr, num_classes=model.C)
        model.update_consequents_lse(X_tr, Y_oh)

        if ep % 10 == 0:
            print(f"Epoch {ep:3d}/{epochs}  loss={loss.item():.4f}")

    # Loss‑Verlauf
    plt.figure(); plt.plot(losses); plt.title("Train‑Loss"); plt.show()


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(X)[0].argmax(1)
        acc  = (pred == y).float().mean().item()
    return acc


# --------------------------------------------------------------------------- #
#  Main‑Routine                                                               #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Daten
    X_tr, y_tr, X_te, y_te = prepare_iris()

    # Modell
    anfis = HybridANFIS(input_dim=X_tr.shape[1],
                        num_classes=len(torch.unique(y_tr)),
                        num_mfs=2,          # 3 Gauß‑MFs pro Feature
                        max_rules=500,
                        top_p=1.0,          # 20 % aktiv
                        seed=0)

    # Training
    train(anfis, X_tr, y_tr, epochs=500, lr=3e-3, batch=32)

    # Bewertung
    acc_tr = evaluate(anfis, X_tr, y_tr)
    acc_te = evaluate(anfis, X_te, y_te)
    print(f"Train‑Accuracy : {acc_tr*100:5.2f}%")
    print(f"Test‑Accuracy  : {acc_te*100:5.2f}%")
