# -------------------------------------------
# popfnn.py  (sauber & kurz)
# -------------------------------------------
import torch, torch.nn as nn

class POPFNN(nn.Module):
    def __init__(self, d, C, M=3, max_rules=1000):
        super().__init__()
        self.d, self.C, self.M = d, C, M
        full = torch.cartesian_prod(*[torch.arange(M) for _ in range(d)])
        if full.size(0) > max_rules:
            full = full[torch.randperm(full.size(0))[:max_rules]]
        self.register_buffer("rules", full)     # [R,d]
        self.R = full.size(0)

        # Antezedenz-MF-Parameter
        self.centers = nn.Parameter(torch.rand(d, M))
        self.widths  = nn.Parameter(torch.ones(d, M)*.3)

        # Konsequenz-MF-Center   (C Klassen × M Zentren)
        self.label_cent = nn.Parameter(torch.linspace(0., 1., M)
                                       .repeat(C))          # [C*M]
        self.W = nn.Parameter(torch.zeros(self.R, C*M))     # POP-Gewicht

    # ---------- Vorwärts ----------
    def _fire(self, x):                          # x [B,d]
        μ = torch.exp(-0.5*((x.unsqueeze(2)-self.centers)/self.widths.abs())**2)
        idx = self.rules               # [R,d]
        μ_sel = μ[..., idx]            # gather über fancy-Index (PyTorch 2!)
        return μ_sel.prod(dim=2)       # [B,R]

    def forward(self, x):                          # → logits [B,C]
        fire = self._fire(x)                       # [B,R]
        logits = []
        for c in range(self.C):
            Wc = self.W[:, c*self.M:(c+1)*self.M]          # [R,M]
            cc = self.label_cent[c*self.M:(c+1)*self.M]    # [M]
            num = (fire.unsqueeze(2)*Wc*cc).sum(dim=(1,2))
            den = (fire.unsqueeze(2)*Wc).sum(dim=(1,2))+1e-9
            logits.append(num/den)
        return torch.stack(logits, dim=1)          # [B,C]

    # ---------- POP-Init ----------
    @torch.no_grad()
    def pop_init(self, X, y):
        y_onehot = torch.eye(self.C, device=X.device)[y]
        for xi, yi in zip(X, y_onehot):
            fire = self._fire(xi.unsqueeze(0)).squeeze(0)  # [R]
            fire = fire.flatten()                          # stelle sicher 1-D
            
            # Debug
            print("fire:", fire.shape, "C,M:", self.C, self.M)

            # Sichere lab_vec-Erzeugung:
            yi_1d   = yi.flatten()                         # [C]
            lab_vec = yi_1d.unsqueeze(1).repeat(1, self.M).flatten()  # [C*M]
            print("lab_vec:", lab_vec.shape)

            # Äußeres Produkt und Accumulation
            self.W += torch.outer(fire, lab_vec)

