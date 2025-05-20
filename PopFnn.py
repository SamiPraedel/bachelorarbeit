import torch, torch.nn as nn, torch.nn.functional as F

class POPFNN(nn.Module):
    def __init__(self, d, C, num_mfs=7):

        super().__init__()
        self.d, self.C, self.M = d, C, num_mfs

        self.centers = nn.Parameter(torch.rand(d, self.M)) 
        self.widths  = nn.Parameter(torch.ones(d, self.M) * 0.3)     


        self.label_cent = nn.Parameter(
            torch.linspace(0., 1., self.M).repeat(C))                # [C*M]


        self.register_buffer("rules", torch.empty(0, d, dtype=torch.long))
        self.R = 0
        self.W = nn.Parameter(torch.empty(0, C * self.M))            # [R, C*M]


    def _fuzzify(self, x):                  # x [B, d]
        μ = torch.exp(-0.5 * ((x.unsqueeze(2) - self.centers) /
                              self.widths.abs()) ** 2)               # [B,d,M]
        return μ


    def _fire(self, x):                     # x [B,d]
        μ = self._fuzzify(x)                                    # [B,d,M]
        idx = self.rules.t().unsqueeze(0)                       # [1,d,R]
        μ_sel = torch.gather(μ, 2, idx.expand(μ.size(0), -1, -1))  # [B,d,R]
        return μ_sel.prod(dim=1)                                # [B,R]


    def forward(self, x):               
        fire = self._fire(x)            
        W_eff = (self.W * self.label_cent)      
        W_eff = W_eff.view(self.R, self.C, self.M).sum(dim=2)  
        return fire @ W_eff                         

    @torch.no_grad()
    def pop_init(self, X, y):
        """
        1. wähle pro Sample den MF-Index mit größter Zugehörigkeit (argmax)
        2. bilde die Menge aller tatsächlich auftretenden Index-Kombis (= Regeln)
        3. initialisiere self.rules, self.W  (regelgetriebene POP-Gewichte)
        4. zähle (outer-product) die initialen Klassengewichte
        """
        # 1) MF-Index pro Dimension
        idx_max = self._fuzzify(X).argmax(dim=2)                 # [N,d]

        # 2) eindeutige Kombis → Regelsatz
        unique_rules = torch.unique(idx_max, dim=0)              # [R,d]
        self.rules = unique_rules
        self.R = unique_rules.size(0)

        # 3) Gewichte neu anlegen (auf gleichem Gerät)
        device = X.device
        self.W = nn.Parameter(torch.zeros(self.R, self.C * self.M,
                                          device=device))

        # 4) POP-Gewichte zählen (vektorisiert)
        fire = self._fire(X)                                     # [N,R]
        lab_mat = F.one_hot(y, self.C).float()                   # [N,C]
        lab_mat = lab_mat.repeat_interleave(self.M, dim=1)       # [N,C*M]
        self.W.add_(fire.T @ lab_mat)                            # [R,C*M]
        row_sums = self.W.sum(dim=1, keepdim=True).clamp(min=1e-6)
        self.W.div_(row_sums)



