import torch, torch.nn as nn, torch.nn.functional as F

class POPFNN(nn.Module):
    def __init__(self, d, C, num_mfs=7, output_discretization_steps=100):

        super().__init__()
        self.d, self.C, self.M = d, C, num_mfs

        self.centers = nn.Parameter(torch.rand(d, self.M)) 
        self.widths  = nn.Parameter(torch.ones(d, self.M) * 0.3)     


        self.label_cent = nn.Parameter(
            torch.linspace(0., 1., self.M).repeat(C))                # [C*M]
        
        self.inference_method = 'mamdani'  # 'tsk' or 'mamdani'

        self.register_buffer("rules", torch.empty(0, d, dtype=torch.long))
        self.R = 0
        self.W = nn.Parameter(torch.empty(0, C * self.M))            # [R, C*M]
        
        if self.inference_method == 'tsk':
            # TSK consequents are crisp numbers (learnable constants)
            self.label_cent = nn.Parameter(torch.linspace(0., 1., self.M).repeat(C)) # [C*M]
        else:
            # Mamdani consequents are fuzzy sets defined by centers and widths
            self.output_centers = nn.Parameter(torch.linspace(0., 1., self.M).repeat(C, 1)) # Shape: [C, M]
            self.output_widths  = nn.Parameter(torch.ones(C, self.M) * 0.2)
            # Create a discretized axis for the output space for defuzzification
            self.register_buffer("output_axis", torch.linspace(0., 1., output_discretization_steps))


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
    
    def forward_s(self, fire):

            # fire shape: [B, R]

            # --- EVALUATION PATH (Fastest, Low-Memory, Non-Differentiable) ---
            if not self.training:
                # This path is unchanged. It's already fast and memory-efficient for evaluation.
                w_view = self.W.view(self.R, self.C, self.M)
                rule_consequents_idx = w_view.argmax(dim=2)
                class_idx = torch.arange(self.C, device=fire.device).unsqueeze(0).T
                consequent_centers = self.output_centers[class_idx, rule_consequents_idx.T]
                consequent_widths = self.output_widths.abs()[class_idx, rule_consequents_idx.T]
                consequent_centers = consequent_centers.T.view(1, self.R, self.C, 1)
                consequent_widths = consequent_widths.view(1, self.R, self.C, 1)
                y_axis = self.output_axis.view(1, 1, 1, -1)
                all_consequent_fs = torch.exp(-0.5 * ((y_axis - consequent_centers) / consequent_widths) ** 2)
                rule_outputs = torch.min(fire.view(-1, self.R, 1, 1), all_consequent_fs)
                agg_fs = torch.max(rule_outputs, dim=1).values
                denominator = agg_fs.sum(dim=2).clamp(min=1e-6)
                numerator = (self.output_axis.view(1, 1, -1) * agg_fs).sum(dim=2)
                return numerator / denominator

            # --- TRAINING PATH (Balanced Speed and Memory) ---
            else:
                batch_size = fire.shape[0]
                outputs = torch.zeros(batch_size, self.C, device=fire.device)
                w_view = self.W.view(self.R, self.C, self.M)

                # Pre-calculate all possible output fuzzy set shapes: [C, M, D]
                y_axis = self.output_axis.view(1, 1, -1)
                centers = self.output_centers.unsqueeze(-1)
                widths = self.output_widths.abs().unsqueeze(-1)
                all_output_fs = torch.exp(-0.5 * ((y_axis - centers) / widths) ** 2)

                # Loop over each item in the batch to keep memory usage low
                for b in range(batch_size):
                    # fire_b shape: [R]
                    fire_b = fire[b]

                    # Vectorized operations for a SINGLE batch item
                    # Reshape for broadcasting
                    fire_exp = fire_b.view(self.R, 1, 1, 1)        # [R, 1, 1, 1]
                    w_exp = w_view.view(self.R, self.C, self.M, 1) # [R, C, M, 1]
                    fs_exp = all_output_fs.view(1, self.C, self.M, -1) # [1, C, M, D]

                    # Weighted implication (soft, differentiable)
                    # Broadcasting: [R,1,1,1] * [R,C,M,1] -> [R,C,M,1]
                    weighted_fire = fire_exp * w_exp
                    # Broadcasting: min([R,C,M,1], [1,C,M,D]) -> [R,C,M,D]
                    rule_outputs = torch.min(weighted_fire, fs_exp)

                    # Aggregate over rules (R) and membership functions (M)
                    # Input: [R,C,M,D], Output: [C,D]
                    agg_fs = torch.max(rule_outputs, dim=(0, 2)).values

                    # Defuzzify the final aggregated set for this class
                    denominator = agg_fs.sum(dim=1).clamp(min=1e-6)     # [C]
                    numerator = (self.output_axis.view(1, -1) * agg_fs).sum(dim=1) # [C]
                    outputs[b] = numerator / denominator

                return outputs

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
