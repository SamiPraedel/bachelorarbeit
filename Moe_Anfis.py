import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils import load_iris_data, load_heart_data, load_wine_data, load_abalon_data, load_K_chess_data, load_Kp_chess_data, load_K_chess_data_splitted, load_K_chess_data_OneHot, load_Poker_data
import pandas as pd

class ExpertANFIS(nn.Module):
    """
    A minimal ANFIS-like classifier for demonstration.
    You can adapt your full 'HybridANFIS' code here.
    """
    def __init__(self, input_dim, num_classes, num_mfs, max_rules):
        super(ExpertANFIS, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_mfs = num_mfs
        self.num_rules = num_mfs ** input_dim
        if self.num_rules > max_rules:
            self.num_rules = max_rules

        # Example membership params
        self.centers = nn.Parameter(torch.ones(input_dim, num_mfs))
        self.widths  = nn.Parameter(torch.ones(input_dim, num_mfs))

        # Example random subset of rules if needed
        self.rules = torch.cartesian_prod(*[torch.arange(num_mfs) for _ in range(input_dim)])
        if self.rules.shape[0] > max_rules:
            self.rules = self.rules[torch.randperm(self.rules.shape[0])[:max_rules]]
        

        self.consequents = nn.Parameter(torch.randn(self.rules.shape[0], num_classes))

    def gaussian_mf(self, x, center, width):
        # x: shape [batch_size, 1]
        return torch.exp(-((x - center)**2)/(2*width**2))

    def forward(self, x):
        """
        x: shape [batch_size, input_dim]
        returns: [batch_size, num_classes]
        """
        eps = 1e-9
        batch_size = x.shape[0]
        num_rules = self.rules.shape[0]

        # 1) membership
        # mfs[i]-> membership in dimension i for num_mfs
        # shape => [batch_size, input_dim, num_mfs]
        membership_list = []
        for i in range(self.input_dim):
            x_i = x[:, i].unsqueeze(1)  # [B,1]
            mf_i = self.gaussian_mf(x_i, self.centers[i], self.widths[i])
            membership_list.append(mf_i) 
        # stack => [batch_size, input_dim, num_mfs]
        mfs = torch.stack(membership_list, dim=1)

        # 2) rule indexing => gather
        # rules_idx => [num_rules, input_dim]
        # expand => [B, input_dim, num_rules], gather => pick membership at each dimension
        rules_idx = self.rules.unsqueeze(0).expand(batch_size, -1, -1).permute(0,2,1)
        # gather => [B, input_dim, num_rules]
        gathered = torch.gather(mfs, dim=2, index=rules_idx)
        # product across dim=1 => [B, num_rules]
        firing_strengths = torch.prod(gathered, dim=1)

        # 3) normalize
        norm_fs = firing_strengths / (firing_strengths.sum(dim=1, keepdim=True) + eps)

        # 4) zero-order T-S => shape of self.consequents => [num_rules, num_classes]
        # Weighted sum => [B, num_classes]
        outputs = torch.einsum('br,rc->bc', norm_fs, self.consequents)

        return outputs


class MixtureOfExperts(nn.Module):
    def __init__(self, 
                 input_dim, 
                 num_classes, 
                 K=3, 
                 num_mfs=2, 
                 max_rules=16, 
                 hidden_gating=16):
        """
        K: number of ANFIS experts
        hidden_gating: hidden dimension in gating MLP
        """
        super().__init__()
        self.K = K
        # Create K ExpertANFIS submodels
        self.experts = nn.ModuleList([
            ExpertANFIS(input_dim, num_classes, num_mfs, max_rules)
            for _ in range(K)
        ])
        # Gating network: input_dim -> hidden -> K (softmax)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_gating),
            nn.ReLU(),
            nn.Linear(hidden_gating, K),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        """
        x => [batch_size, input_dim]
        returns => [batch_size, num_classes]
        """
        batch_size = x.size(0)
        # gating => [B, K]
        gate_weights = self.gate(x)

        # each expert => [B, num_classes]
        expert_outputs = []
        for k in range(self.K):
            y_k = self.experts[k](x)
            expert_outputs.append(y_k)
        # shape => [B, K, num_classes]
        expert_stack = torch.stack(expert_outputs, dim=1)

        # Weighted sum => [B, num_classes]
        # y_final[b,c] = sum_k (gate_weights[b,k] * expert_stack[b,k,c])
        y_final = torch.einsum('bk,bkc->bc', gate_weights, expert_stack)
        return y_final


def train_moe(moe_model, X, Y, num_epochs=50, lr=1e-3, batch_size=32):
    """
    Minimal training loop using cross-entropy for classification.
    """
    device = torch.device("cpu")
    moe_model.to(device)
    optimizer = torch.optim.Adam(moe_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    moe_model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = moe_model(batch_x)  # [B, num_classes]
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")
    print("Training complete.")

if __name__ == "__main__":
    
    X_train, y_train, X_test, y_test = load_K_chess_data_splitted()
    # X_train2, y_train, X_test, y_test = load_K_chess_data_OneHot()
    #X_train, y_train, X_test, y_test = load_iris_data()
    #X_train, y_train, X_test, y_test = load_heart_data()
    # X_train5, y_train, X_test, y_test, s = load_abalon_data()
    # X_train6, y_train, X_test, y_test, s = load_Kp_chess_data()
    #X_train7, y_train, X_test, y_test, = load_Poker_data()


    # Build MoE with 3 experts, each a small ANFIS
    moe_model = MixtureOfExperts(input_dim=6, num_classes=y_train.unique().size(dim=0), K=3, num_mfs=3, max_rules=1000)

    train_moe(moe_model, X_train, y_train, num_epochs=50, lr=1e-2)

    # Evaluate
    moe_model.eval()
    with torch.no_grad():
        preds = torch.argmax(moe_model(X_test), dim=1)
        acc = (preds==y_test).float().mean().item()
    print(f"Final accuracy = {acc*100:.1f}%")
