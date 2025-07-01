import copy, math, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# -------------------------------------------------------------
# 1) Hilfsfunktionen
# -------------------------------------------------------------
def train_weighted_anfis(model, X, y, weights, epochs, lr, device):
    model.to(device)
    X, y, w = X.to(device), y.to(device), weights.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train(); opt.zero_grad()
        logits, _, _ = model(X)                       #  <<<
        loss_vec = F.cross_entropy(logits, y, reduction='none')
        loss     = (w * loss_vec).sum() / w.sum()     # normiert
        loss.backward(); opt.step()
    return model


def _weighted_err(preds, y_true, weights):
    """weighted 0/1-error (tensors on same device)"""
    miss = (preds != y_true).float()
    return (weights * miss).sum() / weights.sum(), miss


def semi_supervised_boost_anfis(
        ModelClass, model_args,
        X_l, y_l, X_u,
        rounds=10,           # M
        epochs=20, lr=1e-3,
        pseudo_th=0.80,
        init_unl_w=1e-2,
        device='cuda' if torch.cuda.is_available() else 'cpu'):

    device = torch.device(device)
    X_l, y_l, X_u = X_l.to(device), y_l.to(device), X_u.to(device)

    N_l, N_u = len(X_l), len(X_u)
    w_l = torch.full((N_l,), 1. / N_l, device=device)
    w_u = torch.full((N_u,), init_unl_w, device=device)

    y_u = torch.full((N_u,), -1, dtype=torch.long, device=device)  # dummy
    X_all = torch.cat([X_l, X_u])
    y_all = torch.cat([y_l, y_u])

    ensemble, alphas = [], []

    for m in range(rounds):
        # ---- 1. Weak learner
        model = ModelClass(**model_args).to(device)
        weights_all = torch.cat([w_l, w_u])
        model = train_weighted_anfis(model, X_all, y_all, weights_all,
                                     epochs, lr, device)

        # ---- 2. Vorhersagen & Konfidenzen
        model.eval()
        with torch.no_grad():
            logits, _, _ = model(X_all)
            probs  = F.softmax(logits, dim=1)
            conf   = probs.max(1).values          # [N_all]
            preds  = probs.argmax(1)

        # ---- 3. Pseudo-Labels hinzufügen
        unl_mask   = (y_all == -1)                # bool-Tensor
        accept     = (unl_mask & (conf >= pseudo_th))
        n_accept   = int(accept.sum())

        if n_accept:
            y_all[accept] = preds[accept]
            # unlabeled Gewicht proportional zur Konfidenz
            w_u = conf[N_l:].clone()
            w_u[~accept[N_l:]] = init_unl_w

        # ---- 4. Fehler auf Labeled
        err, miss_vec = _weighted_err(preds[:N_l], y_l, w_l)
        C = int(y_l.max()) + 1
        if err <= 0 or err >= 1 - 1 / C:
            print(f"Round {m}: degenerate error={err:.4f} ➜ stop")
            break

        alpha = torch.log((1 - err) / err) + math.log(C - 1)
        miss_all = (preds != y_all).float()
        weights_all = torch.cat([w_l, w_u])
        weights_all = weights_all * torch.exp(alpha * miss_all)
        weights_all /= weights_all.sum()

        w_l, w_u = weights_all[:N_l], weights_all[N_l:]

        ensemble.append(copy.deepcopy(model))
        alphas.append(alpha.item())

        print(f"[Round {m+1}] err={err:.3f} | α={alpha:.2f} | "
              f"pseudo added={n_accept:5d} | pool left={int(unl_mask.sum())-n_accept}")

        # Früher Ausstieg, wenn nichts Neues mehr kommt
        if n_accept == 0: break

    return ensemble, alphas


# -------------------------------------------------------------
# 3) Inference – gewichtete Mehrheitsentscheidung
# -------------------------------------------------------------
@torch.no_grad()
def predict_boost_anfis(ensemble, alphas, X, device=None):
    if device is None:
        device = next(ensemble[0].parameters()).device
    X = X.to(device)
    C = ensemble[0].num_classes
    votes = torch.zeros(len(X), C, device=device)

    for wk, alpha in zip(ensemble, alphas):
        wk.eval()
        logits, _, _ = wk(X)
        preds = logits.argmax(1)
        votes.scatter_add_(1, preds.unsqueeze(1), alpha*torch.ones_like(preds, dtype=votes.dtype).unsqueeze(1))

    return votes.argmax(1)


# -------------------------------------------------------------
# 4) Minimal Test (Dummy Data)
# -------------------------------------------------------------
if __name__ == "__main__":
    from anfis_nonHyb import NoHybridANFIS
    from data_utils    import load_K_chess_data_splitted

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X_tr, y_tr, X_te, y_te = load_K_chess_data_splitted()
    # simulate label scarcity
    frac = .1
    idx = torch.randperm(len(y_tr))             # shuffle indices
    n_l = int(len(y_tr)*frac)
    X_l, y_l = X_tr[idx[:n_l]], y_tr[idx[:n_l]]
    X_u       = X_tr[idx[n_l:]]

    model_args = dict(input_dim=X_tr.shape[1],
                      num_classes=int(y_tr.max())+1,
                      num_mfs=4,
                      max_rules=1000,
                      seed=42,
                      zeroG=True)

    ens, alphas = semi_supervised_boost_anfis(
        NoHybridANFIS, model_args,
        X_l, y_l, X_u,
        rounds=8, epochs=15, lr=5e-3,
        pseudo_th=.85, device=device)

    y_hat = predict_boost_anfis(ens, alphas, X_te, device)
    acc   = (y_hat.cpu() == y_te).float().mean().item()*100
    print(f"\nTest-Accuracy (ensemble): {acc:.2f}%")
