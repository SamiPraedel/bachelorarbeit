# ssl_anfis_pipeline.py
# ----------------------------------------------------------
import numpy as np, torch, random, warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics  import accuracy_score
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from data_utils import load_K_chess_data_splitted, load_iris_data, load_Kp_chess_data_ord, 
from anfis_hybrid     import HybridANFIS                 # deine Klasse

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED);  random.seed(SEED);  torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
def make_semi_split(y_tr, frac, seed=0):
    idx_lab, idx_unlab = train_test_split(
        np.arange(len(y_tr)), train_size=frac,
        stratify=y_tr, random_state=seed)
    return torch.tensor(idx_lab), torch.tensor(idx_unlab)

def teacher_pseudo_labels(X_train, y_train, idx_lab, idx_unlab,
                          thr=0.90, seed=SEED):
    rf = RandomForestClassifier(n_estimators=400, max_depth=None,
                                n_jobs=-1, random_state=seed)
    rf.fit(X_train[idx_lab], y_train[idx_lab])

    proba  = rf.predict_proba(X_train[idx_unlab])
    conf   = proba.max(1)
    mask   = conf >= thr

    X_pseudo = X_train[idx_unlab][mask]
    y_pseudo = proba.argmax(1)[mask]
    w_pseudo = conf[mask]                      # Gewichte = Vertrauen

    return X_pseudo, y_pseudo, w_pseudo, rf

# ---------------------------------------------------------------------------
def train_anfis_ssl(X_l, y_l, X_p, y_p, w_p,
                    input_dim, num_classes,
                    num_mfs=4, max_rules=1000,
                    epochs=50, lr=5e-3, seed=SEED,
                    device="cuda" if torch.cuda.is_available() else "cpu"):
    


    X_all = torch.cat([X_l, X_p]).to(device)          # [N, d]
    y_all = torch.cat([y_l, y_p]).to(device)          # [N]
    w_all = torch.cat([torch.ones(len(y_l)), w_p]).to(device)

    y_onehot = F.one_hot(y_all, num_classes).float()

    model = HybridANFIS(input_dim, num_classes,
                        num_mfs, max_rules, seed).to(device)

    opt   = torch.optim.Adam([
                {'params': model.centers, 'lr': lr},
                {'params': model.widths , 'lr': lr},
            ])

    for epoch in range(epochs):
        model.train();   opt.zero_grad()
        logits, norm_fs, x_ext = model(X_all)

        # gewichtete CE-Loss
        loss = (w_all *
                F.cross_entropy(logits, y_all, reduction='none')).mean()
        loss.backward();  opt.step()

        # Consequents als closed-form LS anpassen
        model.update_consequents(norm_fs.detach(), x_ext.detach(), y_onehot)

    return model
# ---------------------------------------------------------------------------

def run_pipeline(label_fracs=(0.1, 0.2, 0.3), thr=0.90):
    #X_tr, y_tr, X_te, y_te = load_K_chess_data_splitted()
    #X_tr, y_tr, X_te, y_te = load_iris_data()
    X_tr, y_tr, X_te, y_te = load_Kp_chess_data_ord()
    input_dim   = X_tr.shape[1]
    num_classes = int(y_tr.max().item() + 1)

    for frac in label_fracs:
        idx_l, idx_u = make_semi_split(y_tr, frac, seed=SEED)
        X_p, y_p, w_p, rf_teacher = teacher_pseudo_labels(
            X_tr.numpy(), y_tr.numpy(), idx_l.numpy(), idx_u.numpy(),
            thr=thr)

        # Torch-Tensors
        X_l, y_l = X_tr[idx_l], y_tr[idx_l]
        X_p = torch.from_numpy(X_p).float()
        y_p = torch.from_numpy(y_p).long()
        w_p = torch.from_numpy(w_p).float()

        model = train_anfis_ssl(X_l, y_l, X_p, y_p, w_p,
                                input_dim, num_classes,
                                num_mfs=4, max_rules=1000)

        # Evaluation
        model.eval()
        with torch.no_grad():
            y_pred = model(X_te.to(model.centers.device))[0].argmax(1).cpu()
        acc_rf  = accuracy_score(y_te, rf_teacher.predict(X_te.numpy()))
        acc_ssl = accuracy_score(y_te, y_pred)
        print(f"{int(frac*100):>2}% labels | RF-sup: {acc_rf*100:5.2f}%"
              f" | ANFIS SSL: {acc_ssl*100:5.2f}%")


if __name__ == "__main__":
    run_pipeline()
