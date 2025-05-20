# ssl_pop_pipeline.py
# ------------------------------------------------------------
import torch, torch.nn.functional as F
from data_utils import load_K_chess_data_splitted, load_iris_data, load_Kp_chess_data_ord, load_abalon_data
from PopFnn     import POPFNN          # deine schnellere Version
from pop_ssl_utils     import semi_split, rf_teacher_pseudo, rule_class_mapping
from sklearn.metrics   import accuracy_score
import numpy as np, random, os

SEED = 42
torch.manual_seed(SEED);  np.random.seed(SEED);  random.seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_pop_ssl(X_L, y_L, X_P, y_P, w_P,
                  d, C, num_mfs=4, epochs=50, lr=5e-4, seed=SEED):

    X_all = torch.cat([X_L, X_P]).to(device)
    y_all = torch.cat([y_L, y_P]).to(device)
    w_all = torch.cat([torch.ones(len(y_L)), w_P]).to(device)

    model = POPFNN(d, C, num_mfs=num_mfs).to(device)
    model.pop_init(X_L.to(device), y_L.to(device))      # Regeln + Init-Gewichte

    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss(reduction="none")

    for _ in range(epochs):
        opt.zero_grad()
        logits = model(X_all)
        (loss(logits, y_all) * w_all).mean().backward()
        opt.step()

    return model


def run(label_fracs=(0.1,0.2,0.3), thr=0.3):
    #X_tr, y_tr, X_te, y_te = load_K_chess_data_splitted()
    X_tr, y_tr, X_te, y_te = load_iris_data()
    d, C = X_tr.shape[1], int(y_tr.max()+1)

    for p in label_fracs:
        idx_L, idx_U = semi_split(y_tr, p, seed=SEED)
        X_P, y_P, w_P, rf = rf_teacher_pseudo(
            X_tr.numpy(), y_tr.numpy(),
            idx_L.numpy(), idx_U.numpy(), thr=thr, seed=SEED)

        # Torch-Tensors
        X_L, y_L = X_tr[idx_L], y_tr[idx_L]
        X_P = torch.from_numpy(X_P).float()
        y_P = torch.from_numpy(y_P).long()
        w_P = torch.from_numpy(w_P).float()

        pop = train_pop_ssl(X_L, y_L, X_P, y_P, w_P, d, C)

        # --- Test-Accuracy
        pop.eval();  rf_acc = rf.score(X_te.numpy(), y_te.numpy())
        with torch.no_grad():
            pred = pop(X_te.to(device)).argmax(1).cpu()
        pop_acc = accuracy_score(y_te, pred)

        print(f"{int(p*100):>2}% labels | RF-sup: {rf_acc*100:5.2f}% "
              f"| POPFNN SSL: {pop_acc*100:5.2f}%")


        rule_class_mapping(pop, top_k=3)


if __name__ == "__main__":
    run()
