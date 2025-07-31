# ──────────────────────────────────────────────────────────────
#  run_experiments.py
#  A clean orchestration script for all SSL variants in lb_scratch
# ──────────────────────────────────────────────────────────────
import argparse, csv, time, pathlib, random, inspect, sys, json
from typing import Callable, Dict, List

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------- import your code -----------
from lb_scratch import (
    ExperimentRunner,                      # the high-level helper class
    NoHybridANFIS, HybridANFIS, POPFNN,    # models you want as baselines
    load_K_chess_data_splitted,            # add more loaders as you wish
)

# ╭──────────────────────────────────────────────────────────╮
# │ 1.  GLOBAL CONFIG – edit once, not down in the code     │
# ╰──────────────────────────────────────────────────────────╯
SEED               = 42
LABEL_FRACTIONS    = [0.1]          # 10 %, 20 %, 50 %
K_LIST        = [7, 15, 25]          # k for k‑NN graph / k‑NN cls
SIGMA_LIST    = [0.10, 0.30, 0.50]   # RBF bandwidths
BETA_LIST     = [0.3, 0.5, 0.7]      # View‑mix weight for MV‑methods
CSV_OUT            = "ssl_results.csv"
import itertools
PARAM_COMBOS = list(itertools.product(K_LIST, SIGMA_LIST, BETA_LIST))

DATASETS: Dict[str, Callable] = {
    "KChess": load_K_chess_data_splitted,
    # "HTRU" : load_htru_data,   # add your other loaders here
}

BASELINE_MODELS = {
    "ANFIS-NoHyb": lambda d, C: NoHybridANFIS(
        input_dim=d, num_classes=C, num_mfs=4, max_rules=1000, seed=SEED
    ),
    # Add other baselines if wanted
}

# ╭──────────────────────────────────────────────────────────╮
# │ 2.  Helper: deterministic seeds                         │
# ╰──────────────────────────────────────────────────────────╯
def set_seed(seed: int = SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ╭──────────────────────────────────────────────────────────╮
# │ 3.  Core experiment routine                              │
# ╰──────────────────────────────────────────────────────────╯
def run_single_dataset(name: str,
                       load_fn: Callable,
                       device: torch.device):

    print(f"\n{'='*26}  DATASET: {name}  {'='*26}")
    X_tr, y_tr, X_te, y_te = load_fn()        # tensors
    C = len(torch.unique(y_tr))
    d = X_tr.shape[1]
    X_te_np, y_te_np = X_te.numpy(), y_te.numpy()

    for frac in LABEL_FRACTIONS:
        # --- split into labelled / unlabelled ----------
        idx_l, idx_u = train_test_split(
            np.arange(len(y_tr)), train_size=frac,
            stratify=y_tr.numpy(), random_state=SEED
        )
        X_l,  y_l  = X_tr[idx_l], y_tr[idx_l]
        y_semi_sup  = np.full(len(y_tr), fill_value=-1, dtype=np.int64)
        y_semi_sup[idx_l] = y_l.numpy()

        print(f"\n  --- label fraction: {frac*100:.0f}% "
              f"(L={len(idx_l)}, U={len(idx_u)}) ---")

        # ---------- Supervised baseline ----------------
        model_bl = BASELINE_MODELS["ANFIS-NoHyb"](d, C).to(device)
        # quick warm-up (could call your own training util here)
        opt = torch.optim.Adam(model_bl.parameters(), lr=5e-3)
        for _ in range(1000):
            model_bl.train(); opt.zero_grad()
            out, *_ = model_bl(X_l.to(device))
            loss = torch.nn.functional.cross_entropy(out, y_l.to(device))
            loss.backward(); opt.step()
        with torch.no_grad():
            model_bl.eval()
            preds = model_bl(X_te.to(device))[0].argmax(1).cpu().numpy()
        acc_sup = accuracy_score(y_te_np, preds)
        print(f"    Supervised baseline acc = {acc_sup*100:.2f}%")
        log_row(name, "Supervised-NoHyb", frac, acc_sup, k=-1, sigma=-1, beta=None)

        # ---------- Feature extraction for rule / MF ----
        model_bl.eval()
        with torch.no_grad():
            _, rule_tr, _ = model_bl(X_tr.to(device))
            _, rule_te, _ = model_bl(X_te.to(device))
            # --- Row‑normalise firing vectors (probability‑like) ---
            rule_tr = rule_tr / rule_tr.sum(1, keepdim=True).clamp_min(1e-9)
            rule_te = rule_te / rule_te.sum(1, keepdim=True).clamp_min(1e-9)
            mf_tr = model_bl._fuzzify(X_tr.to(device)).reshape(len(X_tr), -1)
            mf_te = model_bl._fuzzify(X_te.to(device)).reshape(len(X_te), -1)

        runner = ExperimentRunner()
        for k_val, sigma_val, beta_val in PARAM_COMBOS:
            print(f"\n    >>> Hyper‑Params: k={k_val}, σ={sigma_val}, β={beta_val}\n")

            # --- Rule-Space GRF ----------------------------
            _, acc_rule = runner.run_rule_space_ssl(
                rule_tr, y_semi_sup, rule_te, y_tr, y_te, device,
                k=k_val, sigma=0.30
            )
            log_row(name, "Rule-GRF", frac, acc_rule, k=k_val, sigma=0.30, beta=None)

            # --- MF-Space GRF ------------------------------
            _, acc_mf = runner.run_mf_space_ssl(
                mf_tr, y_semi_sup, mf_te, y_tr, y_te, device,
                k=k_val, sigma=0.30
            )
            log_row(name, "MF-GRF", frac, acc_mf, k=k_val, sigma=0.30, beta=None)

            # --- Raw data LP baseline ----------------------
            _, acc_raw = runner.run_raw_space_ssl(
                X_tr.numpy(), y_semi_sup, X_te_np, y_tr, y_te, k=k_val
            )
            log_row(name, "Raw-LP( knn )", frac, acc_raw, k=k_val, sigma=-1, beta=None)

            # --- FMV-CLP -----------------------------------
            comb_tr = np.hstack([mf_tr.detach().cpu().numpy(), rule_tr.detach().cpu().numpy()])
            comb_te = np.hstack([mf_te.detach().cpu().numpy(), rule_te.detach().cpu().numpy()])
            acc_clp = runner.run_fmv_clp(
                mf_tr.detach().cpu().numpy(), rule_tr.detach().cpu().numpy(), y_semi_sup,
                mf_te.detach().cpu().numpy(), rule_te.detach().cpu().numpy(),
                comb_tr, comb_te, y_te, k=k_val, sigma_m=0.30, sigma_r=0.30, beta=beta_val
            )
            log_row(name, "FMV-CLP", frac, acc_clp, k=k_val, sigma=0.30, beta=beta_val)

            # --- MV-GRF ------------------------------------
            acc_mvg = runner.run_mv_grf(
                mf_tr.detach().cpu().numpy(), rule_tr.detach().cpu().numpy(), y_semi_sup,
                mf_te.detach().cpu().numpy(), rule_te.detach().cpu().numpy(),
                comb_tr, comb_te, y_te, k=k_val, sigma_m=0.30, sigma_r=0.30, beta=beta_val
            )
            log_row(name, "MV-GRF", frac, acc_mvg, k=k_val, sigma=0.30, beta=beta_val)


# ╭──────────────────────────────────────────────────────────╮
# │ 4.  CSV logging helper                                   │
# ╰──────────────────────────────────────────────────────────╯
def log_row(dataset: str, method: str, frac: float, acc: float,
            k: int, sigma: float, beta: float | None):
    header = ["dataset", "method", "labeled_frac",
              "k", "sigma", "beta", "test_acc"]
    file = pathlib.Path(CSV_OUT)
    new = not file.exists()
    with file.open("a", newline="") as f:
        writer = csv.writer(f)
        if new: writer.writerow(header)
        writer.writerow([dataset, method, f"{frac:.2f}", k, sigma, beta if beta is not None else "", f"{acc:.4f}"])


# ╭──────────────────────────────────────────────────────────╮
# │ 5.  Main entry                                           │
# ╰──────────────────────────────────────────────────────────╯
def main():
    parser = argparse.ArgumentParser(
        description="Unified experiment driver for SSL variants (lb_scratch).")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()),
                        help="Subset of datasets to run.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed()
    device = torch.device(args.device)

    for dname in args.datasets:
        if dname not in DATASETS:
            print(f"Unknown dataset '{dname}' – skip.")
            continue
        t0 = time.time()
        run_single_dataset(dname, DATASETS[dname], device)
        print(f"  ↳ finished {dname} in {(time.time()-t0)/60:.1f} min.")

    print(f"\nResults stored in →  {CSV_OUT}")


if __name__ == "__main__":
    main()