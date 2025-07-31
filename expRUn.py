import torch
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data_utils    import load_K_chess_data_splitted, load_htru_data, load_pmd_data, load_letter_data
from anfis_nonHyb import NoHybridANFIS
from anfis_hybrid import HybridANFIS
import torch.nn.functional as F
from anfisHelper import initialize_mfs_with_kmeans
from PopFnn import POPFNN
from kmFmmc import FMNC
from torch import nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import metrics
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from torch.utils.data import DataLoader, TensorDataset
from anfisHelper import initialize_mfs_with_kmeans
from trainAnfis import train_anfis_noHyb
import itertools
import pandas as pd  # for CSV export
from lb_scratch import (GraphSSL, fmv_clp, mv_grf_predict, aw_mv_grf_predict,
                        clc_mv_grf_predict, fap_predict)
from csv_logger import initialize_csv, append_to_csv
from feature_extractor_factory import prepare_feature_extractor
# ---------------------------------------------------------------------
#  Central experiment settings
# ---------------------------------------------------------------------
import experiment_config as cfg

# Read once from cfg.SETTINGS
MODEL_KEYS   = cfg.SETTINGS["models"]
LABEL_FRAC   = cfg.SETTINGS["label_frac"]
GRAPH_GRID   = cfg.SETTINGS["graph_grid"]     # list[(k, sigma)]
GRF_PAR      = cfg.SETTINGS["grf_params"]
FMV_PAR      = cfg.SETTINGS["fmv_params"]
RST_PAR      = cfg.SETTINGS["rst_params"]
results_rows = []
initialize_csv()  # start / overwrite csv each run


class ExperimentRunner:
    @staticmethod
    def run_clc_mv_grf(mf_train, rule_train, y_semi,
                       mf_test, rule_test,
                       X_train_comb, X_test_comb, y_test, k,
                       *, sigma_m=1.0, sigma_r=0.5, gamma=1.0):
        """Coupled‑Laplacian Consensus GRF."""
        y_hat, _ = clc_mv_grf_predict(
            M=mf_train, R=rule_train, y_init=y_semi,
            k=k, sigma_M=sigma_m, sigma_R=sigma_r,
            gamma=gamma)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_comb, y_hat)
        y_pred = knn.predict(X_test_comb)
        test_acc = accuracy_score(y_test.numpy(), y_pred)
        print(f"  CLC‑MV‑GRF Test Accuracy: {test_acc * 100:.2f}%")
        return test_acc
    @staticmethod
    def run_aw_mv_grf(mf_train, rule_train, y_semi,
                      mf_test, rule_test,
                      X_train_comb, X_test_comb, y_test, k,
                      *, sigma_m=1.0, sigma_r=0.5):
        y_hat, _ = aw_mv_grf_predict(
            M=mf_train, R=rule_train, y_init=y_semi,
            k=k, sigma_M=sigma_m, sigma_R=sigma_r)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_comb, y_hat)
        y_pred = knn.predict(X_test_comb)
        test_acc = accuracy_score(y_test.numpy(), y_pred)
        print(f"  AW-MV-GRF Test Accuracy: {test_acc * 100:.2f}%")
        return test_acc
    @staticmethod
    def _log_csv(dataset:str, model:str, ssl_meth:str,
                lab_frac:float, acc:float):
        append_to_csv([dataset, model, ssl_meth,
                       f"{lab_frac:.2f}", f"{acc*100:.2f}"])
    @staticmethod
    def run_rule_self_training(model,             # NoHybridANFIS | HybridANFIS
                               X_l, y_l,          # initial labelled tensors
                               X_u,               # unlabeled pool  (tensor)
                               *,
                               tau_conf   = 0.90,
                               tau_fire   = 0.50,
                               max_rounds = 10,
                               warm_epochs=200,
                               retrain_epochs=40,
                               lr_premise = 5e-3,
                               lr_conseq  = 5e-3,
                               device      = 'cuda'):
        """
        One‑cycle self‑training based on the pseudo.txt algorithm.
        Returns the fine‑tuned model.
        """
        C = model.num_classes
        model = model.to(device)

        # ----------- helper for one epoch supervised training -------------
        def _train_epoch(X, y, weights):
            loader = DataLoader(TensorDataset(X, y, weights),
                                batch_size=512, shuffle=True, drop_last=False)
            model.train()
            for xb, yb, wb in loader:
                xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
                logits, fs, _ = model(xb)
                loss_vec = F.cross_entropy(logits, yb, reduction='none')
                loss = (wb * loss_vec).mean()
                opt.zero_grad(); loss.backward(); opt.step()

        # ----------- 0) warm‑up on labelled data --------------------------
        opt = torch.optim.Adam(model.parameters(), lr=lr_premise)
        for _ in range(warm_epochs):
            _train_epoch(X_l, y_l, torch.ones(len(y_l), device=device))

        # -----------  loop over pseudo‑labelling rounds -------------------
        # keep the pool tensor on the *same* device as the model to avoid
        # device‑mismatch errors during advanced indexing
        X_pool = X_u.clone().to(device)
        for t in range(max_rounds):
            if len(X_pool) == 0: break

            # 1. compute rule confidences on labelled set
            with torch.no_grad():
                model.eval()
                _, fs_L, _ = model(X_l.to(device))
                rule_class_w = fs_L.T @ F.one_hot(y_l.to(device), C).float()
                probs   = F.normalize(rule_class_w, p=1, dim=1)  # [R,C]
                conf, cls = probs.max(1)

            # 2. scan unlabeled pool mini‑batch wise
            new_idx = []
            new_cls = []
            new_w   = []
            for s in range(0, len(X_pool), 4096):
                xb = X_pool[s:s+4096]
                _, fs_b, _ = model(xb.to(device))
                fire, r = fs_b.max(1)                       # best rule
                keep = (conf[r] >= tau_conf) & (fire >= tau_fire)
                if keep.any():
                    idx_local = torch.nonzero(keep).flatten()
                    new_idx.append(idx_local + s)
                    new_cls.append(cls[r[idx_local]])
                    new_w.append(conf[r[idx_local]])
            if not new_idx:
                break

            new_idx = torch.cat(new_idx)
            new_idx = new_idx.to(device)            # ensure index tensor on same device
            y_pseudo = torch.cat(new_cls)
            w_pseudo = torch.cat(new_w)

            # 3. merge & retrain
            X_l = torch.cat([X_l, X_pool[new_idx]])
            y_l = torch.cat([y_l, y_pseudo])
            weights = torch.cat([torch.ones(len(y_l)-len(y_pseudo), device=device),
                                 w_pseudo])

            # shrink pool
            mask = torch.ones(len(X_pool), dtype=torch.bool, device=device)
            mask[new_idx] = False
            X_pool = X_pool[mask].clone()

            # re‑initialise optimiser (keep LR smaller after first round)
            opt = torch.optim.Adam(model.parameters(), lr=lr_premise*0.5)
            for _ in range(retrain_epochs):
                _train_epoch(X_l, y_l, weights)

            # relax threshold for next round
            tau_conf *= 0.95

        return model
    @staticmethod
    def run_supervised_baseline(model, X_l, y_l, X_test, y_test, device):
        """
        Compute a simple supervised baseline accuracy for *any* extractor.

        * For torch‑based models (ANFIS, POPFNN …) we call the network
          in eval‑mode and take `argmax` over the logits.

        * For the FMNC (Fuzzy‑Min‑Max Classifier) we use its built‑in
          `predict` method, because it is **not** a callable torch Module.
        """
        if isinstance(model, FMNC):
            # ── FMNC works completely on CPU / NumPy ─────────────────────
            y_pred = model.predict(X_test.cpu())
            acc    = accuracy_score(y_test.numpy(), y_pred.cpu().numpy())
        else:
            # ── Standard torch model ─────────────────────────────────────
            model.eval()
            with torch.no_grad():
                logits, *_ = model(X_test.to(device))
                y_pred = logits.argmax(dim=1).cpu().numpy()
            acc = accuracy_score(y_test.numpy(), y_pred)

        print(f"  Supervised Baseline Accuracy: {acc * 100:.2f}%")
        return acc

    @staticmethod
    def run_rule_space_ssl(rule_train, y_semi, rule_test, y_train, y_test, device, k, sigma):
        ssl = GraphSSL(k=k, sigma=sigma, method='grf', device=device)
        
        ssl.fit(rule_train, y_semi)

        unl_mask = (y_semi == -1)
        pseudo_acc = accuracy_score(y_train.numpy()[unl_mask], ssl.transduction_[unl_mask])
        print(f"  Rule-Space Pseudo-Label Accuracy: {pseudo_acc * 100:.2f}%")
        y_pred = ssl.predict(rule_test)
        test_acc = accuracy_score(y_test.numpy(), y_pred)
        print(f"  Rule-Space Test Accuracy: {test_acc * 100:.2f}%")
        rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        # scikit-learn models work with numpy arrays on the CPU
        rf_baseline.fit(X_l.cpu().numpy(), y_l.cpu().numpy())
        y_pred_rf = rf_baseline.predict(X_test.cpu().numpy())
        rf_acc = accuracy_score(y_test.numpy(), y_pred_rf)
        print(f"  Final Test Accuracy (Supervised RF Baseline): {rf_acc * 100:.2f}%")
        return pseudo_acc, test_acc

    @staticmethod
    def run_mf_space_ssl(mf_train, y_semi, mf_test, y_train, y_test, device, k, sigma):
        ssl = GraphSSL(k=k, sigma=sigma, method='mf_space', device=device)
        ssl.fit(mf_train, y_semi)
        unl_mask = (y_semi == -1)
        pseudo_acc = accuracy_score(y_train.numpy()[unl_mask], ssl.transduction_[unl_mask])
        print(f"  MF-Space Pseudo-Label Accuracy: {pseudo_acc * 100:.2f}%")
        y_pred = ssl.predict(mf_test)
        test_acc = accuracy_score(y_test.numpy(), y_pred)
        print(f"  MF-Space Test Accuracy: {test_acc * 100:.2f}%")
        return pseudo_acc, test_acc

    @staticmethod
    def run_raw_space_ssl(X_train_np, y_semi, X_test_np, y_train, y_test, k):
        # --- Robust raw‑space baseline: z‑score scaling + scikit‑learn LabelPropagation
        # 1) Standard scale the features so that distance‑based k‑NN behaves sensibly
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_np)
        X_test_scaled  = scaler.transform(X_test_np)

        # 2) k‑NN kernel LP
        lp = LabelPropagation(kernel="knn", n_neighbors=k, max_iter=1000)
        lp.fit(X_train_scaled, y_semi)

        unl_mask   = (y_semi == -1)
        pseudo_acc = accuracy_score(y_train.numpy()[unl_mask], lp.transduction_[unl_mask])
        print(f"  Raw‑Space Pseudo‑Label Accuracy: {pseudo_acc * 100:.2f}%")

        y_pred   = lp.predict(X_test_scaled)
        test_acc = accuracy_score(y_test.numpy(), y_pred)
        print(f"  Raw‑Space Test Accuracy: {test_acc * 100:.2f}%")
        return pseudo_acc, test_acc

    @staticmethod
    def run_fmv_clp(mf_train, rule_train, y_semi, mf_test, rule_test, X_train_comb, X_test_comb, y_test, k, *, sigma_m=1.0, sigma_r=0.5, beta=0.5, alpha=0.9, k_thr=1.0):
        y_hat, _ = fmv_clp(M=mf_train, R=rule_train, y_init=y_semi, k=k, sigma_M=sigma_m, sigma_R=sigma_r, beta=beta, alpha=alpha, k_thr=k_thr)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_comb, y_hat)
        y_pred = knn.predict(X_test_comb)
        test_acc = accuracy_score(y_test.numpy(), y_pred)
        print(f"  FMV-CLP Test Accuracy: {test_acc * 100:.2f}%")
        return test_acc

    @staticmethod
    def run_mv_grf(mf_train, rule_train, y_semi, mf_test, rule_test, X_train_comb, X_test_comb, y_test, k, *, sigma_m=1.0, sigma_r=0.5, beta=0.5):
        y_hat, _ = mv_grf_predict(M=mf_train, R=rule_train, y_init=y_semi, k=k, sigma_M=sigma_m, sigma_R=sigma_r, beta=beta)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_comb, y_hat)
        y_pred = knn.predict(X_test_comb)
        test_acc = accuracy_score(y_test.numpy(), y_pred)
        print(f"  MV-GRF Test Accuracy: {test_acc * 100:.2f}%")
        return test_acc
    
    @staticmethod
    def run_fap(mf_train, rule_train, y_semi,
                mf_test, rule_test, y_test, k,
                *, sigma_m=1.0, sigma_r=0.5,
                alpha=0.9, max_iter=1000, device='cpu'):
        print("\n--- Running Multi‑View SSL (FAP) ---")
        comb_train = np.hstack([mf_train, rule_train])
        comb_test  = np.hstack([mf_test,  rule_test])

        y_hat, _ = fap_predict(mf_train, rule_train, y_semi,
                                k=k, sigma_M=sigma_m, sigma_R=sigma_r,
                                alpha=alpha, max_iter=max_iter,
                                device=device)
        _dbg("FAP y_hat", torch.from_numpy(y_hat))
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(comb_train, y_hat)
        y_pred = knn.predict(comb_test)
        test_acc = accuracy_score(y_test.numpy(), y_pred)
        print(f"  FAP Test Accuracy: {test_acc * 100:.2f}%")
        return test_acc

from anfisHelper import _dbg
    
    
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    X_train, y_train, X_test, y_test = load_K_chess_data_splitted()
    #X_train, y_train, X_test, y_test = load_htru_data()

    
    n_labeled = int(LABEL_FRAC * len(y_train))

    labeled_indices = np.random.choice(np.arange(len(y_train)), size=n_labeled, replace=False)
    y_semi_sup = np.full(len(y_train), -1, dtype=np.int64)
    y_semi_sup[labeled_indices] = y_train[labeled_indices]
    X_l, y_l = X_train[labeled_indices], y_train[labeled_indices]

    # Train ANFIS models (two variants), using config params
    params_nohyb = cfg.SETTINGS["model_params"].get("NoHybridANFIS", {})
    anfis_model = NoHybridANFIS(
        input_dim=X_train.shape[1],
        num_classes=len(y_train.unique()),
        num_mfs = params_nohyb.get("num_mfs", 3),
        max_rules = params_nohyb.get("max_rules", 1000),
        seed=42
    ).to(device)

    # --- identical training schedule as codecach.py ---
    #initialize_mfs_with_kmeans(anfis_model, X_train)
    train_anfis_noHyb(
        anfis_model,
        X_l.to(device), y_l.to(device),
        X_train,
        num_epochs=params_nohyb.get("epochs", 400),
        lr=params_nohyb.get("lr", 0.01)
    )
    model = anfis_model

    runner = ExperimentRunner()

    # --- run the full codecach experiment suite for BOTH ANFIS variants ---
    for k, sig in GRAPH_GRID:
        print(f"\nRunning experiments with k={k}, sigma={sig}...")
        model.eval()
        with torch.no_grad():
            _, rule_activations_train, _ = model(X_train.to(device))
            _, rule_activations_test, _ = model(X_test.to(device))
            mf_values_train = model._fuzzify(X_train.to(device))
            mf_values_test  = model._fuzzify(X_test.to(device))

        # Numpy views
        rule_train_np = rule_activations_train.cpu().numpy()
        rule_test_np  = rule_activations_test.cpu().numpy()
        mf_train_np   = mf_values_train.reshape(mf_values_train.shape[0], -1).cpu().numpy()
        mf_test_np    = mf_values_test.reshape(mf_values_test.shape[0], -1).cpu().numpy()
        X_train_np    = X_train.cpu().numpy()
        X_test_np     = X_test.cpu().numpy()

        # Normalized versions for MF and Rule
   
        
        
        rule_train = F.normalize(rule_activations_train, p=2, dim=1)
        rule_test  = F.normalize(rule_activations_test , p=2, dim=1)

        mf_train   = F.normalize(mf_values_train.reshape(len(mf_values_train), -1), p=2, dim=1)
        mf_test    = F.normalize(mf_values_test .reshape(len(mf_values_test ), -1), p=2, dim=1)
        
        rule_train_np = rule_train.cpu().numpy()
        rule_test_np  = rule_test.cpu().numpy()
        mf_train_norm_np = mf_train.cpu().numpy()
        mf_test_norm_np  = mf_test.cpu().numpy()
        
                # --- Debug: Feature stats ----------------------------------------
        _dbg("rule_train", rule_train)
        _dbg("mf_train",   mf_train)
        _dbg("rule_test",  rule_test)
        _dbg("mf_test",    mf_test)        
        

        # 1) Supervised baseline
        acc_sup = runner.run_supervised_baseline(model, X_l, y_l, X_test, y_test, device)
        runner._log_csv("KChess", "NoHybridANFIS", "Supervised", LABEL_FRAC, acc_sup)

        # --- Rule-Self-Training SSL ---
        print("\n--- Running Rule‑Self‑Training SSL ---")
        model_ssl = ExperimentRunner.run_rule_self_training(
                        model,
                        X_l.to(device), y_l.to(device),
                        X_train[~torch.tensor(np.isin(np.arange(len(y_train)), labeled_indices))],
                        device=device,
                        **RST_PAR
        )

        # 2) Rule‑space GRF
        _, acc_rule = runner.run_rule_space_ssl(rule_train, y_semi_sup,
                                                rule_test, y_train, y_test,
                                                device, k=k, sigma=sig)
        runner._log_csv("KChess", "NoHybridANFIS", "Rule‑GRF",    LABEL_FRAC, acc_rule)

        # 3) MF‑space GRF
        _, acc_mf = runner.run_mf_space_ssl(mf_train_norm_np, y_semi_sup, mf_test_norm_np,
                                           y_train, y_test, device, k=k, sigma=sig)
        runner._log_csv("KChess", "NoHybridANFIS", "MF‑GRF",      LABEL_FRAC, acc_mf)

        # 4) Raw‑space LP
        _, acc_raw = runner.run_raw_space_ssl(X_train_np, y_semi_sup, X_test_np,
                                              y_train, y_test, k=k)
        runner._log_csv("KChess", "NoHybridANFIS", "Raw‑LP",      LABEL_FRAC, acc_raw)

        # 5) FMV‑CLP
        comb_train = np.hstack([mf_train_norm_np, rule_train_np])
        comb_test  = np.hstack([mf_test_norm_np,  rule_test_np])
        acc_fmv = runner.run_fmv_clp(
            mf_train_norm_np, rule_train_np, y_semi_sup,
            mf_test_norm_np, rule_test_np,
            comb_train, comb_test, y_test, k=k,
            **FMV_PAR
        )
        runner._log_csv("KChess", "NoHybridANFIS", "FMV‑CLP",     LABEL_FRAC, acc_fmv)

        # 6) MV‑GRF
        acc_mv = runner.run_mv_grf(
            mf_train_norm_np, rule_train_np, y_semi_sup,
            mf_test_norm_np, rule_test_np,
            comb_train, comb_test, y_test, k=k,
            **GRF_PAR
        )
        runner._log_csv("KChess", "NoHybridANFIS", "MV‑GRF",      LABEL_FRAC, acc_mv)

        # 6b) AW‑MV‑GRF
        acc_aw = runner.run_aw_mv_grf(
            mf_train_norm_np, rule_train_np, y_semi_sup,
            mf_test_norm_np, rule_test_np,
            comb_train, comb_test, y_test, k=k,
            sigma_m=sig, sigma_r=0.5
        )
        runner._log_csv("KChess", "NoHybridANFIS", "AW‑MV‑GRF", LABEL_FRAC, acc_aw)

        # 6c) CLC‑MV‑GRF
        acc_clc = runner.run_clc_mv_grf(
            mf_train_norm_np, rule_train_np, y_semi_sup,
            mf_test_norm_np, rule_test_np,
            comb_train, comb_test, y_test, k=k,
            sigma_m=sig, sigma_r=0.5, gamma=1.0
        )
        runner._log_csv("KChess", "NoHybridANFIS", "CLC‑MV‑GRF", LABEL_FRAC, acc_clc)

        # 7) FAP
        acc_fap = runner.run_fap(mf_train_norm_np, rule_train_np, y_semi_sup,
                                 mf_test_norm_np, rule_test_np, y_test,
                                 k=k, sigma_m=sig, sigma_r=0.5,
                                 alpha=0.9, max_iter=100, device=device)
        runner._log_csv("KChess", "NoHybridANFIS", "FAP",         LABEL_FRAC, acc_fap)

        results_rows.append({"model": f"FAP_{k}", "k": k,
                             "metric": "euclidean", "acc": acc_fap})
        
    
    for model_key in MODEL_KEYS:
        print(f"\n=========== {model_key} ===========")

        # ❶ train the extractor + get all views -------------
        feats = prepare_feature_extractor(
            model_key,
            X_train, y_train, X_test,
            frac_labeled=LABEL_FRAC,
            device=device,
            model_kwargs=cfg.SETTINGS["model_params"].get(model_key, {})
        )
        model          = feats["model"]
        rule_train_np  = feats["rule_train"].numpy()
        rule_test_np   = feats["rule_test"].numpy()
        mf_train_np    = feats["mf_train"].numpy()
        mf_test_np     = feats["mf_test"].numpy()
        labelled_idx   = feats["labeled_indices"]

        # semi-sup label vector
        y_semi = np.full(len(y_train), -1, dtype=np.int64)
        y_semi[labelled_idx] = y_train[labelled_idx]

        X_l = X_train[labelled_idx]
        y_l = y_train[labelled_idx]

        # ❷ run all SSL baselines & log to CSV --------------
        for k, sigma in GRAPH_GRID:
            """acc_sup = ExperimentRunner.run_supervised_baseline(
                        model, X_l, y_l, X_test, y_test, device)
            ExperimentRunner._log_csv("KChess", model_key, "Supervised", LABEL_FRAC, acc_sup)

            _, acc_rule = ExperimentRunner.run_rule_space_ssl(
                            rule_train_np, y_semi, rule_test_np,
                            y_train, y_test, device, k, sigma)
            ExperimentRunner._log_csv("KChess", model_key, "Rule-GRF", LABEL_FRAC, acc_rule)

            _, acc_mf = ExperimentRunner.run_mf_space_ssl(
                            mf_train_np, y_semi, mf_test_np,
                            y_train, y_test, device, k, sigma)
            ExperimentRunner._log_csv("KChess", model_key, "MF-GRF", LABEL_FRAC, acc_mf)

            _, acc_raw = ExperimentRunner.run_raw_space_ssl(
                            X_train.cpu().numpy(), y_semi,
                            X_test.cpu().numpy(), y_train, y_test, k)
            ExperimentRunner._log_csv("KChess", model_key, "Raw-LP", LABEL_FRAC, acc_raw)"""

            comb_tr = np.hstack([mf_train_np, rule_train_np])
            comb_te = np.hstack([mf_test_np,  rule_test_np])

            """acc_fmv = ExperimentRunner.run_fmv_clp(
                        mf_train_np, rule_train_np, y_semi,
                        mf_test_np, rule_test_np,
                        comb_tr, comb_te, y_test, k,
                        **FMV_PAR
            )
            ExperimentRunner._log_csv("KChess", model_key, "FMV-CLP", LABEL_FRAC, acc_fmv)

            acc_mv  = ExperimentRunner.run_mv_grf(
                        mf_train_np, rule_train_np, y_semi,
                        mf_test_np, rule_test_np,
                        comb_tr, comb_te, y_test, k,
                        **GRF_PAR
            )
            ExperimentRunner._log_csv("KChess", model_key, "MV-GRF", LABEL_FRAC, acc_mv)"""

            acc_aw = ExperimentRunner.run_aw_mv_grf(
                        mf_train_np, rule_train_np, y_semi,
                        mf_test_np, rule_test_np,
                        comb_tr, comb_te, y_test, k,
                        sigma_m=sigma, sigma_r=0.5
            )
            ExperimentRunner._log_csv("KChess", model_key, "AW-MV-GRF", LABEL_FRAC, acc_aw)

            acc_clc = ExperimentRunner.run_clc_mv_grf(
                        mf_train_np, rule_train_np, y_semi,
                        mf_test_np, rule_test_np,
                        comb_tr, comb_te, y_test, k,
                        sigma_m=sigma, sigma_r=0.5, gamma=1.0
            )
            ExperimentRunner._log_csv("KChess", model_key, "CLC‑MV‑GRF", LABEL_FRAC, acc_clc)
