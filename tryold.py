import torch
import numpy as np
import pandas as pd
import time
import itertools
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch import nn
import scipy.sparse as sp
from scipy.sparse.linalg import cg

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import accuracy_score
from sklearn.neighbors import kneighbors_graph, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelPropagation

# Assuming these are your local utility files
from data_utils import load_K_chess_data_splitted, load_htru_data, load_pmd_data, load_letter_data
from anfis_nonHyb import NoHybridANFIS
from trainAnfis import train_anfis_noHyb

# Global results list for all experiments/results
results_rows = []


# +---------------------------------------------------------------------------------+
# |  Multi-View SSL Algorithm Implementations & Helpers (FMV-CLP, MV-GRF)             |
# +---------------------------------------------------------------------------------+

def _row_normalise_sparse(mat: sp.csr_matrix) -> sp.csr_matrix:
    """Row-stochastic normalisation for a CSR matrix."""
    row_sum = np.asarray(mat.sum(1)).flatten()
    row_sum[row_sum == 0.] = 1.
    inv = 1. / row_sum
    inv_mat = sp.diags(inv)
    return inv_mat @ mat

def _rbf_weight(dist2: np.ndarray, sigma: float):
    return np.exp(-dist2 / (2. * sigma * sigma))

def _build_knn_rbf_graph(view: np.ndarray, k: int, sigma: float) -> sp.csr_matrix:
    """k-NN graph with RBF weights (symmetrised)."""
    knn_dist = kneighbors_graph(
        view, n_neighbors=k, mode="distance",
        include_self=False, metric="euclidean", n_jobs=-1
    )
    dist2 = knn_dist.data ** 2
    knn_dist.data = _rbf_weight(dist2, sigma)
    W = 0.5 * (knn_dist + knn_dist.T)
    return W.tocsr()

def fmv_clp(M: np.ndarray, R: np.ndarray, y_init: np.ndarray,
            k: int = 10, sigma_M: float = 1.0, sigma_R: float = 1.0,
            alpha: float = 0.9, beta: float = 0.5, k_thr: float = 1.0,
            max_outer: int = 20, device: str = "cpu"):
    """FMV-CLP: Label propagation that mixes MF-view (M) and Rule-view (R)."""
    assert M.shape[0] == R.shape[0] == y_init.shape[0]
    n = M.shape[0]
    classes = np.unique(y_init[y_init != -1])
    C = len(classes)
    class_map = {c: i for i, c in enumerate(classes)}
    labeled_mask = (y_init != -1)

    # 1) build graphs (CPU, SciPy)
    W_M = _build_knn_rbf_graph(M, k, sigma_M)
    W_R = _build_knn_rbf_graph(R, k, sigma_R)

    # 2) row-normalise → transition matrices
    S_M = _row_normalise_sparse(W_M)
    S_R = _row_normalise_sparse(W_R)

    # Convert to torch sparse once
    def _csr_to_torch(csr):
        coo = csr.tocoo()
        idx = torch.vstack((torch.from_numpy(coo.row), torch.from_numpy(coo.col)))
        val = torch.from_numpy(coo.data)
        return torch.sparse_coo_tensor(idx, val, size=csr.shape, device=device, dtype=torch.float32)

    S_M_t = _csr_to_torch(S_M)
    S_R_t = _csr_to_torch(S_R)

    # 3) Y0 seeds
    Y0 = torch.zeros((n, C), dtype=torch.float32, device=device)
    for i, lbl in enumerate(y_init):
        if lbl != -1:
            Y0[i, class_map[lbl]] = 1.

    # 4) Scores
    F_M = Y0.clone()
    F_R = Y0.clone()

    # Outer curriculum loop
    current_labeled_mask = labeled_mask.copy()
    for outer_iter in range(max_outer):
        # 5a propagation
        F_M = alpha * torch.sparse.mm(S_M_t, F_M) + (1 - alpha) * Y0
        F_R = alpha * torch.sparse.mm(S_R_t, F_R) + (1 - alpha) * Y0

        # 5b combine
        F_comb = beta * F_M + (1 - beta) * F_R

        # 5c per-class adaptive τ
        F_np = F_comb.detach().cpu().numpy()
        new_seeds = []
        
        unlabeled_indices = np.where(~current_labeled_mask)[0]
        if len(unlabeled_indices) == 0: break

        for c in range(C):
            conf_c = F_np[unlabeled_indices, c]
            if len(conf_c) == 0: continue
            
            mu, std = conf_c.mean(), conf_c.std(ddof=0)
            tau = mu + k_thr * std
            
            confident_mask = conf_c >= tau
            confident_global_indices = unlabeled_indices[confident_mask]
            
            new_seeds.extend([(idx, c) for idx in confident_global_indices])

        if not new_seeds:
            # print(f"    [FMV-CLP Round {outer_iter+1}] No new seeds found. Stopping.")
            break

        # 5f add new seeds
        num_added = 0
        for idx, c in new_seeds:
            if not current_labeled_mask[idx]:
                Y0[idx, :] = 0.
                Y0[idx, class_map[classes[c]]] = 1.
                current_labeled_mask[idx] = True
                num_added += 1
        
        if num_added == 0: break

    # 6) final prediction
    y_hat_idx = F_comb.argmax(1).cpu().numpy()
    idx2cls = {v: k for k, v in class_map.items()}
    y_hat = np.vectorize(idx2cls.get)(y_hat_idx)

    return y_hat, F_comb

def mv_grf_predict(
        M, R, y_init, k=10, sigma_M=1.0, sigma_R=1.0,
        beta=0.5, reg_eps=1e-5, device="cpu"):
    
    if isinstance(M, torch.Tensor): M = M.cpu().numpy()
    if isinstance(R, torch.Tensor): R = R.cpu().numpy()
    if isinstance(y_init, torch.Tensor): y_init = y_init.cpu().numpy()

    n = M.shape[0]
    classes = np.unique(y_init[y_init >= 0])
    C = len(classes)
    class_map = {c: i for i, c in enumerate(classes)}

    W_M = _build_knn_rbf_graph(M, k, sigma_M)
    W_R = _build_knn_rbf_graph(R, k, sigma_R)

    W = beta * W_M + (1.0 - beta) * W_R
    d = np.asarray(W.sum(axis=1)).flatten()
    L_csr = sp.diags(d) - W
    
    labeled_mask = (y_init >= 0)
    unlabeled_mask = ~labeled_mask
    idx_L = np.where(labeled_mask)[0]
    idx_U = np.where(unlabeled_mask)[0]

    L_uu = L_csr[idx_U][:, idx_U]
    L_ul = L_csr[idx_U][:, idx_L]

    f_l = torch.zeros(len(idx_L), C, device=device)
    y_labeled_mapped = np.array([class_map[lbl] for lbl in y_init[idx_L]])
    f_l[torch.arange(len(idx_L)), torch.from_numpy(y_labeled_mapped)] = 1.0

    I_uu = sp.identity(L_uu.shape[0], format='csr') * reg_eps
    A = L_uu + I_uu
    b = -L_ul @ f_l.cpu().numpy()

    f_u_list = [cg(A, b[:, c])[0] for c in range(C)]
    f_u = torch.from_numpy(np.vstack(f_u_list).T).float().to(device)

    F_final = torch.zeros(n, C, device=device)
    F_final[idx_L] = f_l
    F_final[idx_U] = F.softmax(f_u, dim=1)

    y_hat_idx = F_final.argmax(dim=1).cpu().numpy()
    idx2cls = {v: k for k, v in class_map.items()}
    y_hat = np.vectorize(idx2cls.get)(y_hat_idx)

    return y_hat, F_final.cpu()


# +---------------------------------------------------------------------------------+
# |  Single-View SSL Algorithm Implementation (GraphSSL)                            |
# +---------------------------------------------------------------------------------+

class GraphSSL(BaseEstimator, ClassifierMixin):
    """Memory-efficient, sparse-graph implementation of single-view SSL."""
    def __init__(self, k=10, sigma=1.0, max_iter=100, tol=1e-6,
                 alpha=0.99, method='grf'):
        self.k = k
        self.sigma = sigma
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.method = method

    def _prepare_fit(self, X, y):
        if isinstance(X, torch.Tensor): X_np = X.detach().cpu().numpy()
        else: X_np = np.array(X)
        if isinstance(y, torch.Tensor): y_np = y.detach().cpu().numpy()
        else: y_np = np.array(y)
        if X_np.ndim > 2: X_np = X_np.reshape(X_np.shape[0], -1)

        self.X_train_np_ = X_np
        self.classes_ = np.unique(y_np[y_np != -1])
        self.n_classes_ = len(self.classes_)
        n_samples = X_np.shape[0]
        Y_ = np.zeros((n_samples, self.n_classes_))
        labeled_mask = (y_np != -1)
        for i, cls in enumerate(self.classes_): Y_[y_np == cls, i] = 1
        return X_np, y_np, Y_, labeled_mask

    def fit_grf(self, X, y):
        X_np, y_np, Y_, labeled_mask = self._prepare_fit(X, y)
        unlabeled_mask = ~labeled_mask
        W = _build_knn_rbf_graph(X_np, self.k, self.sigma)
        D = sp.diags(np.asarray(W.sum(axis=1)).flatten())
        L = D - W
        L_uu = L[unlabeled_mask, :][:, unlabeled_mask]
        L_ul = L[unlabeled_mask, :][:, labeled_mask]
        rhs = -L_ul @ Y_[labeled_mask]
        f_u_list = []
        for i in range(self.n_classes_):
            b = rhs[:, i]
            reg = sp.identity(L_uu.shape[0]) * 1e-5
            f_u_col, _ = cg(L_uu + reg, b, tol=1e-6, maxiter=1000)
            f_u_list.append(f_u_col)
        f_u = np.asarray(f_u_list).T
        self.label_distributions_ = np.zeros_like(Y_)
        self.label_distributions_[labeled_mask] = Y_[labeled_mask]
        exp_f_u = np.exp(f_u - np.max(f_u, axis=1, keepdims=True))
        self.label_distributions_[unlabeled_mask] = exp_f_u / exp_f_u.sum(axis=1, keepdims=True)
        self.transduction_ = self.classes_[self.label_distributions_.argmax(1)]
        return self

    def fit_iterative(self, X, y):
        X_np, y_np, Y_init, labeled_mask = self._prepare_fit(X, y)
        W = _build_knn_rbf_graph(X_np, self.k, self.sigma)
        D_inv = sp.diags(1.0 / (np.asarray(W.sum(axis=1)).flatten() + 1e-12))
        S = D_inv @ W
        F = Y_init.copy()
        for _ in range(self.max_iter):
            F_old = F.copy()
            F = self.alpha * (S @ F) + (1 - self.alpha) * Y_init
            if np.abs(F - F_old).sum() < self.tol: break
        self.label_distributions_ = F
        self.transduction_ = self.classes_[self.label_distributions_.argmax(1)]
        return self

    def fit(self, X, y):
        if self.method == 'grf': return self.fit_grf(X, y)
        elif self.method == 'iterative': return self.fit_iterative(X, y)
        else: raise NotImplementedError(f"Method '{self.method}' is not implemented.")

    def predict(self, X):
        check_is_fitted(self)
        if isinstance(X, torch.Tensor): X_np = X.detach().cpu().numpy()
        else: X_np = np.array(X)
        if X_np.ndim > 2: X_np = X_np.reshape(X_np.shape[0], -1)
        knn = KNeighborsClassifier(n_neighbors=self.k)
        knn.fit(self.X_train_np_, self.transduction_)
        return knn.predict(X_np)


# +---------------------------------------------------------------------------------+
# |  Experiment Runner Class                                                        |
# +---------------------------------------------------------------------------------+

class ExperimentRunner:
    """A class to encapsulate the logic for running various SSL experiments."""
    @staticmethod
    def run_supervised_baseline(model, X_test, y_test, device):
        model.eval()
        with torch.no_grad():
            logits, *_ = model(X_test.to(device))
            preds = logits.argmax(dim=1).cpu().numpy()
        acc = accuracy_score(y_test.numpy(), preds)
        print(f"  Supervised Baseline Accuracy: {acc * 100:.2f}%")
        return acc

    @staticmethod
    def run_single_view_ssl(features_train, y_semi, features_test, y_test, method, k, sigma):
        print(f"\n--- Running Single-View SSL ({method}) ---")
        ssl = GraphSSL(k=k, sigma=sigma, method=method)
        ssl.fit(features_train, y_semi)
        y_pred = ssl.predict(features_test)
        test_acc = accuracy_score(y_test.numpy(), y_pred)
        print(f"  Single-View ({method}) Test Accuracy: {test_acc * 100:.2f}%")
        return test_acc

    @staticmethod
    def run_raw_space_ssl(X_train_np, y_semi, X_test_np, y_test, k):
        print("\n--- Running Raw-Space SSL (Label Propagation) ---")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_np)
        X_test_scaled  = scaler.transform(X_test_np)
        lp = LabelPropagation(kernel="knn", n_neighbors=k, max_iter=1000, n_jobs=-1)
        lp.fit(X_train_scaled, y_semi)
        y_pred   = lp.predict(X_test_scaled)
        test_acc = accuracy_score(y_test.numpy(), y_pred)
        print(f"  Raw‑Space Test Accuracy: {test_acc * 100:.2f}%")
        return test_acc

    @staticmethod
    def run_fmv_clp(mf_train, rule_train, y_semi, mf_test, rule_test, y_test, k, **kwargs):
        print("\n--- Running Multi-View SSL (FMV-CLP) ---")
        comb_train = np.hstack([mf_train, rule_train])
        comb_test  = np.hstack([mf_test,  rule_test])
        y_hat, _ = fmv_clp(M=mf_train, R=rule_train, y_init=y_semi, k=k, **kwargs)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(comb_train, y_hat)
        y_pred = knn.predict(comb_test)
        test_acc = accuracy_score(y_test.numpy(), y_pred)
        print(f"  FMV-CLP Test Accuracy: {test_acc * 100:.2f}%")
        return test_acc

    @staticmethod
    def run_mv_grf(mf_train, rule_train, y_semi, mf_test, rule_test, y_test, k, **kwargs):
        print("\n--- Running Multi-View SSL (MV-GRF) ---")
        comb_train = np.hstack([mf_train, rule_train])
        comb_test  = np.hstack([mf_test,  rule_test])
        y_hat, _ = mv_grf_predict(M=mf_train, R=rule_train, y_init=y_semi, k=k, **kwargs)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(comb_train, y_hat)
        y_pred = knn.predict(comb_test)
        test_acc = accuracy_score(y_test.numpy(), y_pred)
        print(f"  MV-GRF Test Accuracy: {test_acc * 100:.2f}%")
        return test_acc


# +---------------------------------------------------------------------------------+
# |  Main Execution Block                                                           |
# +---------------------------------------------------------------------------------+

if __name__ == '__main__':
    # ========================== HYPERPARAMETERS ============================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Global configuration: Using device: {device}")

    DATASET_LOADERS = [load_K_chess_data_splitted, load_htru_data]
    LABEL_FRACTIONS = [0.1]
    SEEDS = [42]

    ANFIS_TRAIN_PARAMS = {'num_epochs': 400, 'lr': 0.01, 'num_mfs': 4, 'max_rules': 100}

    SSL_GRID = [
        {'k': 7, 'sigma': 0.5, 'sigma_M': 1.0, 'sigma_R': 0.5, 'beta': 0.5, 'alpha': 0.9, 'k_thr': 1.0},
        {'k': 15, 'sigma': 0.3, 'sigma_M': 1.0, 'sigma_R': 0.3, 'beta': 0.5, 'alpha': 0.9, 'k_thr': 1.0},
    ]
    # =======================================================================

    runner = ExperimentRunner()
    
    for dataset_loader in DATASET_LOADERS:
        dataset_name = dataset_loader.__name__
        
        for seed in SEEDS:
            for label_frac in LABEL_FRACTIONS:
                print(f"\n{'='*80}\nRunning on Dataset: {dataset_name} | Seed: {seed} | Label Frac: {label_frac}\n{'='*80}")

                # --- 1. Data Loading and SSL Split ---
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

                X_train, y_train, X_test, y_test = dataset_loader()
                n_labeled = int(label_frac * len(y_train))
                labeled_indices = np.random.choice(len(y_train), size=n_labeled, replace=False)
                y_semi_sup = np.full(len(y_train), -1, dtype=np.int64)
                y_semi_sup[labeled_indices] = y_train[labeled_indices]
                X_l, y_l = X_train[labeled_indices], y_train[labeled_indices]

                # --- 2. ANFIS Model Training & Feature Extraction ---
                print("--- Training ANFIS & Extracting Views ---")
                anfis_model = NoHybridANFIS(
                    input_dim=X_train.shape[1], num_classes=len(y_train.unique()),
                    num_mfs=ANFIS_TRAIN_PARAMS['num_mfs'], max_rules=ANFIS_TRAIN_PARAMS['max_rules'], seed=seed
                ).to(device)
                train_anfis_noHyb(
                    anfis_model, X_l.to(device), y_l.to(device), X_train,
                    num_epochs=ANFIS_TRAIN_PARAMS['num_epochs'], lr=ANFIS_TRAIN_PARAMS['lr']
                )

                anfis_model.eval()
                with torch.no_grad():
                    mf_values_train = anfis_model._fuzzify(X_train.to(device))
                    mf_values_test  = anfis_model._fuzzify(X_test.to(device))
                    _, rule_activations_train, _ = anfis_model(X_train.to(device))
                    _, rule_activations_test, _  = anfis_model(X_test.to(device))
                
                # Normalize features (L2 norm) and convert to numpy
                mf_train_np = F.normalize(mf_values_train.view(len(X_train), -1), p=2).cpu().numpy()
                mf_test_np  = F.normalize(mf_values_test.view(len(X_test), -1), p=2).cpu().numpy()
                rule_train_np = F.normalize(rule_activations_train, p=2).cpu().numpy()
                rule_test_np  = F.normalize(rule_activations_test, p=2).cpu().numpy()
                X_train_np = X_train.cpu().numpy()
                X_test_np = X_test.cpu().numpy()

                # --- 3. Run Experiments for each SSL Setting ---
                for ssl_params in SSL_GRID:
                    k = ssl_params['k']
                    print(f"\n--- Running experiments for k={k} ---")

                    acc_sup = runner.run_supervised_baseline(anfis_model, X_test, y_test, device)
                    
                    acc_raw = runner.run_raw_space_ssl(X_train_np, y_semi_sup, X_test_np, y_test, k=k)

                    acc_rule_grf = runner.run_single_view_ssl(
                        rule_train_np, y_semi_sup, rule_test_np, y_test,
                        method='grf', k=k, sigma=ssl_params['sigma']
                    )
                    acc_mf_grf = runner.run_single_view_ssl(
                        mf_train_np, y_semi_sup, mf_test_np, y_test,
                        method='grf', k=k, sigma=ssl_params['sigma']
                    )

                    acc_mv_grf = runner.run_mv_grf(
                        mf_train_np, rule_train_np, y_semi_sup, mf_test_np, rule_test_np, y_test,
                        k=k, sigma_M=ssl_params['sigma_M'], sigma_R=ssl_params['sigma_R'],
                        beta=ssl_params['beta'], device=device
                    )
                    
                    acc_fmv_clp = runner.run_fmv_clp(
                        mf_train_np, rule_train_np, y_semi_sup, mf_test_np, rule_test_np, y_test,
                        k=k, sigma_M=ssl_params['sigma_M'], sigma_R=ssl_params['sigma_R'],
                        beta=ssl_params['beta'], alpha=ssl_params['alpha'],
                        k_thr=ssl_params['k_thr'], device=device
                    )
                    
                    # --- 4. Log Results ---
                    results_rows.append({
                        'dataset': dataset_name, 'seed': seed, 'label_fraction': label_frac,
                        'k': k, 'sigma': ssl_params['sigma'], 'beta': ssl_params['beta'],
                        'acc_supervised': acc_sup,
                        'acc_raw_space_lp': acc_raw,
                        'acc_rule_space_grf': acc_rule_grf,
                        'acc_mf_space_grf': acc_mf_grf,
                        'acc_mv_grf': acc_mv_grf,
                        'acc_fmv_clp': acc_fmv_clp
                    })

    # ========================== SAVE RESULTS ===============================
    if results_rows:
        results_df = pd.DataFrame(results_rows)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        results_filename = f'multi_view_experiment_results_{timestamp}.csv'
        results_df.to_csv(results_filename, index=False)
        print(f"\nAll experiments complete. Results saved to '{results_filename}'.")
        print(results_df.round(3))
    else:
        print("\nNo experiments were run or no results were generated.")