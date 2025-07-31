import torch
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.neighbors import kneighbors_graph, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelPropagation
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import time
import scipy.sparse as sp
from scipy.sparse.linalg import cg

# --- Your Local Imports ---
# NOTE: Using HybridANFIS and its corresponding trainer for better feature extraction
from data_utils import load_K_chess_data_splitted, load_htru_data, load_pmd_data, load_letter_data
from anfis_hybrid import HybridANFIS
from trainAnfis import train_anfis_hybrid # Assuming you have a trainer for the hybrid model
from sklearn.metrics import accuracy_score

# Global results list for all experiments/results
results_rows = []


# +---------------------------------------------------------------------------------+
# |  Graph Construction & SSL Algorithm Implementations                             |
# +---------------------------------------------------------------------------------+

def _build_knn_rbf_graph(view: np.ndarray, k: int, sigma: float) -> sp.csr_matrix:
    """k-NN graph with RBF (Gaussian) weights (symmetrised)."""
    # Handle potential sigma=0 case, which would cause division by zero
    if sigma == 0:
        sigma = 1e-6

    try:
        # Get distances to k nearest neighbors
        knn_dist = kneighbors_graph(
            view, n_neighbors=k, mode="distance",
            metric="euclidean", n_jobs=-1
        )
        # Apply the RBF kernel to the distances
        dist2 = knn_dist.data ** 2
        knn_dist.data = np.exp(-dist2 / (2. * sigma * sigma))
        # Symmetrize the graph to ensure undirected edges
        W = 0.5 * (knn_dist + knn_dist.T)
        return W.tocsr()
    except Exception as e:
        print(f"Error in _build_knn_rbf_graph: {e}")
        return sp.csr_matrix((view.shape[0], view.shape[0]))

def _build_knn_connectivity_graph(view: np.ndarray, k: int) -> sp.csr_matrix:
    """k-NN graph with binary connectivity weights."""
    try:
        # mode='connectivity' returns a matrix with 1s for neighbors
        W = kneighbors_graph(
            view, n_neighbors=k, mode='connectivity',
            metric="euclidean", n_jobs=-1
        )
        # Symmetrize the graph
        W = W.maximum(W.T)
        return W.tocsr()
    except Exception as e:
        print(f"Error in _build_knn_connectivity_graph: {e}")
        return sp.csr_matrix((view.shape[0], view.shape[0]))


def get_grf_propagated_labels(features: np.ndarray, y_semi: np.ndarray, k: int, sigma: float, graph_type: str):
    """Core GRF propagation logic. Returns the soft label distribution matrix F."""
    # --- Prepare Data ---
    y_np = y_semi.copy()
    classes = np.unique(y_np[y_np != -1])
    n_classes = len(classes)
    class_map = {c: i for i, c in enumerate(classes)}

    n_samples = features.shape[0]
    Y_ = np.zeros((n_samples, n_classes))
    labeled_mask = (y_np != -1)
    for i, cls in enumerate(classes):
        Y_[y_np == cls, i] = 1

    # --- Build Graph & Laplacian ---
    unlabeled_mask = ~labeled_mask
    if graph_type == 'rbf':
        W = _build_knn_rbf_graph(features, k, sigma)
    elif graph_type == 'connectivity':
        W = _build_knn_connectivity_graph(features, k)
    else:
        raise ValueError(f"Unknown graph_type: '{graph_type}'")

    D = sp.diags(np.asarray(W.sum(axis=1)).flatten())
    L = D - W

    # --- Solve System ---
    L_uu = L[unlabeled_mask, :][:, unlabeled_mask]
    L_ul = L[unlabeled_mask, :][:, labeled_mask]
    rhs = -L_ul @ Y_[labeled_mask]

    f_u_list = []
    for i in range(n_classes):
        b = rhs[:, i]
        reg = sp.identity(L_uu.shape[0]) * 1e-5
        f_u_col, _ = cg(L_uu + reg, b, tol=1e-6, maxiter=1000)
        f_u_list.append(f_u_col)

    f_u = np.asarray(f_u_list).T
    
    # --- Assemble Final Distribution ---
    F_final = np.zeros_like(Y_)
    F_final[labeled_mask] = Y_[labeled_mask]
    exp_f_u = np.exp(f_u - np.max(f_u, axis=1, keepdims=True))
    F_final[unlabeled_mask] = exp_f_u / (exp_f_u.sum(axis=1, keepdims=True) + 1e-9)

    return F_final, classes

def get_iterative_propagated_labels(features: np.ndarray, y_semi: np.ndarray, k: int, sigma: float, alpha: float, max_iter: int, graph_type: str):
    """Core iterative propagation logic. Returns the soft label distribution matrix F."""
    # --- Prepare Data ---
    y_np = y_semi.copy()
    classes = np.unique(y_np[y_np != -1])
    n_classes = len(classes)
    n_samples = features.shape[0]
    Y_init = np.zeros((n_samples, n_classes))
    for i, cls in enumerate(classes):
        Y_init[y_np == cls, i] = 1

    # --- Build Graph & Propagate ---
    if graph_type == 'rbf':
        W = _build_knn_rbf_graph(features, k, sigma)
    elif graph_type == 'connectivity':
        W = _build_knn_connectivity_graph(features, k)
    else:
        raise ValueError(f"Unknown graph_type: '{graph_type}'")
        
    D_inv = sp.diags(1.0 / (np.asarray(W.sum(axis=1)).flatten() + 1e-12))
    S = D_inv @ W

    F = Y_init.copy()
    for _ in range(max_iter):
        F_old = F.copy()
        F = alpha * (S @ F) + (1 - alpha) * Y_init
        if np.abs(F - F_old).sum() < 1e-6:
            break

    return F, classes

# +---------------------------------------------------------------------------------+
# |  Experiment Runner Class                                                        |
# +---------------------------------------------------------------------------------+

class ExperimentRunner:
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
    def run_raw_space_ssl(X_train_np, y_semi, X_test_np, y_test, k):
        print("\n--- Running Raw-Space SSL (Label Propagation) ---")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_np)
        X_test_scaled  = scaler.transform(X_test_np)
        lp = LabelPropagation(kernel="knn", n_neighbors=k, max_iter=1000, n_jobs=-1)
        lp.fit(X_train_scaled, y_semi)
        y_pred = lp.predict(X_test_scaled)
        test_acc = accuracy_score(y_test.numpy(), y_pred)
        print(f"  Raw‑Space Test Accuracy: {test_acc * 100:.2f}%")
        return test_acc

    @staticmethod
    def run_late_fusion_mv_grf(mf_train, rule_train, y_semi, mf_test, rule_test, y_test, k, *, sigma_mf, sigma_rule, beta, graph_type):
        print(f"\n--- Running Multi-View SSL (MV-GRF with Late Fusion, graph: {graph_type}) ---")
        # 1. Propagate on each view independently
        F_m, _ = get_grf_propagated_labels(mf_train, y_semi, k, sigma_mf, graph_type)
        F_r, classes = get_grf_propagated_labels(rule_train, y_semi, k, sigma_rule, graph_type)

        # 2. Fuse the soft label predictions
        F_final = beta * F_m + (1 - beta) * F_r
        y_hat = classes[F_final.argmax(1)]

        # 3. Use a k-NN to predict on the test set
        comb_train = np.hstack([mf_train, rule_train])
        comb_test  = np.hstack([mf_test,  rule_test])
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(comb_train, y_hat)
        y_pred = knn.predict(comb_test)
        
        test_acc = accuracy_score(y_test.numpy(), y_pred)
        print(f"  Late Fusion MV-GRF Test Accuracy: {test_acc * 100:.2f}%")
        return test_acc

    @staticmethod
    def run_late_fusion_fap(mf_train, rule_train, y_semi, mf_test, rule_test, y_test, k, *, sigma_mf, sigma_rule, alpha, max_iter, graph_type):
        print(f"\n--- Running Multi-View SSL (FAP with Late Fusion, graph: {graph_type}) ---")
        # 1. Propagate on each view independently
        F_m, _ = get_iterative_propagated_labels(mf_train, y_semi, k, sigma_mf, alpha, max_iter, graph_type)
        F_r, classes = get_iterative_propagated_labels(rule_train, y_semi, k, sigma_rule, alpha, max_iter, graph_type)

        # 2. Fuse with element-wise product (AND fusion) and re-normalize
        F_final = F_m * F_r
        F_final = F_final / (F_final.sum(axis=1, keepdims=True) + 1e-9)
        y_hat = classes[F_final.argmax(1)]

        # 3. Use a k-NN to predict on the test set
        comb_train = np.hstack([mf_train, rule_train])
        comb_test  = np.hstack([mf_test,  rule_test])
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(comb_train, y_hat)
        y_pred = knn.predict(comb_test)
        
        test_acc = accuracy_score(y_test.numpy(), y_pred)
        print(f"  Late Fusion FAP Test Accuracy: {test_acc * 100:.2f}%")
        return test_acc


# +---------------------------------------------------------------------------------+
# |  Main Execution Block                                                           |
# +---------------------------------------------------------------------------------+

if __name__ == '__main__':
    # ========================== HYPERPARAMETERS ============================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Global configuration: Using device: {device}")

    DATASET_LOADERS = [load_K_chess_data_splitted, load_htru_data, load_letter_data, load_pmd_data]
    LABEL_FRACTIONS = [0.1]
    SEEDS = [42]

    ANFIS_TRAIN_PARAMS = {'epochs': 150, 'lr_conseq': 0.01, 'lr_premise': 0.001, 'num_mfs': 3, 'max_rules': 100}

    # Grid now includes graph_type to test both methods
    SSL_GRID = [
        # RBF Graph experiments
        {'graph_type': 'rbf', 'k': 7,  'sigma_mf': 0.3, 'sigma_rule': 0.5, 'beta': 0.7, 'alpha': 0.99},
        {'graph_type': 'rbf', 'k': 15, 'sigma_mf': 0.5, 'sigma_rule': 0.7, 'beta': 0.5, 'alpha': 0.99},
        # Connectivity Graph experiments (sigma values are ignored but kept for consistent structure)
        {'graph_type': 'connectivity', 'k': 7,  'sigma_mf': 0, 'sigma_rule': 0, 'beta': 0.7, 'alpha': 0.99},
        {'graph_type': 'connectivity', 'k': 15, 'sigma_mf': 0, 'sigma_rule': 0, 'beta': 0.5, 'alpha': 0.99},
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

                # --- 2. Hybrid ANFIS Training & Feature Extraction ---
                print("--- Training HYBRID ANFIS & Extracting Views ---")
                anfis_model = HybridANFIS(
                    input_dim=X_train.shape[1], num_classes=len(y_train.unique()),
                    num_mfs=ANFIS_TRAIN_PARAMS['num_mfs'], max_rules=ANFIS_TRAIN_PARAMS['max_rules'], seed=seed
                ).to(device)
                
                train_anfis_hybrid(
                    anfis_model, X_l.to(device), y_l.to(device), X_train,
                    num_epochs=ANFIS_TRAIN_PARAMS['epochs'],
                    #lr_conseq=ANFIS_TRAIN_PARAMS['lr_conseq'],
                    lr=ANFIS_TRAIN_PARAMS['lr_premise']
                )

                anfis_model.eval()
                with torch.no_grad():
                    mf_values_train = anfis_model._fuzzify(X_train.to(device))
                    mf_values_test  = anfis_model._fuzzify(X_test.to(device))
                    _, rule_activations_train, _ = anfis_model(X_train.to(device))
                    _, rule_activations_test, _  = anfis_model(X_test.to(device))
                
                mf_train_np = F.normalize(mf_values_train.view(len(X_train), -1), p=2).cpu().numpy()
                mf_test_np  = F.normalize(mf_values_test.view(len(X_test), -1), p=2).cpu().numpy()
                rule_train_np = F.normalize(rule_activations_train, p=2).cpu().numpy()
                rule_test_np  = F.normalize(rule_activations_test, p=2).cpu().numpy()
                X_train_np = X_train.cpu().numpy()
                X_test_np = X_test.cpu().numpy()

                # --- 3. Run Experiments for each SSL Setting ---
                for ssl_params in SSL_GRID:
                    k = ssl_params['k']
                    graph_type = ssl_params['graph_type']
                    print(f"\n--- Running experiments for k={k}, graph_type='{graph_type}' ---")

                    acc_sup = runner.run_supervised_baseline(anfis_model, X_test, y_test, device)
                    
                    acc_raw = runner.run_raw_space_ssl(X_train_np, y_semi_sup, X_test_np, y_test, k=k)

                    acc_mv_grf_late = runner.run_late_fusion_mv_grf(
                        mf_train_np, rule_train_np, y_semi_sup, mf_test_np, rule_test_np, y_test,
                        k=k, sigma_mf=ssl_params['sigma_mf'], sigma_rule=ssl_params['sigma_rule'],
                        beta=ssl_params['beta'], graph_type=graph_type
                    )
                    
                    acc_fap_late = runner.run_late_fusion_fap(
                        mf_train_np, rule_train_np, y_semi_sup, mf_test_np, rule_test_np, y_test,
                        k=k, sigma_mf=ssl_params['sigma_mf'], sigma_rule=ssl_params['sigma_rule'],
                        alpha=ssl_params['alpha'], max_iter=100, graph_type=graph_type
                    )
                    
                    # --- 4. Log Results ---
                    results_rows.append({
                        'dataset': dataset_name, 'seed': seed, 'label_fraction': label_frac, 'k': k,
                        'graph_type': graph_type,
                        'sigma_mf': ssl_params['sigma_mf'], 'sigma_rule': ssl_params['sigma_rule'],
                        'beta': ssl_params['beta'],
                        'acc_supervised_hybrid': acc_sup,
                        'acc_raw_space_lp': acc_raw,
                        'acc_mv_grf_late': acc_mv_grf_late,
                        'acc_fap_late': acc_fap_late
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
