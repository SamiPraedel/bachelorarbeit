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
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from torch.utils.data import DataLoader, TensorDataset
from anfisHelper import initialize_mfs_with_kmeans
import itertools
import pandas as pd  # for CSV export
from trainAnfis import train_anfis_noHyb

# Global results list for all experiments/results
results_rows = []

class GraphSSL(BaseEstimator, ClassifierMixin):
    """
    A unified class for graph-based Semi-Supervised Learning in PyTorch.
    Supports three methods:
    1. 'iterative': k-NN graph with iterative label propagation.
    2. 'grf': k-NN graph with Gaussian Random Field analytical solution.
    3. 'bipartite': Sample-Rule graph with iterative propagation.
    """
    def __init__(self, k=10, sigma=1.0, max_iter=1000, tol=1e-6,
                 alpha=0.99, method='grf', distance_metric='euclidean',
                 device='cpu'):
        self.k = k
        self.sigma = sigma
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha # Teleportation probability for iterative methods
        self.method = method
        self.distance_metric = distance_metric  # store the chosen metric
        self.device = torch.device(device)
        
    def distance(self, X, metic='euclidean'):
        if metic == 'euclidean':
            X_norm_sq = torch.sum(X**2, dim=-1).unsqueeze(1)
            distances = X_norm_sq - 2 * (X @ X.T) + X_norm_sq.T
            
        elif metic == 'cosine':
            X_norm = F.normalize(X, p=2, dim=1)  # (n, d)
            sim = X_norm @ X_norm.T 
            distances = 1.0 - sim 
        elif metic == 'jaccard':
            eps = 1e-9
            # Broadcasting auf (n,1,d) und (1,n,d)
            Xi = X.unsqueeze(1)  # (n,1,d)
            Xj = X.unsqueeze(0)  # (1,n,d)

            # 1) Summiere die Minima / Maxima
            intersection = torch.min(Xi, Xj).sum(dim=2)   # (n,n)
            union        = torch.max(Xi, Xj).sum(dim=2)   # (n,n)

            # 2) Jaccard-Similarity und -Distanz
            jaccard_sim = intersection / (union + eps)
            distances   = 1.0 - jaccard_sim    
        elif metic == 'manhattan':
            distances = torch.cdist(X, X, p=1)
            
        
        distances.clamp_(min=0)
        return distances


    def _build_knn_graph(self, X):
        """Builds the k-NN graph and the affinity matrix W."""
        distances_sq = self.distance(X, metic=self.distance_metric)
        _, indices = torch.topk(distances_sq, k=self.k + 1, dim=1, largest=False)
        neighbors_idx = indices[:, 1:]
        n_samples = X.shape[0]
        row_indices = torch.arange(n_samples, device=self.device).unsqueeze(1).expand(-1, self.k)
        neighbor_dist_sq = torch.gather(distances_sq, 1, neighbors_idx)
        weights = torch.exp(-neighbor_dist_sq / (2 * self.sigma**2))
        W = torch.zeros(n_samples, n_samples, device=self.device)
        W[row_indices.cpu(), neighbors_idx] = weights
        return (W + W.T) / 2
    
    
    def fit_mf_space(self, X, y):
        """
        Fits using GRF on the flattened MF value space.
        X is expected to be of shape [N, D, M] (samples, dims, mfs).
        """
        # Flatten the MF values to create the feature space
        X_flat = X.reshape(X.shape[0], -1)
        return self.fit_grf(X_flat, y)

    
    
    
    def _prepare_fit(self, X, y):
        """Shared setup for all fit methods."""
        if not isinstance(X, torch.Tensor): X = torch.from_numpy(X).float()
        if not isinstance(y, torch.Tensor): y = torch.from_numpy(y).long()
        
        X = X.to(self.device)
        y = y.to(self.device)
        
        #X_cpu = X.cpu()
        #y_cpu = y.cpu()
        
        y_cpu = y.cpu().numpy()
        
        #X, y = X.to(self.device), y.to(self.device)

        
        self.classes_ = np.unique(y_cpu[y_cpu != -1])
        self.n_classes_ = len(self.classes_)
        self.X_train_ = X.to(self.device)
        
        n_samples = X.shape[0]
        Y_ = torch.zeros(n_samples, self.n_classes_, device=self.device)
        for i, cls in enumerate(self.classes_):
            Y_[y_cpu == cls, i] = 1
        labeled_mask = (y != -1)
        return X, y, Y_, labeled_mask

    # def fit_iterative(self, X, y):
    #     """Fits using iterative Label Propagation on a k-NN graph."""
    #     X, y, Y_init, labeled_mask = self._prepare_fit(X, y)
    #     #W = self._build_knn_graph(X)
    #     W = self.build_sparse_knn_graph_sklearn(X)
    #     D = torch.diag(W.sum(axis=1))
    #     D_inv_sqrt = torch.diag(1.0 / (torch.sqrt(torch.diag(D)) + 1e-9))
    #     S = D_inv_sqrt @ W @ D_inv_sqrt
    #     F = Y_init.clone()
    #     for i in range(self.max_iter):
    #         F_old = F.clone()
    #         # Update step with teleportation
    #         F = self.alpha * (S @ F) + (1 - self.alpha) * Y_init
    #         if torch.abs(F - F_old).sum() < self.tol: break
    #     self.label_distributions_ = F
    #     self.transduction_ = self.classes_[torch.argmax(self.label_distributions_, axis=1).cpu().numpy()]
    #     return self
    
    
    def fit_iterative(self, X, y):
        """Fits using iterative Label Propagation on a sparse k-NN graph."""
        X_cpu, y_cpu, Y_init, labeled_mask = self._prepare_fit(X, y)
        
       # W_sparse = self._build_knn_graph_sparse(X_cpu)
        W_sparse = self._build_knn_graph(X_cpu)
        # Sparse matrix operations
        D_vals = torch.sparse.sum(W_sparse, dim=1).to_dense()
        D_inv_sqrt_vals = 1.0 / (torch.sqrt(D_vals) + 1e-9)
        
        # Normalization using element-wise multiplication with sparse tensor
        S_sparse = W_sparse.multiply(D_inv_sqrt_vals.view(-1, 1)).multiply(D_inv_sqrt_vals.view(1, -1))

        F = Y_init.clone()
        for i in range(self.max_iter):
            F_old = F.clone()
            # Use sparse matrix multiplication
            propagated_F = torch.sparse.mm(S_sparse, F)
            F = self.alpha * propagated_F + (1 - self.alpha) * Y_init
            if torch.abs(F - F_old).sum() < self.tol: break
            
        self.label_distributions_ = F
        self.transduction_ = self.classes_[torch.argmax(self.label_distributions_, axis=1).cpu().numpy()]
        return self
    
    def fit_iterative2(self, X, y,
                  k: int = 15,
                  alpha: float | None = None,
                  n_iter: int = 40):
        """
        Iterative label propagation  (Zhou et al. 2004).

        Parameters
        ----------
        X : torch.Tensor or np.ndarray –  [N, d]  Sample embeddings
        y : torch.LongTensor / np.ndarray
            True labels (-1 für unlabeled).
        k : int           –  k-NN für den Graph (default 15)
        alpha : float|None –  Damp-Faktor.
            None ➜ automatisch:  alpha = 0.99  bei N>5 000  sonst 0.9
        n_iter : int      –  feste #Iterationen  (kein early stop)
        """
        # --- Prepare inputs and save training data ---
        if not isinstance(X, torch.Tensor): X = torch.from_numpy(X).float()
        if not isinstance(y, torch.Tensor): y = torch.from_numpy(y).long()
        self.X_train_ = X.to(self.device)

        # Graph building on CPU is often faster for sklearn/scipy
        X_cpu = X.detach().to("cpu")
        y_np = y.cpu().numpy().astype(np.int64)
        N = len(X_cpu)
        self.classes_  = np.unique(y_np[y_np >= 0])
        C             = len(self.classes_)
        self.n_classes_ = C

        # Y_init  (one-hot für Labeled, sonst 0)
        Y_init = torch.zeros(N, C)
        if (y_np >= 0).any():
            Y_init[torch.arange(N)[y_np >= 0],
                np.searchsorted(self.classes_, y_np[y_np >= 0])] = 1.

        # --- 1) k-NN-Graph  (symmetrisch) 
        
        nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(X_cpu)
        knn_idx = nn.kneighbors(return_distance=False)

        row_ind = np.repeat(np.arange(N), k)
        col_ind = knn_idx.reshape(-1)
        data    = np.ones(len(row_ind), dtype=np.float32)

        # Symmetrisieren:  A = A ∪ Aᵀ
        import scipy.sparse as sp
        W = sp.coo_matrix((data, (row_ind, col_ind)), shape=(N, N))
        W = ((W + W.T) > 0).astype(np.float32).tocoo()

        # --- 2) Row-stochastic Übergangs­matrix P 
        deg = np.asarray(W.sum(1)).flatten() + 1e-12          # avoid zero
        deg_inv = 1.0 / deg
        W.data *= deg_inv[W.row]                              # in-place row-norm
        P = torch.sparse_coo_tensor(
                indices=torch.vstack((torch.from_numpy(W.row),
                                    torch.from_numpy(W.col))),
                values = torch.from_numpy(W.data),
                size   = (N, N)).coalesce().to(self.device)

        # --- 3) Iteration 
        if alpha is None:
            alpha = 0.99 if N > 5_000 else 0.9

        F = Y_init.to(self.device)
        Y_init = Y_init.to(self.device)

        for _ in range(n_iter):
            F = alpha * torch.sparse.mm(P, F) + (1-alpha) * Y_init

        self.label_distributions_ = F.cpu()
        self.transduction_ = self.classes_[F.argmax(1).cpu().numpy()]
        return self


    def fit_grf(self, X, y):
        """Fits using the Gaussian Random Field analytical solution on a k-NN graph."""
        X, y, Y_, labeled_mask = self._prepare_fit(X, y)
        unlabeled_mask = ~labeled_mask
        W = self._build_knn_graph(X)
        D = torch.diag(W.sum(axis=1))
        L = D - W
        L_uu = L[unlabeled_mask, :][:, unlabeled_mask]
        L_ul = L[unlabeled_mask, :][:, labeled_mask]
        f_l = Y_[labeled_mask]
        #b = -L_ul @ f_l   
        b = (-L_ul @ f_l).to(dtype=L_uu.dtype, device=L_uu.device)   # [U,C]
        n_u = L_uu.shape[0]
        eps  = 1e-4            # a bit larger for safety
        I_U  = torch.eye(n_u, dtype=L_uu.dtype, device=L_uu.device)
        A    = L_uu + eps * I_U     # [U,U]
        f_u = torch.linalg.solve(A, b)
        #f_u = torch.linalg.solve(L_uu, b)
        self.label_distributions_ = torch.zeros_like(Y_)
        self.label_distributions_[labeled_mask] = f_l
        self.label_distributions_[unlabeled_mask] = F.softmax(f_u, dim=1)
        self.transduction_ = self.classes_[self.label_distributions_.argmax(1).cpu().numpy()]
        return self
    
    def _build_knn_graph_sparse(self, X):

        n_samples = X.shape[0]

        W_scipy = kneighbors_graph(X.cpu().numpy(), self.k, mode='distance', include_self=False)
        
        # Apply the Gaussian (RBF) kernel to the distances
        W_scipy.data = np.exp(-W_scipy.data**2 / (2 * self.sigma**2))

        # Make the matrix symmetric
        W_scipy = W_scipy.maximum(W_scipy.T)
        
        # Convert the scipy sparse matrix to a PyTorch sparse COO tensor
        coo = W_scipy.tocoo()
        indices = torch.from_numpy(np.vstack((coo.row, coo.col))).long()
        values = torch.from_numpy(coo.data).float()
        
        return torch.sparse_coo_tensor(indices, values, coo.shape, device=self.device).coalesce()

    def fit(self, X, y):
        """Main fit method that dispatches to the correct algorithm."""
        if self.method == 'grf':
            return self.fit_grf(X, y)
        elif self.method == 'iterative':
            return self.fit_iterative(X, y)
        elif self.method == 'mf_space':
            return self.fit_mf_space(X, y)

    def predict(self, X):
        """Predicts labels for new data points using a k-NN approach."""
        check_is_fitted(self)
        if not isinstance(X, torch.Tensor): X = torch.from_numpy(X).float()
        X = X.to(self.device)
        
        # If the model was trained on MF space, flatten the input X
        if self.method == 'mf_space':
            X = X.reshape(X.shape[0], -1)

        X_norm_sq = torch.sum(X**2, dim=-1).unsqueeze(1)
        X_train_norm_sq = torch.sum(self.X_train_**2, dim=-1).unsqueeze(1)
        distances_sq = (X_norm_sq - 2 * (X @ self.X_train_.T) + X_train_norm_sq.T).clamp(min=0)
        _, neighbor_indices = torch.topk(distances_sq, k=self.k, dim=1, largest=False)
        # FIX: neighbor_indices must be on the same device as label_distributions_
        neighbor_distributions = self.label_distributions_[neighbor_indices]
        y_pred_dist = neighbor_distributions.sum(axis=1)
        return self.classes_[torch.argmax(y_pred_dist, axis=1).cpu().numpy()]



# --- NEW: FMV-CLP Algorithm and Helpers ---
import scipy.sparse as sp
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
        
        # Get indices of currently unlabeled points
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
            print(f"    [FMV-CLP Round {outer_iter+1}] No new seeds found. Stopping.")
            break

        # 5f add new seeds
        num_added = 0
        for idx, c in new_seeds:
            if not current_labeled_mask[idx]:
                Y0[idx, :] = 0.
                Y0[idx, class_map[classes[c]]] = 1.
                current_labeled_mask[idx] = True
                num_added += 1
        
        print(f"    [FMV-CLP Round {outer_iter+1}] Added {num_added} new pseudo-labels.")
        if num_added == 0: break


    # 6) final prediction
    y_hat_idx = F_comb.argmax(1).cpu().numpy()
    idx2cls = {v: k for k, v in class_map.items()}
    y_hat = np.vectorize(idx2cls.get)(y_hat_idx)

    return y_hat, F_comb

def _build_knn_rbf(X, k, sigma):
    A = kneighbors_graph(X, n_neighbors=k, mode="distance", include_self=False, n_jobs=-1)
    d2 = np.square(A.data)
    A.data = np.exp(-d2 / (2.0 * sigma ** 2))
    A_sym = A.maximum(A.T)
    return A_sym

def mv_grf_predict(
        M, R,
        y_init,
        k=10,
        sigma_M=1.0, sigma_R=1.0,
        beta=0.5,
        reg_eps=1e-5,
        device="cpu"):
    
    if isinstance(M, torch.Tensor): M = M.cpu().numpy()
    if isinstance(R, torch.Tensor): R = R.cpu().numpy()
    if isinstance(y_init, torch.Tensor): y_init = y_init.cpu().numpy()

    n = M.shape[0]
    classes = np.unique(y_init[y_init >= 0])
    C = len(classes)
    class_map = {c: i for i, c in enumerate(classes)}

    W_M = _build_knn_rbf(M, k, sigma_M)
    W_R = _build_knn_rbf(R, k, sigma_R)

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

    # Solve (L_uu + εI) f_u = - L_ul f_l
    I_uu = sp.identity(L_uu.shape[0], format='csr') * reg_eps
    A = L_uu + I_uu
    b = -L_ul @ f_l.cpu().numpy()

    f_u_list = [sp.linalg.cg(A, b[:, c])[0] for c in range(C)]
    f_u = torch.from_numpy(np.vstack(f_u_list).T).float().to(device)

    F_final = torch.zeros(n, C, device=device)
    F_final[idx_L] = f_l
    F_final[idx_U] = F.softmax(f_u, dim=1)

    y_hat_idx = F_final.argmax(dim=1).cpu().numpy()
    idx2cls = {v: k for k, v in class_map.items()}
    y_hat = np.vectorize(idx2cls.get)(y_hat_idx)

    return y_hat, F_final.cpu()

def feature_extraction(model, X, epochs=100):
    if isinstance(model, NoHybridANFIS):
        # For NoHybridANFIS, we can directly use the forward pass
        
        with torch.no_grad():
            _, rule_activations, _ = model(X)
        return rule_activations.cpu().numpy()
    
    
    
    
    
    
    
    

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    X_train, y_train, X_test, y_test = load_K_chess_data_splitted()
    #X_train, y_train, X_test, y_test = load_htru_data()
    #X_train, y_train, X_test, y_test = load_letter_data()
    
    
    
    n_labeled = int(0.1 * len(y_train))
    np.random.seed(42)
    labeled_indices = np.random.choice(np.arange(len(y_train)), size=n_labeled, replace=False)
    y_semi_sup = np.full(len(y_train), -1, dtype=np.int64)
    y_semi_sup[labeled_indices] = y_train[labeled_indices]
    
    X_l, y_l = X_train[labeled_indices], y_train[labeled_indices]
    
   
    
    #  Experiment 1: NoHybridANFIS ---
    anfis_model = NoHybridANFIS(
        input_dim=X_train.shape[1],
        num_classes=len(y_train.unique()), num_mfs=4, max_rules=100,
        seed=42
    ).to(device)
    
    anfis_model_train = NoHybridANFIS(
        input_dim=X_train.shape[1],
        num_classes=len(y_train.unique()), num_mfs=4, max_rules=100,
        seed=42
    ).to(device)
    
    initialize_mfs_with_kmeans(anfis_model_train, X_train)
    
    train_anfis_noHyb(anfis_model, X_l.to(device), y_l.to(device), X_train, num_epochs=400, lr=0.01)

    # Supervised Baseline Test 
    print("\n--- Running Supervised Baseline (ANFIS trained on 10% data) ---")
    anfis_model.eval()
    with torch.no_grad():
        logits_test, _, _ = anfis_model(X_test.to(device))
        preds_test = logits_test.argmax(1).cpu()
        baseline_acc = accuracy_score(y_test.numpy(), preds_test.numpy())
    print(f"  Final Test Accuracy (Supervised Baseline): {baseline_acc * 100:.2f}%")
    
    
    
    
    
    print("\n Supervised Random Forest")
    rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    # scikit-learn models work with numpy arrays on the CPU
    rf_baseline.fit(X_l.cpu().numpy(), y_l.cpu().numpy())
    y_pred_rf = rf_baseline.predict(X_test.cpu().numpy())
    rf_acc = accuracy_score(y_test.numpy(), y_pred_rf)
    print(f"  Final Test Accuracy (Supervised RF Baseline): {rf_acc * 100:.2f}%")
    
    
    
    
    models = [anfis_model]
    # --- END NEW PART ---
    
    for anfis_model in models:
        
        anfis_model.eval()
        with torch.no_grad():
            _, rule_activations_train, _ = anfis_model(X_train.to(device))
            _, rule_activations_test, _ = anfis_model(X_test.to(device))
            mf_values_train = anfis_model._fuzzify(X_train.to(device))
            mf_values_test = anfis_model._fuzzify(X_test.to(device))
        

        

        # --- Test Rule-Space SSL ---
        print(" Running SSL with: Rule-Space GRF ---")
        ssl_rule_model = GraphSSL(k=15, sigma=0.3, method='grf', device=device)
        ssl_rule_model.fit(rule_activations_train, y_semi_sup)
        unlabeled_mask = (y_semi_sup == -1)
        pseudo_label_acc = accuracy_score(y_train.numpy()[unlabeled_mask], ssl_rule_model.transduction_[unlabeled_mask])
        print(f"  Pseudo-Label Accuracy: {pseudo_label_acc * 100:.2f}%")
        y_pred = ssl_rule_model.predict(rule_activations_test)
        test_acc = accuracy_score(y_test.numpy(), y_pred)
        print(f"  Final Test Accuracy (Rule-Space): {test_acc * 100:.2f}%")
        
        # --- Test MF-Space SSL ---
        print(" Running SSL with: MF-Space GRF ---")
        ssl_mf_model = GraphSSL(k=15, sigma=0.3, method='mf_space', device=device)
        ssl_mf_model.fit(mf_values_train, y_semi_sup)
        pseudo_label_acc_mf = accuracy_score(y_train.numpy()[unlabeled_mask], ssl_mf_model.transduction_[unlabeled_mask])
        print(f"  Pseudo-Label Accuracy: {pseudo_label_acc_mf * 100:.2f}%")
        y_pred_mf = ssl_mf_model.predict(mf_values_test)
        test_acc_mf = accuracy_score(y_test.numpy(), y_pred_mf)
        print(f"  Final Test Accuracy (MF-Space): {test_acc_mf * 100:.2f}%")
        
        
        
        rule_activations_train_np = rule_activations_train.cpu().numpy()
        rule_activations_test_np = rule_activations_test.cpu().numpy()
        mf_values_train_flat_np = mf_values_train.reshape(mf_values_train.shape[0], -1).cpu().numpy()
        mf_values_test_flat_np = mf_values_test.reshape(mf_values_test.shape[0], -1).cpu().numpy()
        X_train_np = X_train.cpu().numpy()
        X_test_np = X_test.cpu().numpy()
        
        
        # --- Test Rule-Space SSL with sklearn ---
        print("\n--- Running SSL with: Rule-Space LabelPropagation (sklearn) ---")
        lp_rule_model = LabelPropagation(kernel='knn', n_neighbors=15)
        lp_rule_model.fit(rule_activations_train_np, y_semi_sup)
        pseudo_label_acc = accuracy_score(y_train.numpy()[unlabeled_mask], lp_rule_model.transduction_[unlabeled_mask])
        print(f"  Pseudo-Label Accuracy: {pseudo_label_acc * 100:.2f}%")
        y_pred = lp_rule_model.predict(rule_activations_test_np)
        test_acc = accuracy_score(y_test.numpy(), y_pred)
        print(f"  Final Test Accuracy (Rule-Space): {test_acc * 100:.2f}%")
        
        # --- Test MF-Space SSL with sklearn ---
        print("\n--- Running SSL with: MF-Space LabelPropagation (sklearn) ---")
        lp_mf_model = LabelPropagation(kernel='knn', n_neighbors=10)
        lp_mf_model.fit(mf_values_train_flat_np, y_semi_sup)
        pseudo_label_acc_mf = accuracy_score(y_train.numpy()[unlabeled_mask], lp_mf_model.transduction_[unlabeled_mask])
        print(f"  Pseudo-Label Accuracy: {pseudo_label_acc_mf * 100:.2f}%")
        y_pred_mf = lp_mf_model.predict(mf_values_test_flat_np)
        test_acc_mf = accuracy_score(y_test.numpy(), y_pred_mf)
        print(f"  Final Test Accuracy (MF-Space): {test_acc_mf * 100:.2f}%")


        # --- Raw Data SSL Baseline with sklearn ---
        print("\n--- Running SSL with: Raw-Space LabelPropagation (sklearn) ---")
        lp_raw_model = LabelPropagation(kernel='knn', n_neighbors=15)
        lp_raw_model.fit(X_train_np, y_semi_sup)
        pseudo_label_acc_raw = accuracy_score(y_train.numpy()[unlabeled_mask], lp_raw_model.transduction_[unlabeled_mask])
        print(f"  Pseudo-Label Accuracy: {pseudo_label_acc_raw * 100:.2f}%")
        y_pred_raw = lp_raw_model.predict(X_test_np)
        test_acc_raw = accuracy_score(y_test.numpy(), y_pred_raw)
        print(f"  Final Test Accuracy (Raw-Space): {test_acc_raw * 100:.2f}%")

        # --- FMV‑CLP (MF + Rule Views) ---
        print("\n--- Running SSL with: FMV‑CLP (MF + Rule) ---")
        combined_train = np.hstack([mf_values_train_flat_np, rule_activations_train_np])
        combined_test  = np.hstack([mf_values_test_flat_np,  rule_activations_test_np])
        y_hat_fmv, _ = fmv_clp(M=mf_values_train_flat_np,
                               R=rule_activations_train_np,
                               y_init=y_semi_sup,
                               k=15, sigma_M=0.3, sigma_R=0.3,
                               beta=0.5)
        knn = KNeighborsClassifier(n_neighbors=15)
        knn.fit(combined_train, y_hat_fmv)
        y_pred_fmv = knn.predict(combined_test)
        acc_fmv = accuracy_score(y_test.numpy(), y_pred_fmv)
        print(f"  Final Test Accuracy (FMV-CLP): {acc_fmv * 100:.2f}%")

        # --- MV‑GRF (MF + Rule Views) ---
        print("\n--- Running SSL with: MV‑GRF (MF + Rule) ---")
        y_hat_mv, _ = mv_grf_predict(M=mf_values_train_flat_np,
                                     R=rule_activations_train_np,
                                     y_init=y_semi_sup,
                                     k=15, sigma_M=0.3, sigma_R=0.3,
                                     beta=0.5)
        knn = KNeighborsClassifier(n_neighbors=15)
        knn.fit(combined_train, y_hat_mv)
        y_pred_mv = knn.predict(combined_test)
        acc_mv = accuracy_score(y_test.numpy(), y_pred_mv)
        print(f"  Final Test Accuracy (MV-GRF): {acc_mv * 100:.2f}%")

    
    """device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    #X_train, y_train, X_test, y_test = load_K_chess_data_splitted()
    X_train, y_train, X_test, y_test = load_htru_data()
    #X_train, y_train, X_test, y_test = load_letter_data()
    
    
    
    n_labeled = int(0.1 * len(y_train))
    np.random.seed(42)
    labeled_indices = np.random.choice(np.arange(len(y_train)), size=n_labeled, replace=False)
    y_semi_sup = np.full(len(y_train), -1, dtype=np.int64)
    y_semi_sup[labeled_indices] = y_train[labeled_indices]
    
    X_l, y_l = X_train[labeled_indices], y_train[labeled_indices]
    
   
   
   
   
   
   
   
   
    
    # --- Experiment 1: NoHybridANFIS ---
    print("\n\n--- EXPERIMENT 1: Using NoHybridANFIS as Feature Extractor ---")
    anfis_model = NoHybridANFIS(
        input_dim=X_train.shape[1],
        num_classes=len(y_train.unique()), num_mfs=4, max_rules=100,
        seed=42
    ).to(device)
    
    #initialize_mfs_with_kmeans(anfis_model, X_l)
    
    optimizer = torch.optim.Adam(anfis_model.parameters(), lr=0.01)
    for epoch in range(100):
        anfis_model.train()
        optimizer.zero_grad()
        logits, _, _ = anfis_model(X_l.to(device))
        loss = F.cross_entropy(logits, y_l.to(device))
        loss.backward()
        optimizer.step()

    # --- NEW: Supervised Baseline Test ---
    print("\n--- Running Supervised Baseline (ANFIS trained on 10% data) ---")
    anfis_model.eval()
    with torch.no_grad():
        logits_test, _, _ = anfis_model(X_test.to(device))
        preds_test = logits_test.argmax(1).cpu()
        baseline_acc = accuracy_score(y_test.numpy(), preds_test.numpy())
    print(f"  Final Test Accuracy (Supervised Baseline): {baseline_acc * 100:.2f}%")
    # --- END NEW PART ---

    anfis_model.eval()
    with torch.no_grad():
        _, rule_activations_train, _ = anfis_model(X_train.to(device))
        _, rule_activations_test, _ = anfis_model(X_test.to(device))
        mf_values_train = anfis_model._fuzzify(X_train.to(device))
        mf_values_test = anfis_model._fuzzify(X_test.to(device))

    print(f"Shape of ANFIS rule space: {rule_activations_train.shape}")
    print(f"Shape of ANFIS MF space: {mf_values_train.shape}")
    
    print("\n\n--- EXPERIMENT 3: Supervised Random Forest (trained on 50% data) ---")
    rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    # scikit-learn models work with numpy arrays on the CPU
    rf_baseline.fit(X_l.cpu().numpy(), y_l.cpu().numpy())
    y_pred_rf = rf_baseline.predict(X_test.cpu().numpy())
    rf_acc = accuracy_score(y_test.numpy(), y_pred_rf)
    print(f"  Final Test Accuracy (Supervised RF Baseline): {rf_acc * 100:.2f}%")
    
    

    # --- Test Rule-Space SSL ---
    print("\n--- Running SSL with: Rule-Space GRF ---")
    ssl_rule_model = GraphSSL(k=15, sigma=0.3, method='grf', device=device)
    ssl_rule_model.fit(rule_activations_train, y_semi_sup)
    unlabeled_mask = (y_semi_sup == -1)
    pseudo_label_acc = accuracy_score(y_train.numpy()[unlabeled_mask], ssl_rule_model.transduction_[unlabeled_mask])
    print(f"  Pseudo-Label Accuracy: {pseudo_label_acc * 100:.2f}%")
    y_pred = ssl_rule_model.predict(rule_activations_test)
    test_acc = accuracy_score(y_test.numpy(), y_pred)
    print(f"  Final Test Accuracy (Rule-Space): {test_acc * 100:.2f}%")
    
    # --- Test MF-Space SSL ---
    print("\n--- Running SSL with: MF-Space GRF ---")
    ssl_mf_model = GraphSSL(k=15, sigma=0.3, method='mf_space', device=device)
    ssl_mf_model.fit(mf_values_train, y_semi_sup)
    pseudo_label_acc_mf = accuracy_score(y_train.numpy()[unlabeled_mask], ssl_mf_model.transduction_[unlabeled_mask])
    print(f"  Pseudo-Label Accuracy: {pseudo_label_acc_mf * 100:.2f}%")
    y_pred_mf = ssl_mf_model.predict(mf_values_test)
    test_acc_mf = accuracy_score(y_test.numpy(), y_pred_mf)
    print(f"  Final Test Accuracy (MF-Space): {test_acc_mf * 100:.2f}%")
    
    
    
    
    rule_activations_train_np = rule_activations_train.cpu().numpy()
    rule_activations_test_np = rule_activations_test.cpu().numpy()
    mf_values_train_flat_np = mf_values_train.reshape(mf_values_train.shape[0], -1).cpu().numpy()
    mf_values_test_flat_np = mf_values_test.reshape(mf_values_test.shape[0], -1).cpu().numpy()
    X_train_np = X_train.cpu().numpy()
    X_test_np = X_test.cpu().numpy()
    
    
    # --- Test Rule-Space SSL with sklearn ---
    print("\n--- Running SSL with: Rule-Space LabelPropagation (sklearn) ---hdjnksajhjhdshgfkj")
    lp_rule_model = LabelPropagation(kernel='knn', n_neighbors=15)
    lp_rule_model.fit(rule_activations_train_np, y_semi_sup)
    pseudo_label_acc = accuracy_score(y_train.numpy()[unlabeled_mask], lp_rule_model.transduction_[unlabeled_mask])
    print(f"  Pseudo-Label Accuracy: {pseudo_label_acc * 100:.2f}%")
    y_pred = lp_rule_model.predict(rule_activations_test_np)
    test_acc = accuracy_score(y_test.numpy(), y_pred)
    print(f"  Final Test Accuracy (Rule-Space): {test_acc * 100:.2f}%")
    
    # --- Test MF-Space SSL with sklearn ---
    print("\n--- Running SSL with: MF-Space LabelPropagation (sklearn) ---")
    lp_mf_model = LabelPropagation(kernel='knn', n_neighbors=10)
    lp_mf_model.fit(mf_values_train_flat_np, y_semi_sup)
    pseudo_label_acc_mf = accuracy_score(y_train.numpy()[unlabeled_mask], lp_mf_model.transduction_[unlabeled_mask])
    print(f"  Pseudo-Label Accuracy: {pseudo_label_acc_mf * 100:.2f}%")
    y_pred_mf = lp_mf_model.predict(mf_values_test_flat_np)
    test_acc_mf = accuracy_score(y_test.numpy(), y_pred_mf)
    print(f"  Final Test Accuracy (MF-Space): {test_acc_mf * 100:.2f}%")

    # --- Raw Data SSL Baseline with sklearn ---
    print("\n--- Running SSL with: Raw-Space LabelPropagation (sklearn) ---")
    lp_raw_model = LabelPropagation(kernel='knn', n_neighbors=15)
    lp_raw_model.fit(X_train_np, y_semi_sup)
    pseudo_label_acc_raw = accuracy_score(y_train.numpy()[unlabeled_mask], lp_raw_model.transduction_[unlabeled_mask])
    print(f"  Pseudo-Label Accuracy: {pseudo_label_acc_raw * 100:.2f}%")
    y_pred_raw = lp_raw_model.predict(X_test_np)
    test_acc_raw = accuracy_score(y_test.numpy(), y_pred_raw)
    print(f"  Final Test Accuracy (Raw-Space): {test_acc_raw * 100:.2f}%")"""