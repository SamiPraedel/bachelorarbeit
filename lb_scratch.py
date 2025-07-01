import torch
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data_utils    import load_K_chess_data_splitted, load_htru_data, load_pmd_data
from anfis_nonHyb import NoHybridANFIS
import torch.nn.functional as F
from anfisHelper import initialize_mfs_with_kmeans
from PopFnn import POPFNN
from kmFmmc import FMNC
from torch import nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import metrics




class GraphSSL(BaseEstimator, ClassifierMixin):
    """
    A unified class for graph-based Semi-Supervised Learning in PyTorch.
    Supports three methods:
    1. 'iterative': k-NN graph with iterative label propagation.
    2. 'grf': k-NN graph with Gaussian Random Field analytical solution.
    3. 'bipartite': Sample-Rule graph with iterative propagation.
    """
    def __init__(self, k=10, sigma=1.0, max_iter=1000, tol=1e-6, 
                 alpha=0.99, method='grf', device='cpu'):
        self.k = k
        self.sigma = sigma
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha # Teleportation probability for iterative methods
        self.method = method
        self.device = torch.device(device)

    def _build_knn_graph(self, X):
        """Builds the k-NN graph and the affinity matrix W."""
        X_norm_sq = torch.sum(X**2, dim=-1).unsqueeze(1)
        distances_sq = X_norm_sq - 2 * (X @ X.T) + X_norm_sq.T
        distances_sq.clamp_(min=0)
        _, indices = torch.topk(distances_sq, k=self.k + 1, dim=1, largest=False)
        neighbors_idx = indices[:, 1:]
        n_samples = X.shape[0]
        row_indices = torch.arange(n_samples, device=self.device).unsqueeze(1).expand(-1, self.k)
        neighbor_dist_sq = torch.gather(distances_sq, 1, neighbors_idx)
        weights = torch.exp(-neighbor_dist_sq / (2 * self.sigma**2))
        W = torch.zeros(n_samples, n_samples, device=self.device)
        W[row_indices, neighbors_idx] = weights
        return (W + W.T) / 2

    def _prepare_fit(self, X, y):
        """Shared setup for all fit methods."""
        if not isinstance(X, torch.Tensor): X = torch.from_numpy(X).float()
        if not isinstance(y, torch.Tensor): y = torch.from_numpy(y).long()
        X, y = X.to(self.device), y.to(self.device)
        y_np = y.cpu().numpy()
        self.classes_ = np.unique(y_np[y_np != -1])
        self.n_classes_ = len(self.classes_)
        self.X_train_ = X
        n_samples = X.shape[0]
        Y_ = torch.zeros(n_samples, self.n_classes_, device=self.device)
        for i, cls in enumerate(self.classes_):
            Y_[y == cls, i] = 1
        labeled_mask = (y != -1)
        return X, y, Y_, labeled_mask

    def fit_iterative(self, X, y):
        """Fits using iterative Label Propagation on a k-NN graph."""
        X, y, Y_init, labeled_mask = self._prepare_fit(X, y)
        W = self._build_knn_graph(X)
        D = torch.diag(W.sum(axis=1))
        D_inv_sqrt = torch.diag(1.0 / (torch.sqrt(torch.diag(D)) + 1e-9))
        S = D_inv_sqrt @ W @ D_inv_sqrt
        F = Y_init.clone()
        for i in range(self.max_iter):
            F_old = F.clone()
            # Update step with teleportation
            F = self.alpha * (S @ F) + (1 - self.alpha) * Y_init
            if torch.abs(F - F_old).sum() < self.tol: break
        self.label_distributions_ = F
        self.transduction_ = self.classes_[torch.argmax(self.label_distributions_, axis=1).cpu().numpy()]
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
        b = -L_ul @ f_l
        f_u = torch.linalg.solve(L_uu, b)
        self.label_distributions_ = torch.zeros_like(Y_)
        self.label_distributions_[labeled_mask] = f_l
        self.label_distributions_[unlabeled_mask] = F.softmax(f_u, dim=1)
        self.transduction_ = self.classes_[self.label_distributions_.argmax(1).cpu().numpy()]
        return self

    def fit_bipartite(self, X_rules, y):
        device = self.device
        X = X_rules.to(device).float()
        y = torch.as_tensor(y, device=device, dtype=torch.long)
        self.classes_ = torch.unique(y[y!=-1]).cpu().numpy()
        C = self.n_classes_ = len(self.classes_)
        y_remap = y.clone()
        for i,c in enumerate(self.classes_): y_remap[y==c] = i
        self.X_train_ = X
        W = F.normalize(X, p=1, dim=1)
        N,R = W.shape
        rows = torch.arange(N, device=device).repeat_interleave(R)
        cols = torch.arange(R, device=device).repeat(N)
        vals = W.flatten()
        A_idx = torch.cat([torch.stack([rows, cols+N]), torch.stack([cols+N, rows])], dim=1)
        A = torch.sparse_coo_tensor(A_idx, torch.cat([vals, vals]), size=(N+R, N+R)).coalesce()
        deg_inv = 1.0 / torch.sparse.sum(A, dim=1).to_dense().clamp(min=1e-12)
        A_rw = torch.sparse_coo_tensor(A.indices(), A.values()*deg_inv[A.indices()[0]], A.size())
        Y = torch.zeros(N+R, C, device=device)
        Y[y_remap != -1, y_remap[y_remap != -1]] = 1.
        Y0 = Y.clone()
        for _ in range(self.max_iter):
            Y_new = self.alpha * torch.sparse.mm(A_rw, Y) + (1-self.alpha)*Y0
            if torch.norm(Y_new-Y, p=1) < self.tol: break
            Y = Y_new
        self.label_distributions_ = Y[:N]
        self.transduction_ = self.classes_[self.label_distributions_.argmax(1).cpu().numpy()]
        return self

    def fit(self, X, y):
        """Main fit method that dispatches to the correct algorithm."""
        if self.method == 'grf':
            return self.fit_grf(X, y)
        elif self.method == 'iterative':
            return self.fit_iterative(X, y)
        elif self.method == 'bipartite':
            return self.fit_bipartite(X, y)
        else:
            raise ValueError("Method must be 'grf', 'iterative', or 'bipartite'")

    def predict(self, X):
        """Predicts labels for new data points using a k-NN approach."""
        check_is_fitted(self)
        if not isinstance(X, torch.Tensor): X = torch.from_numpy(X).float()
        X = X.to(self.device)
        
        # This k-NN prediction logic is now used for all methods.
        # It finds the k-nearest neighbors for the test points within the
        # training data's feature space (self.X_train_) and aggregates their
        # final label distributions.
        X_norm_sq = torch.sum(X**2, dim=-1).unsqueeze(1)
        X_train_norm_sq = torch.sum(self.X_train_**2, dim=-1).unsqueeze(1)
        distances_sq = X_norm_sq - 2 * (X @ self.X_train_.T) + X_train_norm_sq.T
        distances_sq.clamp_(min=0)
        
        _, neighbor_indices = torch.topk(distances_sq, k=self.k, dim=1, largest=False)
        neighbor_distributions = self.label_distributions_[neighbor_indices]
        y_pred_dist = neighbor_distributions.sum(axis=1)
        
        return self.classes_[torch.argmax(y_pred_dist, axis=1).cpu().numpy()]
    
def train_ifgst(
    fuzzy_model,
    X_l_orig, y_l_orig,
    X_u_orig,
    num_rounds=5,
    epochs_per_round=10,
    confidence_threshold=0.95,
    device='cpu'
):
    """
    Performs iterative self-training, using a GraphSSL model as a 'teacher'
    to generate pseudo-labels for retraining a fuzzy 'student' model.
    """
    X_l, y_l = X_l_orig.clone(), y_l_orig.clone()
    X_u = X_u_orig.clone()
    print(X_l.shape)
    
    print("--- Initial training on labeled data ---")
    if isinstance(fuzzy_model, (NoHybridANFIS, POPFNN)):
        optimizer = torch.optim.Adam(fuzzy_model.parameters(), lr=0.01)
        for _ in range(500): # Warm-up epochs
            fuzzy_model.train()
            optimizer.zero_grad()
            logits = fuzzy_model(X_l.to(device)) if isinstance(fuzzy_model, POPFNN) else fuzzy_model(X_l.to(device))[0]
            loss = F.cross_entropy(logits, y_l.to(device))
            loss.backward()
            optimizer.step()
    elif isinstance(fuzzy_model, FMNC):
        fuzzy_model.fit(X_l, y_l, epochs=5)

    for r in range(num_rounds):
        print(f"\n--- Self-Training Round {r+1}/{num_rounds} ---")
        if len(X_u) == 0:
            print("Unlabeled pool is empty. Stopping.")
            break

        # 1. Feature Extraction
        X_all = torch.cat([X_l, X_u])
        with torch.no_grad():
            # FIX: Only call .eval() if the model is a torch.nn.Module
            if isinstance(fuzzy_model, nn.Module):
                fuzzy_model.eval()
            
            X_all_dev = X_all.to(device)
            if isinstance(fuzzy_model, FMNC):
                rule_activations = fuzzy_model._get_rule_activations(X_all_dev)
            else:
                rule_activations = fuzzy_model._get_rule_activations(X_all_dev) if hasattr(fuzzy_model, '_get_rule_activations') else fuzzy_model._fire(X_all_dev)

        # 2. Pseudo-Label Generation (Teacher)
        y_semi_sup = torch.cat([y_l, torch.full((len(X_u),), -1, dtype=torch.long)])
        ssl_teacher = GraphSSL(method='iterative', k=15, sigma=0.2, device=device)
        ssl_teacher.fit(rule_activations, y_semi_sup)

        # 3. Confident Selection
        confidences = ssl_teacher.label_distributions_.max(1).values
  
        pseudo_labels = ssl_teacher.transduction_

        unlabeled_confidences = confidences[len(y_l):]

        unlabeled_pseudo_labels = pseudo_labels[len(y_l):]
        high_conf_mask = (unlabeled_confidences > confidence_threshold)
        
        if high_conf_mask.sum() == 0:
            print("No new pseudo-labels passed the confidence threshold. Stopping.")
            break
            
        # 4. Dataset Augmentation
        X_new = X_u[high_conf_mask.cpu()]
        y_new = torch.from_numpy(unlabeled_pseudo_labels[high_conf_mask.cpu().numpy()]).long()
        
        print(f"Adding {len(X_new)} new high-confidence pseudo-labels.")
        X_l = torch.cat([X_l, X_new])
        y_l = torch.cat([y_l, y_new])
        X_u = X_u[~high_conf_mask.cpu()]

        # 5. Fuzzy System Retraining (Student)
        print("Fine-tuning the fuzzy model on the augmented dataset...")
        if isinstance(fuzzy_model, (NoHybridANFIS, POPFNN)):
             optimizer = torch.optim.Adam(fuzzy_model.parameters(), lr=0.005) # Lower LR for fine-tuning
             for _ in range(epochs_per_round):
                fuzzy_model.train()
                optimizer.zero_grad()
                logits = fuzzy_model(X_l.to(device)) if isinstance(fuzzy_model, POPFNN) else fuzzy_model(X_l.to(device))[0]
                loss = F.cross_entropy(logits, y_l.to(device))
                loss.backward()
                optimizer.step()
        elif isinstance(fuzzy_model, FMNC):
            fuzzy_model.fit(X_l, y_l, epochs=epochs_per_round)

    print("\n--- IFGST training complete! ---")
    return fuzzy_model

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    
    datas = [load_pmd_data, ]
    for k in datas:
        X_train, y_train, X_test, y_test = k()
        
        clf2 = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0)
        clf2.fit(X_train, y_train)
        y_pred = clf2.predict(X_test)
        print('Accuracy: ', metrics.accuracy_score(y_test,y_pred))
        
        
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
            num_classes=len(y_train.unique()), num_mfs=4,max_rules=1000,
            seed=42,
            zeroG=True
        ).to(device)
        
        optimizer = torch.optim.Adam(anfis_model.parameters(), lr=0.01)
        for epoch in range(500):
            anfis_model.train()
            optimizer.zero_grad()
            logits, _, _ = anfis_model(X_l.to(device))
            loss = F.cross_entropy(logits, y_l.to(device))
            loss.backward()
            optimizer.step()

        anfis_model.eval()
        with torch.no_grad():
            _, rule_activations_train_anfis, _ = anfis_model(X_train.to(device))
            _, rule_activations_test_anfis, _ = anfis_model(X_test.to(device))
        print(f"Shape of ANFIS feature space: {rule_activations_train_anfis.shape}")

        methods_to_test = {
            "Gaussian Random Field": GraphSSL(k=15, sigma=0.2, method='grf', device=device),
            "Iterative k-NN Prop": GraphSSL(k=15, sigma=0.2, method='iterative', device=device),

        }

        for name, model in methods_to_test.items():
            print(f"\n--- Running SSL with: {name} ---")
            model.fit(rule_activations_train_anfis, y_semi_sup)
            unlabeled_mask = (y_semi_sup == -1)
            pseudo_label_acc = accuracy_score(y_train.numpy()[unlabeled_mask], model.transduction_[unlabeled_mask])
            print(f"  Pseudo-Label Accuracy: {pseudo_label_acc * 100:.2f}%")
            y_pred = model.predict(rule_activations_test_anfis)
            test_acc = accuracy_score(y_test.numpy(), y_pred)
            print(f"  Final Test Accuracy: {test_acc * 100:.2f}%")

        # --- Experiment 2: POPFNN ---
        print("\n\n--- EXPERIMENT 2: Using POPFNN as Feature Extractor ---")
        popfnn_model = POPFNN(
            d=X_train.shape[1],
            C=len(y_train.unique()),
            num_mfs=4
        ).to(device)
        
        print("Initializing POPFNN rules with pop_init...")
        popfnn_model.pop_init(X_l.to(device), y_l.to(device))
        print(f"POPFNN initialized with {popfnn_model.R} rules.")

        print("Fine-tuning POPFNN...")
        optimizer_pop = torch.optim.Adam(popfnn_model.parameters(), lr=0.01)
        for epoch in range(500):
            popfnn_model.train()
            optimizer_pop.zero_grad()
            logits = popfnn_model(X_l.to(device))
            loss = F.cross_entropy(logits, y_l.to(device))
            loss.backward()
            optimizer_pop.step()

        popfnn_model.eval()
        with torch.no_grad():
            rule_activations_train_pop = popfnn_model._fire(X_train.to(device))
            rule_activations_test_pop = popfnn_model._fire(X_test.to(device))
        print(f"Shape of POPFNN feature space: {rule_activations_train_pop.shape}")

        for name, model in methods_to_test.items():
            print(f"\n--- Running SSL with: {name} ---")
            model.fit(rule_activations_train_pop, y_semi_sup)
            unlabeled_mask = (y_semi_sup == -1)
            pseudo_label_acc = accuracy_score(y_train.numpy()[unlabeled_mask], model.transduction_[unlabeled_mask])
            print(f"  Pseudo-Label Accuracy: {pseudo_label_acc * 100:.2f}%")
            y_pred = model.predict(rule_activations_test_pop)
            test_acc = accuracy_score(y_test.numpy(), y_pred)
            print(f"  Final Test Accuracy: {test_acc * 100:.2f}%")
            
        # --- Experiment 3: FuzzyMinMax (FMNC) ---
        print("\n\n--- EXPERIMENT: Using advanced FMNC as Feature Extractor ---")
        fmm_model = FMNC(
            gamma=1.7, 
            theta0=2.5,
            theta_min=0.1,
            theta_decay=0.1,
            bound_mode="sum",
            aggr="mean",
            m_min=0.8,
            device=device
        )
        
        print("Seeding FMNC hyperboxes with k-Means...")
        fmm_model.seed_boxes_kmeans(X_l, y_l, k=3)
        
        print("Training FMNC classifier on labeled data...")
        fmm_model.fit(X_l, y_l, epochs=1, shuffle=True)
        print(f"FMNC created {len(fmm_model.V)} rules (hyperboxes).")

        print("Extracting hyperbox activations for SSL...")
        rule_activations_train_fmm = fmm_model._get_rule_activations(X_train)
        rule_activations_test_fmm = fmm_model._get_rule_activations(X_test)
        print(f"Shape of FMM feature space: {rule_activations_train_fmm.shape}")

        methods_to_test_fmm = {
            "Gaussian Random Field": GraphSSL(k=15, sigma=0.2, method='grf', device=device),
            "Iterative k-NN Prop": GraphSSL(k=15, sigma=0.2, method='iterative', device=device)
        }

        for name, model in methods_to_test_fmm.items():
            print(f"\n--- Running SSL with: {name} ---")
            model.fit(rule_activations_train_fmm, y_semi_sup)
            unlabeled_mask = (y_semi_sup == -1)
            # Corrected line for calculating pseudo-label accuracy
            pseudo_label_acc = accuracy_score(y_train.numpy()[unlabeled_mask], model.transduction_[unlabeled_mask])
            print(f"  Pseudo-Label Accuracy: {pseudo_label_acc * 100:.2f}%")
            y_pred = model.predict(rule_activations_test_fmm)
            test_acc = accuracy_score(y_test.numpy(), y_pred)
            print(f"  Final Test Accuracy: {test_acc * 100:.2f}%")
            
            
            
            

            
            
        """unlabeled_mask = (y_semi_sup == -1)
        X_u = X_train[unlabeled_mask]

        # --- IFGST with POPFNN ---
        print("\n\n--- EXPERIMENT: IFGST with POPFNN ---")
        popfnn_factory = lambda: POPFNN(d=X_train.shape[1], C=len(y_train.unique()), num_mfs=4).to(device)
        popfnn_student = popfnn_factory()
        popfnn_student.pop_init(X_l.to(device), y_l.to(device))
        
        final_popfnn_model = train_ifgst(
            fuzzy_model=popfnn_student,
            X_l_orig=X_l, y_l_orig=y_l, X_u_orig=X_u,
            num_rounds=20, epochs_per_round=4, confidence_threshold=0.20,
            device=device
        )
        with torch.no_grad():
            final_popfnn_model.eval()
            y_pred_pop = final_popfnn_model(X_test.to(device)).argmax(1)
            test_acc = accuracy_score(y_test.numpy(), y_pred_pop.cpu().numpy())
            print(f"\nFinal Test Accuracy (IFGST POPFNN): {test_acc * 100:.2f}%")"""
            
            
            
            

            

        # --- IFGST with FMNC ---
        """print("\n\n--- EXPERIMENT: IFGST with FMNC ---")
        fmnc_factory = lambda: FMNC(gamma=1.7, theta0=2.5, theta_min=0.1, theta_decay=0.1, 
                                    bound_mode="sum", aggr="mean", m_min=0.8, device=device)
        fmnc_student = fmnc_factory()
        fmnc_student.seed_boxes_kmeans(X_l, y_l, k=3)
        
        final_fmnc_model = train_ifgst(
            fuzzy_model=fmnc_student,
            X_l_orig=X_l, y_l_orig=y_l, X_u_orig=X_u,
            num_rounds=1, epochs_per_round=2, confidence_threshold=0.90,
            device=device
        )
        with torch.no_grad():
            y_pred_fmnc = final_fmnc_model.predict(X_test)
            test_acc = accuracy_score(y_test.numpy(), y_pred_fmnc.cpu().numpy())
            print(f"\nFinal Test Accuracy (IFGST FMNC): {test_acc * 100:.2f}%")"""

