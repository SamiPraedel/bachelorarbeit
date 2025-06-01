# fmnc_mixed.py
from __future__ import annotations
import torch, numpy as np
from torch import Tensor
from typing import Optional, Literal, List
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import OneHotEncoder

class FMNC:
    def __init__(self,
                 gamma: float = 0.6,
                 theta0: float = 1.0,
                 theta_min: float = 0.6,
                 theta_decay: float = 0.97,
                 bound_mode: Literal["sum", "max"] = "sum",
                 aggr: Literal["min", "mean"] = "min",
                 m_min: float = 0.8,
                 discrete_features: Optional[List[int]] = None,
                 device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
                 batch_size: int = 256):
        
        # Hyperparameters
        self.g, self.th = gamma, theta0
        self.th_min, self.th_decay = theta_min, theta_decay
        self.bound_mode, self.aggr = bound_mode, aggr
        self.m_min = m_min
        self.dev = torch.device(device)
        self.bs = batch_size
        
        # Box-Container
        self.V = self.W = self.cls = None

        # Mixed data handling
        self.discrete_indices_original = discrete_features # Store original indices
        self.ohe = None
        self.is_fitted = False
        
        # These will store the indices in the *transformed* data
        self.final_cont_indices_transformed = None
        self.final_disc_indices_transformed = None
        self.num_original_features = None


    def _preprocess_data(self, X: Tensor) -> Tensor:
        """One-hot encodes discrete features if specified."""
        if self.num_original_features is None and not self.is_fitted:
            self.num_original_features = X.shape[1]

        if self.discrete_indices_original is None: # All features are continuous
            if not self.is_fitted:
                self.final_cont_indices_transformed = torch.arange(X.shape[1], dtype=torch.long, device=self.dev)
                self.final_disc_indices_transformed = torch.tensor([], dtype=torch.long, device=self.dev)
            return X

        X_np = X.cpu().numpy()
        
        if not self.is_fitted:
            continuous_indices_original = [
                i for i in range(self.num_original_features) if i not in self.discrete_indices_original
            ]
            n_cont = len(continuous_indices_original)
            
            if not self.discrete_indices_original: # discrete_features was an empty list []
                 self.ohe = None
                 self.final_cont_indices_transformed = torch.arange(self.num_original_features, dtype=torch.long, device=self.dev)
                 self.final_disc_indices_transformed = torch.tensor([], dtype=torch.long, device=self.dev)
            else: # discrete_features has actual indices
                if any(idx < 0 or idx >= self.num_original_features for idx in self.discrete_indices_original):
                    raise ValueError("Discrete feature index out of bounds.")

                self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop=None)
                self.ohe.fit(X_np[:, self.discrete_indices_original])
            
                n_ohe_features = self.ohe.get_feature_names_out().shape[0] if self.ohe else 0
                
                # Corrected: Continuous features are first in transformed data
                self.final_cont_indices_transformed = torch.arange(n_cont, dtype=torch.long, device=self.dev)
                self.final_disc_indices_transformed = torch.arange(n_cont, n_cont + n_ohe_features, dtype=torch.long, device=self.dev)

        # Apply transformation using already fitted OHE and determined indices
        continuous_indices_original_for_transform = [
            i for i in range(self.num_original_features) if i not in (self.discrete_indices_original or [])
        ]
        
        if not continuous_indices_original_for_transform:
             X_cont_part = np.empty((X_np.shape[0], 0), dtype=X_np.dtype)
        else:
             X_cont_part = X_np[:, continuous_indices_original_for_transform]
        
        if self.ohe and self.discrete_indices_original:
            if not self.discrete_indices_original: # Should not happen if self.ohe is not None
                 X_disc_part_ohe = np.empty((X_np.shape[0], 0), dtype=float)
            else:
                 X_disc_part_ohe = self.ohe.transform(X_np[:, self.discrete_indices_original])
            X_transformed_np = np.concatenate([X_cont_part, X_disc_part_ohe], axis=1)
        else:
            X_transformed_np = X_cont_part # Only continuous or no discrete specified
            
        return torch.tensor(X_transformed_np, dtype=torch.float32, device=self.dev)


    def _memb_batch(self, X_transformed: Tensor) -> Tensor:
        V_transformed, W_transformed = self.V, self.W
        
        is_purely_continuous = (self.discrete_indices_original is None) or \
                               (not self.discrete_indices_original)
        
        if is_purely_continuous:
            # This path is for when discrete_features=None or discrete_features=[]
            # All original features are treated as continuous, OHE is None.
            # X_transformed is the same as original X in this case.
            # V_transformed and W_transformed also store original feature values.
            left  = 1 - self.g * torch.clamp(V_transformed.unsqueeze(0) - X_transformed.unsqueeze(1), min=0)
            right = 1 - self.g * torch.clamp(X_transformed.unsqueeze(1) - W_transformed.unsqueeze(0), min=0)
            m = torch.minimum(left, right)
            return m.amin(2) if self.aggr == "min" else m.mean(2)
        else: # Mixed data or all discrete (OHE is involved)
            # Continuous part calculation
            if self.final_cont_indices_transformed is not None and self.final_cont_indices_transformed.numel() > 0:
                X_cont_transformed_slice = X_transformed[:, self.final_cont_indices_transformed]
                V_cont_transformed_slice = V_transformed[:, self.final_cont_indices_transformed]
                W_cont_transformed_slice = W_transformed[:, self.final_cont_indices_transformed]
                
                # This check is crucial if somehow the slice resulted in 0 columns, though
                # with arange(n_cont), if n_cont > 0, shape[1] will be > 0.
                if X_cont_transformed_slice.shape[1] > 0:
                    left_cont  = 1 - self.g * torch.clamp(V_cont_transformed_slice.unsqueeze(0) - X_cont_transformed_slice.unsqueeze(1), min=0)
                    right_cont = 1 - self.g * torch.clamp(X_cont_transformed_slice.unsqueeze(1) - W_cont_transformed_slice.unsqueeze(0), min=0)
                    m_cont_temp = torch.minimum(left_cont, right_cont)
                    m_cont = m_cont_temp.amin(2) if self.aggr == "min" else m_cont_temp.mean(2)
                else: # Should ideally not be reached if numel() > 0 logic is sound
                    m_cont = torch.ones(X_transformed.shape[0], V_transformed.shape[0], device=self.dev)
            else: # No continuous features
                m_cont = torch.ones(X_transformed.shape[0], V_transformed.shape[0], device=self.dev)

            # Discrete part calculation (exact match on one-hot encoded dimensions)
            if self.final_disc_indices_transformed is not None and self.final_disc_indices_transformed.numel() > 0:
                X_disc_transformed_slice = X_transformed[:, self.final_disc_indices_transformed]
                V_disc_transformed_slice = V_transformed[:, self.final_disc_indices_transformed]
                m_disc = (X_disc_transformed_slice.unsqueeze(1) == V_disc_transformed_slice.unsqueeze(0)).all(dim=2).float()
            else: # No discrete features (or OHE resulted in no columns, unlikely)
                m_disc = torch.ones(X_transformed.shape[0], V_transformed.shape[0], device=self.dev)
                
            return torch.minimum(m_cont, m_disc)


    def _span(self, v_transformed: Tensor, w_transformed: Tensor) -> float:
        # Span is only meaningful for continuous dimensions in the transformed space
        if self.final_cont_indices_transformed is None or self.final_cont_indices_transformed.numel() == 0 :
             return 0.0 
        
        v_cont_part = v_transformed[self.final_cont_indices_transformed]
        w_cont_part = w_transformed[self.final_cont_indices_transformed]
            
        side = w_cont_part - v_cont_part
        return side.sum().item() if self.bound_mode == "sum" else side.max().item()

    def _add_box(self, x_transformed: Tensor, y: int):
        self.V  = x_transformed.unsqueeze(0) if self.V  is None else torch.cat([self.V,  x_transformed.unsqueeze(0)])
        self.W  = x_transformed.unsqueeze(0) if self.W  is None else torch.cat([self.W,  x_transformed.unsqueeze(0)])
        lbl     = torch.tensor([y], device=self.dev)
        self.cls = lbl            if self.cls is None else torch.cat([self.cls, lbl])

    def _contract(self, j: int):
        if self.final_cont_indices_transformed is None or self.final_cont_indices_transformed.numel() == 0:
            return # No continuous dimensions to contract
        
        cont_indices_slice = self.final_cont_indices_transformed
        
        V_cont_j = self.V[j, cont_indices_slice]
        # W_cont_j = self.W[j, cont_indices_slice] # Not needed for inter_low/high with V_cont_j
        V_oth_cont_all = self.V[:, cont_indices_slice]
        W_oth_cont_all = self.W[:, cont_indices_slice]

        mask = self.cls != self.cls[j]
        if not mask.any(): return
        
        V_oth_cont_masked = V_oth_cont_all[mask]
        W_oth_cont_masked = W_oth_cont_all[mask]

        inter_low  = torch.maximum(V_oth_cont_masked, V_cont_j) # V_cont_j is [D_cont]
        inter_high = torch.minimum(W_oth_cont_masked, self.W[j, cont_indices_slice]) # self.W[j, cont_indices_slice] is [D_cont]
        inter_len  = inter_high - inter_low
        overlap    = (inter_len > 0).all(1) 

        for k_sub in overlap.nonzero(as_tuple=False).flatten():
            k = torch.arange(len(self.V), device=self.dev)[mask][k_sub].item()
            
            for i_cont_idx_in_slice in (inter_len[k_sub] > 0).nonzero(as_tuple=False).flatten():
                i_transformed = cont_indices_slice[i_cont_idx_in_slice]
                
                vj, wj = self.V[j, i_transformed], self.W[j, i_transformed]
                vk, wk = self.V[k, i_transformed], self.W[k, i_transformed]
                
                if   vj < vk < wj < wk: mid = (vk + wj) / 2; vk, wj = mid, mid
                elif vk < vj < wk < wj: mid = (vj + wk) / 2; vj, wk = mid, mid
                elif vj < vk < wk < wj:
                    if (wj - vk) > (wk - vj): vj = wk
                    else:                     wj = vk
                else: 
                    if (wk - vj) > (wj - vk): vk = wj
                    else:                     wk = vj

                self.V[j, i_transformed], self.W[j, i_transformed] = vj, wj
                self.V[k, i_transformed], self.W[k, i_transformed] = vk, wk
    
    def fit(self, X: Tensor, y: Tensor, epochs: int = 3, shuffle: bool = True):
        if not self.is_fitted: 
            self.num_original_features = X.shape[1]
            # Call _preprocess_data once to fit OHE and set up indices
            _ = self._preprocess_data(X) 
            self.is_fitted = True
        
        X_transformed = self._preprocess_data(X) 
        y = y.to(self.dev)
        
        for ep in range(epochs):
            idx = torch.randperm(len(X_transformed)) if shuffle else torch.arange(len(X_transformed))
            Xs_transformed, ys = X_transformed[idx], y[idx]

            for s_idx in range(0, len(Xs_transformed), self.bs):
                xb_transformed = Xs_transformed[s_idx : s_idx+self.bs]
                yb = ys[s_idx : s_idx+self.bs]

                if xb_transformed.shape[0] == 0: continue # Skip empty batches

                if self.V is None: 
                    for x_i_transformed, y_i_val in zip(xb_transformed, yb):
                        self._add_box(x_i_transformed, int(y_i_val))
                    continue
                
                num_boxes_before_batch = self.V.shape[0] 
                memb_batch_vs_existing_boxes = self._memb_batch(xb_transformed)

                for i_in_batch in range(len(xb_transformed)):
                    x_i_transformed = xb_transformed[i_in_batch]
                    y_i_val = int(yb[i_in_batch])

                    if num_boxes_before_batch == 0: # Should be caught by self.V is None, but as safeguard
                         self._add_box(x_i_transformed, y_i_val)
                         num_boxes_before_batch = self.V.shape[0] # Update as a box was added
                         continue

                    m_for_sample = memb_batch_vs_existing_boxes[i_in_batch][:num_boxes_before_batch].clone()
                    
                    m_for_sample[self.cls[:num_boxes_before_batch] != y_i_val] = -1 
                    
                    if m_for_sample.numel() == 0 or m_for_sample.max() < self.m_min : 
                        self._add_box(x_i_transformed, y_i_val)
                        # num_boxes_before_batch does not change for subsequent samples in *this* batch
                        continue
                    
                    j_box_idx = int(m_for_sample.argmax()) 

                    v_current_j = self.V[j_box_idx].clone()
                    w_current_j = self.W[j_box_idx].clone()
                    
                    cont_indices_for_exp = self.final_cont_indices_transformed
                    if cont_indices_for_exp is not None and cont_indices_for_exp.numel() > 0:
                        v_current_j[cont_indices_for_exp] = torch.minimum(
                            self.V[j_box_idx, cont_indices_for_exp], 
                            x_i_transformed[cont_indices_for_exp]
                        )
                        w_current_j[cont_indices_for_exp] = torch.maximum(
                            self.W[j_box_idx, cont_indices_for_exp], 
                            x_i_transformed[cont_indices_for_exp]
                        )

                    if self._span(v_current_j, w_current_j) <= self.th:
                        self.V[j_box_idx] = v_current_j
                        self.W[j_box_idx] = w_current_j 
                        self._contract(j_box_idx)
                    else:
                        self._add_box(x_i_transformed, y_i_val)
            
            if self.V is not None:
                 print(f"epoch {ep+1}/{epochs} – θ={self.th:.3f}  #boxes={len(self.V)}")
            else:
                 print(f"epoch {ep+1}/{epochs} – θ={self.th:.3f}  #boxes=0")
            self.th = max(self.th * self.th_decay, self.th_min)

    @torch.no_grad()
    def predict(self, X: Tensor) -> Tensor:
        if not self.is_fitted: raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        if self.V is None: return torch.tensor([], dtype=torch.long, device=self.dev) 
        
        X_transformed = self._preprocess_data(X)
        if X_transformed.shape[0] == 0: return torch.tensor([], dtype=torch.long, device=self.dev)

        # Ensure V is not None before calling _memb_batch if predict is called on an empty model
        if self.V is None or self.V.shape[0] == 0:
            return torch.full((X_transformed.shape[0],), -1, dtype=torch.long, device=self.dev) # Default prediction

        memb = self._memb_batch(X_transformed)
        if memb.shape[1] == 0: 
            return torch.full((X_transformed.shape[0],), -1, dtype=torch.long, device=self.dev)

        return self.cls[memb.argmax(1)]

    def score(self, X: Tensor, y: Tensor) -> float:
        y_pred = self.predict(X)
        if len(y_pred) == 0 and len(y)==0: return 1.0
        if len(y_pred) == 0 or len(y)==0: return 0.0
        return (y_pred == y.to(self.dev)).float().mean().item()

    @torch.no_grad()
    def mcc_score(self, X: Tensor, y_true: Tensor) -> float:
        y_pred = self.predict(X)
        if len(y_pred) == 0 and len(y_true)==0: return 1.0 
        if len(y_pred) == 0 or len(y_true)==0: return 0.0 
        
        y_true_cpu = y_true.cpu().numpy()
        y_pred_cpu = y_pred.cpu().numpy()

        if len(np.unique(y_pred_cpu)) == 1 and len(np.unique(y_true_cpu)) > 1:
             # Sklearn's MCC handles this case (often returns 0.0)
             pass
        if len(np.unique(y_true_cpu)) == 1 and len(np.unique(y_pred_cpu)) > 1:
             pass
        if len(y_pred_cpu) != len(y_true_cpu): # Should not happen if predict is correct
            print(f"Warning: y_pred length ({len(y_pred_cpu)}) and y_true length ({len(y_true_cpu)}) mismatch in mcc_score.")
            return 0.0


        return float(matthews_corrcoef(y_true_cpu, y_pred_cpu))

# ---------------------------------------------------------------
if __name__ == "__main__":
    # Attempt to import data_utils, handle if not found for basic testing
    try:
        from data_utils import load_Kp_chess_data_ord
        data_utils_available = True
    except ImportError:
        print("Warning: data_utils.py not found or load_Kp_chess_data_ord is missing. Skipping KpChess test.")
        data_utils_available = False
        # Fallback to a simple synthetic dataset if data_utils is not available
        from sklearn.datasets import make_classification
        print("\n--- USING SYNTHETIC DATA AS FALLBACK ---")
        # Create a synthetic dataset with 35 binary features
        X_synthetic, y_synthetic = make_classification(
            n_samples=500, n_features=35, n_informative=10, n_redundant=5, 
            n_classes=3, random_state=42
        )
        # Convert features to binary (0 or 1) - this is a simplification
        X_synthetic = (X_synthetic > np.median(X_synthetic, axis=0)).astype(float) 
        
        Xtr = torch.from_numpy(X_synthetic).float()
        ytr = torch.from_numpy(y_synthetic).long()
        Xte, yte = Xtr, ytr # Use same for test for simplicity
        
        # Since all 35 synthetic features are now binary
        discrete_feature_indices = list(range(35))
        dataset_name = "Synthetic Binary Data"


    if data_utils_available:
        print("\n--- LOADING Kp_chess_data_ord ---")
        # Xtr, ytr, Xte, yte = load_K_chess_data_splitted()   # → Tensoren
        Xtr, ytr, Xte, yte = load_Kp_chess_data_ord()   # → Tensoren
        print(f"Original y_test shape: {yte.shape}")
        print(f"Original X_train shape: {Xtr.shape}")
        dataset_name = "Kp_chess_data_ord"

        if Xtr.shape[1] != 35:
            raise ValueError(f"Expected 35 features from load_Kp_chess_data_ord, but got {Xtr.shape[1]}")
        
        discrete_feature_indices = list(range(35)) # All 35 features are treated as discrete/binary

    print(f"\n--- CONFIGURING FMNC for {dataset_name} ---")
    print(f"Treating features with original indices {discrete_feature_indices} as discrete.")

    clf = FMNC(
        gamma        = 0.2,
        theta0       = 1.0, 
        theta_min    = 0.6,
        theta_decay  = 0.97,
        bound_mode   = "sum",
        aggr         = "min",
        m_min        = 0.8,
        discrete_features = discrete_feature_indices
    )
    
    print("\n--- FITTING FMNC ---")
    clf.fit(Xtr, ytr, epochs=1, shuffle=True)
    
    print("\n--- EVALUATING FMNC ---")
    test_accuracy = clf.score(Xte, yte)
    print(f"Test-Acc ({dataset_name}): {test_accuracy:.4f}")
    
    if clf.V is not None:
        print(f"#Boxes created: {len(clf.V)}")
        print(f"Shape of internal boxes (V): {clf.V.shape}")
        if clf.ohe:
            num_ohe_features = clf.ohe.get_feature_names_out().shape[0]
            print(f"Number of one-hot encoded features generated by OHE: {num_ohe_features}")
            
            expected_transformed_dim = num_ohe_features
            if clf.final_cont_indices_transformed is not None:
                expected_transformed_dim = clf.final_cont_indices_transformed.numel() + num_ohe_features
            
            if clf.V.shape[1] != expected_transformed_dim:
                 print(f"Warning: Box dimension {clf.V.shape[1]} does not match expected OHE features ({num_ohe_features}) + continuous features ({clf.final_cont_indices_transformed.numel() if clf.final_cont_indices_transformed else 0}). Expected {expected_transformed_dim}")

            if clf.final_cont_indices_transformed is None or clf.final_cont_indices_transformed.numel() == 0:
                 print("No continuous features were identified by the model.")
            else:
                 print(f"Continuous features identified at transformed indices: {clf.final_cont_indices_transformed.tolist()}")
    else:
        print("No boxes were created by the model.")

