# rule_based_trainers.py
import torch
import torch.nn.functional as F
import torch.nn as nn # For CrossEntropyLoss in POPFNN
import math
from anfis_nonHyb import NoHybridANFIS

from anfis_hybrid import HybridANFIS
from PopFnn import POPFNN
from anfisHelper import initialize_mfs_with_kmeans


# -------------------------------------------------------------
# utils --------------------------------------------------------
# -------------------------------------------------------------
def _entropy(p, eps=1e-12, base=None):
    p = p.clamp_min(eps)
    logp = p.log()
    if base is not None:                 # base-x entropy
        logp /= math.log(base)
    return -(p * logp).sum(-1)

def _val_split(X_l, y_l, frac=.1):
    n_val = max(1, int(len(X_l)*frac))
    idx   = torch.randperm(len(X_l), device=X_l.device)
    return (X_l[idx[n_val:]], y_l[idx[n_val:]]), (X_l[idx[:n_val]], y_l[idx[:n_val]])

# -------------------------------------------------------------
# 1)  HYBRID-ANFIS RULE-SSL -----------------------------------
# -------------------------------------------------------------
def train_hybrid_anfis_rule_ssl(
    X_l, y_l, X_u,
    input_dim, num_classes, device,
    num_mfs=4, max_rules=1000,
    lr=5e-3, seed=42,
    rule_conf_thresh=0.90, firing_thresh=0.50,
    num_iterations=3,
    epochs_initial=40,
    epochs_per_iteration=40,
    batch_u=256,
    patience=5,
    **kwargs  # Add kwargs to accept unused parameters from config
):
    torch.manual_seed(seed)
    model = HybridANFIS(input_dim, num_classes,
                        num_mfs, max_rules, seed=seed).to(device)
    initialize_mfs_with_kmeans(model, torch.cat([X_l, X_u]))
    opt = torch.optim.Adam([{'params': model.centers, 'lr': lr},
                            {'params': model.widths,  'lr': lr}])


    (X_tr, y_tr), (X_val, y_val) = _val_split(X_l, y_l, .1)
    y_tr_onehot = F.one_hot(y_tr, num_classes).float()

    for _ in range(epochs_initial):
        model.train()
        opt.zero_grad()
        logit, fs, x_ext = model(X_tr)
        F.cross_entropy(logit, y_tr).backward()
        opt.step()
        model.update_consequents(fs.detach(), x_ext.detach(), y_tr_onehot)


    pseudo_hist, best_val = [], 1e9
    mf_frozen = False

    X_pool = X_u.clone()
    X_p, y_p, w_p = (torch.empty(0, input_dim,  device=device),
                     torch.empty(0,           device=device, dtype=torch.long),
                     torch.empty(0,           device=device))

    for it in range(num_iterations):

     
        with torch.no_grad():
            model.eval()
            _, fs_l, _ = model(X_l)
            rule_class_w = fs_l.t() @ F.one_hot(y_l, num_classes).float()
            probs        = F.normalize(rule_class_w, p=1, dim=1)
            purity       = 1 - _entropy(probs, base=num_classes)     # 1-normalized H
            conf, cls    = probs.max(1)
            confident    = (conf > rule_conf_thresh*(0.95**it)) | (purity > .85)

        # ========== 2. SCAN UNLABELED POOL batch-wise ==========
        collected_X_p_iter = []
        collected_y_p_iter = []
        collected_w_p_iter = []
        indices_to_remove_from_pool = [] # Store original indices from X_pool

        model.eval()
        for s in range(0, len(X_pool), batch_u):
            xb = X_pool[s:s+batch_u]
            if xb.shape[0] == 0: continue

            _, fs_u_batch, _ = model(xb) # fs_u_batch: [current_batch_size, num_rules]
            mval_batch, ridx_batch = fs_u_batch.max(1) # mval_batch, ridx_batch: [current_batch_size]
            
            # Check confidence of the best rule for each sample in the batch
            best_rule_is_confident_batch = confident[ridx_batch] # [current_batch_size] (boolean)
            
            # Combine with firing strength threshold
            mask_batch = (best_rule_is_confident_batch & (mval_batch > firing_thresh)) # [current_batch_size] (boolean)

            if mask_batch.any():
                selected_indices_in_batch = torch.nonzero(mask_batch).flatten() # Indices within xb

                # Get data for these selected samples
                collected_X_p_iter.append(xb[selected_indices_in_batch])
                ridx_selected_batch = ridx_batch[selected_indices_in_batch] # Rule indices for selected samples
                
                collected_y_p_iter.append(cls[ridx_selected_batch])   # Predicted class using the best rule
                collected_w_p_iter.append(conf[ridx_selected_batch])  # Confidence of that prediction
                
                # Store original indices from X_pool to remove them later
                indices_to_remove_from_pool.append(selected_indices_in_batch + s)

        if not collected_X_p_iter: # No new pseudo-labels in this iteration
            break  # nothing new

  
        X_p_iter = torch.cat(collected_X_p_iter)
        y_p_iter = torch.cat(collected_y_p_iter)
        w_p_iter = torch.cat(collected_w_p_iter)

        X_p = torch.cat([X_p, X_p_iter])
        y_p = torch.cat([y_p, y_p_iter])
        w_p = torch.cat([w_p, w_p_iter])

        if indices_to_remove_from_pool:
            all_selected_original_indices = torch.cat(indices_to_remove_from_pool)
            keep_mask_for_pool = torch.ones(len(X_pool), dtype=torch.bool, device=device)
            if all_selected_original_indices.numel() > 0:
                keep_mask_for_pool[all_selected_original_indices] = False
            X_pool = X_pool[keep_mask_for_pool]
        elif not X_p_iter.numel(): # If X_p_iter is empty, means no pseudo labels, pool remains same
            pass # X_pool remains unchanged

    
        if it >= 1 and not mf_frozen:       # freeze MF once after first round
            for p in [model.centers, model.widths]:
                p.requires_grad_(False)
            mf_frozen = True
            # The existing optimizer 'opt' will no longer update centers and widths
            # as their requires_grad is False. No need to re-initialize opt.

        X_train = torch.cat([X_l, X_p])
        y_train = torch.cat([y_l, y_p])
        w_train = torch.cat([torch.ones(len(y_l), device=device), w_p])
        y_one   = F.one_hot(y_train, num_classes).float()

        stop = wait = 0
        for ep in range(epochs_per_iteration):
            model.train()
            opt.zero_grad()
            
            # Forward pass is always needed for LSE and potentially for MF training
            logit, fs, x_ext = model(X_train)

            if not mf_frozen:
                # If MFs are trainable, calculate loss and update them via optimizer
                loss = (w_train * F.cross_entropy(logit, y_train, reduction='none')).mean()
                loss.backward()
                opt.step()
            
            # LSE update for consequents (always happens)
            # fs and x_ext are from the model's current state.
            # If MFs were updated by opt.step(), LSE uses fs from that updated state if logit,fs,x_ext were re-evaluated.
            # If MFs were frozen, LSE uses fs from frozen MFs.
            # Current structure: LSE uses fs from the same forward pass that might have generated MF grads.
            model.update_consequents(fs.detach(), x_ext.detach(), y_one)

            # ---- early-stop on small val set -----------------
            with torch.no_grad():
                model.eval() # Ensure model is in eval mode for validation
                val_logit_eval, _, _ = model(X_val) 
                current_val_epoch_loss = F.cross_entropy(val_logit_eval, y_val)
            
            if current_val_epoch_loss.item() < best_val - 1e-3: # Use .item()
                best_val, wait = current_val_epoch_loss.item(), 0
            else:
                wait += 1
                if wait == patience:
                    break

        pseudo_hist.append(len(X_p_iter))

    # Return model and number of pseudo-labels to match expected unpacking
    return model, int(len(X_p))


def train_nohybrid_anfis_rule_ssl(
    X_l, y_l, X_u,
    input_dim, num_classes, device,
    num_mfs=7, max_rules=2000, zeroG=False,
    lr=5e-3, seed=42,
    rule_conf_thresh=0.90, firing_thresh=0.50,
    num_iterations=3,
    epochs_initial=40,
    epochs_per_iteration=40,
    batch_u=4096,
    patience=5,
):
    torch.manual_seed(seed)
    model = NoHybridANFIS(input_dim, num_classes, num_mfs,
                          max_rules, seed, zeroG).to(device)
    initialize_mfs_with_kmeans(model, torch.cat([X_l, X_u]))
    opt = torch.optim.Adam([{'params': model.centers, 'lr': lr},
                            {'params': model.widths,  'lr': lr}])


    (X_tr, y_tr), (X_val, y_val) = _val_split(X_l, y_l, .1)
    for _ in range(epochs_initial):
        model.train(); opt.zero_grad()
        out, fs, _ = model(X_tr)
        F.cross_entropy(out, y_tr).backward(); opt.step()

    X_pool = X_u.clone()
    # Correctly initialize cumulative pseudo-label tensors
    X_p_cumulative = torch.empty(0, input_dim, device=device)
    y_p_cumulative = torch.empty(0, dtype=torch.long, device=device)
    w_p_cumulative = torch.empty(0, device=device)
    
    for it in range(num_iterations):

        with torch.no_grad():
            model.eval()
            _, fs_l, _ = model(X_l)
            rcw = fs_l.t() @ F.one_hot(y_l, num_classes).float()
            probs = F.normalize(rcw, p=1, dim=1)
            purity = 1 - _entropy(probs, base=num_classes)
            conf, cls = probs.max(1)
            confident = (conf > rule_conf_thresh*(0.95**it)) | (purity > .85)

      
        collected_X_p_iter_nh = []
        collected_y_p_iter_nh = []
        collected_w_p_iter_nh = []
        indices_to_remove_from_pool_nh = []

        for s in range(0, len(X_pool), batch_u):
            xb = X_pool[s:s+batch_u]
            if xb.shape[0] == 0: continue

            _, fs_u_batch, _ = model(xb)
            mval_batch, ridx_batch = fs_u_batch.max(1)
            
            best_rule_is_confident_batch = confident[ridx_batch]
            mask_batch = (best_rule_is_confident_batch & (mval_batch > firing_thresh))

            if mask_batch.any():
                selected_indices_in_batch = torch.nonzero(mask_batch).flatten()
                
                collected_X_p_iter_nh.append(xb[selected_indices_in_batch])
                ridx_selected_batch = ridx_batch[selected_indices_in_batch]
                collected_y_p_iter_nh.append(cls[ridx_selected_batch])
                collected_w_p_iter_nh.append(conf[ridx_selected_batch])
                indices_to_remove_from_pool_nh.append(selected_indices_in_batch + s)

        if not collected_X_p_iter_nh:
            break

        X_p_iter_current = torch.cat(collected_X_p_iter_nh)
        y_p_iter_current = torch.cat(collected_y_p_iter_nh)
        w_p_iter_current = torch.cat(collected_w_p_iter_nh)

        X_p_cumulative = torch.cat([X_p_cumulative, X_p_iter_current])
        y_p_cumulative = torch.cat([y_p_cumulative, y_p_iter_current])
        w_p_cumulative = torch.cat([w_p_cumulative, w_p_iter_current])

        if indices_to_remove_from_pool_nh:
            all_selected_original_indices_nh = torch.cat(indices_to_remove_from_pool_nh)
            keep_mask_for_pool_nh = torch.ones(len(X_pool), dtype=torch.bool, device=device)
            if all_selected_original_indices_nh.numel() > 0:
                keep_mask_for_pool_nh[all_selected_original_indices_nh] = False
            X_pool = X_pool[keep_mask_for_pool_nh]
        elif not X_p_iter_current.numel():
             pass

        # ---------- retrain (freeze MFs after 1st) -----------
        if it >= 1:
            for p in [model.centers, model.widths]:
                p.requires_grad_(False)
            # Optimizer update for only consequent parameters if MFs are frozen
            # This assumes consequents are part of model.parameters() and require_grad
            # If consequents are not optimized by Adam, this might not be needed or needs adjustment
            # For NoHybridANFIS, all parameters are typically optimized by Adam initially.
            # If MFs are frozen, only consequents (if they require_grad) would be updated.
            # The current NoHybridANFIS has consequents as nn.Parameter, so they are in model.parameters().
            # We need to ensure their requires_grad status is handled correctly if MFs are frozen.
            # For simplicity, we can re-create the optimizer with only trainable parameters.
            trainable_params = filter(lambda p: p.requires_grad, model.parameters())
            # Check if there are any trainable parameters left
            if any(True for _ in trainable_params): # Re-check trainable_params as filter is an iterator
                 opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr*0.3)
            else: # No trainable parameters left, e.g. if consequents also don't require grad
                 pass # No optimizer steps will be effective

        X_train = torch.cat([X_l, X_p_cumulative])
        y_train = torch.cat([y_l, y_p_cumulative])
        w_train = torch.cat([torch.ones(len(y_l), device=device), w_p_cumulative])

        best, wait = 1e9, 0
        for _ in range(epochs_per_iteration):
            model.train(); opt.zero_grad()
            out, fs, _ = model(X_train)
            (w_train*F.cross_entropy(out, y_train, reduction='none')).mean().backward()
            opt.step()

            with torch.no_grad():
                val_loss = F.cross_entropy(model(X_val)[0], y_val)
            if val_loss < best - 1e-3:
                best, wait = val_loss.item(), 0
            else:
                wait += 1
                if wait == patience:
                    break

    # Return model and total number of pseudo-labels generated
    return model, int(len(X_p_cumulative))



def train_popfnn_rule_ssl(
    X_l, y_l, X_u, input_dim, num_classes, device,
    num_mfs=4, lr=5e-4, seed=42, 
    rule_conf_thresh=0.9,
    # New parameters for iterative training
    num_iterations=3,
    epochs_per_iteration=50,
    **kwargs
):
    """Trains POPFNN using an ITERATIVE rule-based self-training process."""
    torch.manual_seed(seed) 
    
    # --- 1. Initial Model Setup ---
    model = POPFNN(input_dim, num_classes, num_mfs=num_mfs).to(device)
    initialize_mfs_with_kmeans(model, torch.cat([X_l, X_u]))
    
    # --- 2. Initial Rule Generation from Labeled Data ---
    #print("--- Initializing POPFNN rules with labeled data ---")
    model.pop_init(X_l, y_l)
    if model.R == 0: # No rules were generated, train supervised only
        #print("No initial rules found. Training supervised only.")
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs_per_iteration * num_iterations): # Train for total epochs
            opt.zero_grad()
            loss = F.cross_entropy(model(X_l), y_l)
            loss.backward(); opt.step()
        return model, 0

    # --- 3. Iterative Self-Training Loop ---
    X_unlabeled_pool = X_u.clone()
    X_p_all = torch.empty(0, X_l.shape[1], device=device)
    y_p_all = torch.empty(0, device=device, dtype=torch.long)
    w_p_all = torch.empty(0, device=device)
    
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    for i in range(num_iterations):
       # print(f"\n--- Self-Training Iteration {i + 1}/{num_iterations} ---")

        if X_unlabeled_pool.shape[0] == 0:
           # print("No more unlabeled data to process. Stopping iterations.")
            break

        # --- 4. Generate New Pseudo-Labels from the CURRENT Unlabeled Pool ---
        with torch.no_grad():
            rule_class_weights = model.W.view(model.R, model.C, model.M).sum(dim=2)
            rule_confidence, confident_class = torch.max(rule_class_weights, dim=1)
            confident_rules_mask = rule_confidence > rule_conf_thresh
            
            idx_p_new = torch.tensor([], dtype=torch.long, device=device)
            if confident_rules_mask.sum().item() > 0:
                fire_u = model._fire(X_unlabeled_pool)
                sample_max_firing, sample_best_rule_idx = torch.max(fire_u, dim=1)
                best_rule_is_confident = confident_rules_mask[sample_best_rule_idx]
                idx_p_new = torch.where(best_rule_is_confident)[0]

        if len(idx_p_new) == 0:
            #print("No new high-confidence pseudo-labels generated. Stopping iterations.")
            break
            
        print(f"Generated {len(idx_p_new)} new pseudo-labels in this iteration.")
        
        # --- 5. Add New Pseudo-Labels and Update Pools ---
        X_p_new = X_unlabeled_pool[idx_p_new]
        best_rules_for_new_pseudo = sample_best_rule_idx[idx_p_new]
        y_p_new = confident_class[best_rules_for_new_pseudo]
        w_p_new = rule_confidence[best_rules_for_new_pseudo]
        
        X_p_all = torch.cat([X_p_all, X_p_new])
        y_p_all = torch.cat([y_p_all, y_p_new])
        w_p_all = torch.cat([w_p_all, w_p_new])
        
        mask = torch.ones(X_unlabeled_pool.shape[0], dtype=torch.bool, device=device)
        mask[idx_p_new] = False
        X_unlabeled_pool = X_unlabeled_pool[mask]
        
        # --- 6. Re-Train the Model on Labeled + ALL Cumulative Pseudo-Labeled Data ---
        X_train_iter = torch.cat([X_l, X_p_all])
        y_train_iter = torch.cat([y_l, y_p_all])
        w_train_iter = torch.cat([torch.ones(len(y_l), device=device), w_p_all])

        print(f"Re-training model on {len(X_l)} labeled + {len(X_p_all)} pseudo-labeled samples...")
        for _ in range(epochs_per_iteration):
            model.train()
            opt.zero_grad()
            logits = model(X_train_iter)
            loss = (loss_fn(logits, y_train_iter) * w_train_iter).mean()
            loss.backward()
            opt.step()
            
    return model, len(X_p_all)