# rule_based_trainers.py
import torch
import torch.nn.functional as F
import torch.nn as nn # For CrossEntropyLoss in POPFNN

from anfis_hybrid import HybridANFIS
from PopFnn import POPFNN
from anfisHelper import initialize_mfs_with_kmeans

def train_hybrid_anfis_rule_ssl(
    X_l, y_l, X_u, input_dim, num_classes, device,
    num_mfs=4, max_rules=1000, lr=5e-3, seed=42,
    rule_conf_thresh=0.9, firing_thresh=0.5,
    # New parameters for iterative training
    num_iterations=3,
    epochs_initial=40,
    epochs_per_iteration=40,
    **kwargs
):
    """
    Trains HybridANFIS using an ITERATIVE rule-based self-training process.
    """
    # --- 1. Initial Model and Optimizer Setup ---
    model = HybridANFIS(input_dim, num_classes, num_mfs, max_rules, seed=seed).to(device)
    # Initialize membership functions based on the entire dataset's feature distribution
    initialize_mfs_with_kmeans(model, torch.cat([X_l, X_u]))
    opt = torch.optim.Adam([{'params': model.centers, 'lr': lr}, {'params': model.widths, 'lr': lr}])
    
    # --- 2. Initial Supervised Training on Labeled Data ---
    # This creates the first baseline version of the model.
    #print(f"--- Performing initial supervised training for {epochs_initial} epochs ---")
    y_l_onehot = F.one_hot(y_l, num_classes).float()
    for _ in range(epochs_initial):
        model.train()
        opt.zero_grad()
        logits, norm_fs, x_ext = model(X_l)
        loss = F.cross_entropy(logits, y_l)
        loss.backward()
        opt.step()
        model.update_consequents(norm_fs.detach(), x_ext.detach(), y_l_onehot)

    # --- 3. Iterative Self-Training Loop ---
    # This pool will shrink as we generate pseudo-labels
    X_unlabeled_pool = X_u.clone()
    
    # These will accumulate all pseudo-labels found across iterations
    X_p_all = torch.empty(0, X_l.shape[1], device=device)
    y_p_all = torch.empty(0, device=device, dtype=torch.long)
    w_p_all = torch.empty(0, device=device)

    for i in range(num_iterations):
        #print(f"\n--- Self-Training Iteration {i + 1}/{num_iterations} ---")
        
        if X_unlabeled_pool.shape[0] == 0:
            #print("No more unlabeled data to process. Stopping iterations.")
            break
            
        # --- 4. Generate New Pseudo-Labels from the CURRENT Unlabeled Pool ---
        with torch.no_grad():
            model.eval()
            _, norm_fs_l, _ = model(X_l) # Use Labeled data to determine rule confidence
            rule_class_weights = norm_fs_l.t() @ y_l_onehot
            rule_class_probs = F.normalize(rule_class_weights, p=1, dim=1)
            rule_confidence, confident_class = torch.max(rule_class_probs, dim=1)
            confident_rules_mask = rule_confidence > rule_conf_thresh
            
            idx_p_new = torch.tensor([], dtype=torch.long, device=device)
            if confident_rules_mask.sum().item() > 0:
                # Use confident rules to make guesses on the remaining unlabeled pool
                _, norm_fs_u, _ = model(X_unlabeled_pool)
                sample_max_firing, sample_best_rule_idx = torch.max(norm_fs_u, dim=1)
                best_rule_is_confident = confident_rules_mask[sample_best_rule_idx]
                firing_is_strong = sample_max_firing > firing_thresh
                pseudo_label_mask = best_rule_is_confident & firing_is_strong
                idx_p_new = torch.where(pseudo_label_mask)[0]

        if len(idx_p_new) == 0:
            #print("No new high-confidence pseudo-labels generated. Stopping iterations.")
            break
        
        #print(f"Generated {len(idx_p_new)} new pseudo-labels in this iteration.")
        
        # --- 5. Add New Pseudo-Labels to the Cumulative Set ---
        X_p_new = X_unlabeled_pool[idx_p_new]
        best_rules_for_new_pseudo = sample_best_rule_idx[idx_p_new]
        y_p_new = confident_class[best_rules_for_new_pseudo]
        w_p_new = rule_confidence[best_rules_for_new_pseudo]
        
        X_p_all = torch.cat([X_p_all, X_p_new])
        y_p_all = torch.cat([y_p_all, y_p_new])
        w_p_all = torch.cat([w_p_all, w_p_new])
        
        # --- 6. Update the Unlabeled Pool (Remove Newly Labeled Samples) ---
        mask = torch.ones(X_unlabeled_pool.shape[0], dtype=torch.bool, device=device)
        mask[idx_p_new] = False
        X_unlabeled_pool = X_unlabeled_pool[mask]
        
        # --- 7. Re-Train the Model on Labeled + ALL Cumulative Pseudo-Labeled Data ---
        X_train_iter = torch.cat([X_l, X_p_all])
        y_train_iter = torch.cat([y_l, y_p_all])
        w_train_iter = torch.cat([torch.ones(len(y_l), device=device), w_p_all])
        y_train_iter_onehot = F.one_hot(y_train_iter, num_classes).float()
        
        #print(f"Re-training model on {len(X_l)} labeled + {len(X_p_all)} pseudo-labeled samples...")
        for _ in range(epochs_per_iteration):
            model.train()
            opt.zero_grad()
            logits, norm_fs, x_ext = model(X_train_iter)
            loss = (w_train_iter * F.cross_entropy(logits, y_train_iter, reduction='none')).mean()
            loss.backward()
            opt.step()
            model.update_consequents(norm_fs.detach(), x_ext.detach(), y_train_iter_onehot)

    # Return the final trained model and the total number of pseudo-labels generated
    return model, len(X_p_all)


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