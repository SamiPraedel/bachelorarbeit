# rule_based_trainers.py
import torch
import torch.nn.functional as F
import torch.nn as nn # For CrossEntropyLoss in POPFNN

from anfis_hybrid import HybridANFIS
from PopFnn import POPFNN
from anfisHelper import initialize_mfs_with_kmeans, initialize_mfs_with_fcm, set_rule_subset

def train_hybrid_anfis_rule_ssl(X_l, y_l, X_u, input_dim, num_classes, device,
                                num_mfs=4, max_rules=1000, epochs=200, lr=5e-3, seed=42,
                                initial_train_ratio=0.2, rule_conf_thresh=0.9, firing_thresh=0.5, **kwargs):
    model = HybridANFIS(input_dim, num_classes, num_mfs, max_rules, seed=seed).to(device)
    X_all = torch.cat([X_l, X_u])
    initialize_mfs_with_kmeans(model, X_all)
    opt = torch.optim.Adam([{'params': model.centers, 'lr': lr}, {'params': model.widths, 'lr': lr}])
    
    initial_epochs = int(epochs * initial_train_ratio)
    y_l_onehot = F.one_hot(y_l, num_classes).float()
    
    for _ in range(initial_epochs):
        model.train()
        opt.zero_grad()
        logits, norm_fs, x_ext = model(X_l)
        loss = F.cross_entropy(logits, y_l)
        loss.backward()
        opt.step()
        model.update_consequents(norm_fs.detach(), x_ext.detach(), y_l_onehot)

    X_p, y_p, w_p = torch.empty(0, X_l.shape[1], device=device), torch.empty(0, device=device, dtype=torch.long), torch.empty(0, device=device)
    
    with torch.no_grad():
        model.eval()
        _, norm_fs_l, _ = model(X_l)
        rule_class_weights = norm_fs_l.t() @ y_l_onehot
        rule_class_probs = F.normalize(rule_class_weights, p=1, dim=1)
        rule_confidence, confident_class = torch.max(rule_class_probs, dim=1)
        confident_rules_mask = rule_confidence > rule_conf_thresh
        
        if confident_rules_mask.sum().item() > 0 and len(X_u) > 0:
            _, norm_fs_u, _ = model(X_u)
            sample_max_firing, sample_best_rule_idx = torch.max(norm_fs_u, dim=1)
            best_rule_is_confident = confident_rules_mask[sample_best_rule_idx]
            firing_is_strong = sample_max_firing > firing_thresh
            pseudo_label_mask = best_rule_is_confident & firing_is_strong
            idx_p = torch.where(pseudo_label_mask)[0]
            
            if len(idx_p) > 0:
                X_p = X_u[idx_p]
                best_rules_for_pseudo_samples = sample_best_rule_idx[idx_p]
                y_p = confident_class[best_rules_for_pseudo_samples]
                w_p = rule_confidence[best_rules_for_pseudo_samples]
    
    X_all = torch.cat([X_l, X_p])
    y_all = torch.cat([y_l, y_p])
    w_all = torch.cat([torch.ones(len(y_l), device=device), w_p])
    y_all_onehot = F.one_hot(y_all, num_classes).float()

    for _ in range(epochs - initial_epochs):
        model.train()
        opt.zero_grad()
        logits, norm_fs, x_ext = model(X_all)
        loss = (w_all * F.cross_entropy(logits, y_all, reduction='none')).mean()
        loss.backward()
        opt.step()
        model.update_consequents(norm_fs.detach(), x_ext.detach(), y_all_onehot)

    return model, len(X_p)


def train_popfnn_rule_ssl(X_l, y_l, X_u, input_dim, num_classes, device,
                          num_mfs=4, epochs=50, lr=5e-4, seed=42, 
                          rule_conf_thresh=0.9, **kwargs):
    torch.manual_seed(seed) 

    model = POPFNN(input_dim, num_classes, num_mfs=num_mfs).to(device)
    X_all = torch.cat([X_l, X_u])
    initialize_mfs_with_kmeans(model, X_all)
    model.pop_init(X_l, y_l)
    
    if model.R == 0: 
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            loss = F.cross_entropy(model(X_l), y_l)
            loss.backward(); opt.step()
        return model, 0

    X_p, y_p, w_p = torch.empty(0, X_l.shape[1], device=device), torch.empty(0, device=device, dtype=torch.long), torch.empty(0, device=device)
    
    with torch.no_grad():
        rule_class_weights = model.W.view(model.R, model.C, model.M).sum(dim=2)
        rule_confidence, confident_class = torch.max(rule_class_weights, dim=1)
        confident_rules_mask = rule_confidence > rule_conf_thresh

        if confident_rules_mask.sum().item() > 0 and len(X_u) > 0:
            fire_u = model._fire(X_u) # _fire method might need to be public or accessed differently if not intended
            sample_max_firing, sample_best_rule_idx = torch.max(fire_u, dim=1)
            best_rule_is_confident = confident_rules_mask[sample_best_rule_idx]
            idx_p = torch.where(best_rule_is_confident)[0]

            if len(idx_p) > 0:
                X_p = X_u[idx_p]
                best_rules_for_pseudo_samples = sample_best_rule_idx[idx_p]
                y_p = confident_class[best_rules_for_pseudo_samples]
                w_p = rule_confidence[best_rules_for_pseudo_samples]

    X_all = torch.cat([X_l, X_p])
    y_all = torch.cat([y_l, y_p])
    w_all = torch.cat([torch.ones(len(y_l), device=device), w_p])
    
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(X_all)
        loss = (loss_fn(logits, y_all) * w_all).mean()
        loss.backward()
        opt.step()
        
    return model, len(X_p)
