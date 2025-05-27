# teacher_student_trainers.py
import torch
import torch.nn.functional as F
import torch.nn as nn

from anfis_hybrid import HybridANFIS
from anfis_nonHyb import NoHybridANFIS
from PopFnn import POPFNN

def train_hybrid_anfis_ssl(X_l, y_l, X_p, y_p, w_p,
                           input_dim, num_classes, device,
                           num_mfs=4, max_rules=1000,
                           epochs=50, lr=5e-3, seed=42, **kwargs):
    X_all = torch.cat([X_l, X_p])
    y_all = torch.cat([y_l, y_p])
    w_all = torch.cat([torch.ones(len(y_l), device=device), w_p])
    y_onehot = F.one_hot(y_all, num_classes).float()

    model = HybridANFIS(input_dim, num_classes, num_mfs, max_rules, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': model.centers, 'lr': lr}, {'params': model.widths, 'lr': lr}])

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits, norm_fs, x_ext = model(X_all)
        loss = (w_all * F.cross_entropy(logits, y_all, reduction='none')).mean()
        loss.backward()
        opt.step()
        model.update_consequents(norm_fs.detach(), x_ext.detach(), y_onehot)
    return model

def train_nohybrid_anfis_ssl(X_l, y_l, X_p, y_p, w_p,
                             input_dim, num_classes, device,
                             num_mfs=7, max_rules=2000, zeroG=False,
                             epochs=100, lr=5e-3, seed=42, **kwargs):
    X_all = torch.cat([X_l, X_p])
    y_all = torch.cat([y_l, y_p])
    w_all = torch.cat([torch.ones(len(y_l), device=device), w_p**2])

    model = NoHybridANFIS(input_dim, num_classes, num_mfs, max_rules, seed=seed, zeroG=zeroG).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits, norm_fs, mask = model(X_all)
        loss_main = (w_all * ce_loss_fn(logits, y_all)).mean()
        loss_aux = model.load_balance_loss(norm_fs.detach(), mask)
        loss = loss_main + loss_aux
        loss.backward()
        opt.step()
    return model

def train_popfnn_ssl(X_l, y_l, X_p, y_p, w_p,
                     input_dim, num_classes, device,
                     num_mfs=4, epochs=50, lr=5e-4, seed=42, **kwargs):
    X_all = torch.cat([X_l, X_p])
    y_all = torch.cat([y_l, y_p])
    w_all = torch.cat([torch.ones(len(y_l), device=device), w_p])

    model = POPFNN(input_dim, num_classes, num_mfs=num_mfs).to(device)
    model.pop_init(X_l, y_l)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(X_all)
        loss = (loss_fn(logits, y_all) * w_all).mean()
        loss.backward()
        opt.step()
    return model
