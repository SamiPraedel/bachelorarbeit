import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import skfuzzy as fuzz
from torch import nn
from sklearn.metrics import mutual_info_score
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np, torch

def initialize_mfs_with_kmeans(model, data):
    """
    data: NumPy array of shape (num_samples, input_dim).
        Each column is one feature/dimension.

    Returns:
    centers: NumPy array of shape (input_dim, num_mfs)
    widths:  NumPy array of shape (input_dim, num_mfs)
    """
    
    input_dim = data.shape[1]
    num_mfs = model.M  # Anzahl der Membership Functions pro Dimension

    centers_list = []
    widths_list = []

    for i in range(input_dim):
        # Daten für Dimension i extrahieren:
        X_dim = data[:, i].reshape(-1, 1)  # Shape: [num_samples, 1]

        km = KMeans(n_clusters=num_mfs, random_state=0, n_init='auto')
        km.fit(X_dim.cpu())

        # Clusterzentren als 1D-Array extrahieren:
        centers_i = km.cluster_centers_.flatten()

        stds_i = []
        labels = km.labels_
        for c_idx in range(num_mfs):
            cluster_points = X_dim[labels == c_idx]

            if not isinstance(cluster_points, np.ndarray):
                cluster_points = cluster_points.cpu().numpy()
            if len(cluster_points) > 1:
                std_val = np.std(cluster_points)
            else:
                std_val = 0.1  # Fallback für einen einzelnen Datenpunkt
            std_val = max(std_val, 0.0001)  # Minimaler positiver Wert
            stds_i.append(std_val)


        pairs = sorted(zip(centers_i, stds_i), key=lambda pair: pair[0])

        centers_i, stds_i = zip(*pairs)
        centers_i = np.array(centers_i)
        stds_i = np.array(stds_i)

        centers_list.append(centers_i)
        widths_list.append(stds_i)


    centers = np.array(centers_list, dtype=np.float32)  # Shape: (input_dim, num_mfs)
    widths  = np.array(widths_list,  dtype=np.float32)   # Shape: (input_dim, num_mfs)
    dev = data.device
    with torch.no_grad():
        model.centers.data.copy_(torch.tensor(centers, device=dev))
        model.widths .data.copy_(torch.tensor(widths,  device=dev))



    return centers, widths

def initialize_mfs_with_fcm(model, data, m=2.0, error=1e-5, maxiter=200):
    """
    Initialize MF parameters via Fuzzy C-Means clustering.

    Returns:
        centers: np.ndarray shape (input_dim, num_mfs)
        widths:  np.ndarray shape (input_dim, num_mfs)
    """
    # Ensure data is a NumPy array
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    X = data.T 
    num_mfs = model.M

    # run FCM
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        X, c=num_mfs, m=m, error=error, maxiter=maxiter, init=None)

    centers = cntr.T
    widths = np.zeros_like(centers)

    # compute widths per dimension and cluster
    for i in range(centers.shape[0]):  # each dimension
        Xi = data[:, i]  
        for k in range(num_mfs):
            u_k = u[k]  # membership of all samples to cluster k
            nume = np.sum((u_k**m) * (Xi - centers[i, k])**2)
            deno = np.sum(u_k**m)
            sigma = np.sqrt(nume/deno) if deno > 0 else 0.1
            widths[i, k] = max(sigma, 0.01)

    with torch.no_grad():
        model.centers[:] = torch.tensor(centers, dtype=torch.float32)
        model.widths[:]  = torch.tensor(widths,  dtype=torch.float32)


    return centers, widths

def weighted_sampler(X_train_t, y_train_t, y_train):
    class_sample_count = np.array([len(np.where(y_train == t)[0]) 
                                    for t in np.unique(y_train)])
    weight_per_class = 1.0 / class_sample_count
    sample_weights = np.array([weight_per_class[int(label)] for label in y_train])
    sample_weights = torch.from_numpy(sample_weights).float()
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # DataLoader with sampler
    train_dataset = TensorDataset(X_train_t, y_train_t)
    return DataLoader(train_dataset, batch_size=32, sampler=sampler)

def set_rule_subset(self, rule_indices):
    k = rule_indices.numel()
    self.rules     = self.all_rules[rule_indices]
    self.rule_idx  = self.rules.t()
    self.num_rules = k
    self.consequents = nn.Parameter(
        torch.ones(k, self.input_dim+1, self.num_classes),
        requires_grad=False
    )
    
def rule_stats(model, X, y):
    fire = model._forward_mf_only(X)           # [N,R]  bei POPFNN: model._fire
    cov  = fire.sum(0)                         # Coverage
    #print("Coverage per rule:", cov.cpu().numpy())
    # Klassenmatrix
    Y_onehot = F.one_hot(y, model.num_classes).float().to(fire)
    print("Y_onehot shape:", Y_onehot.shape)
    
    mass = fire.T @ Y_onehot                # [R,C]
    p_rc = mass / mass.sum(1, keepdim=True).clamp_min(1e-9)
    entropy = -(p_rc * p_rc.log()).sum(1) / np.log(model.num_classes)
    gini    = 1 - (p_rc**2).sum(1)

    # Mutual Information (Binary activation vs. Class)
    act = (fire > 0.1).float()                 # [N,R]
    mi = []
    for r in range(model.num_rules):
        mi.append( mutual_info_score(act[:,r].cpu(), y.cpu()) )
    mi = torch.tensor(mi, device=fire.device)

    return cov, entropy, mi

def rule_stats_wta(model, X, y, topk=1):
    """
    Winner-Takes-All-Statistik:
    - Sample wird jenen top-k Regeln gutgeschrieben, die am stärksten feuern.
    - Rückgabe: cov, H (Entropie)  [R]
    """
    fire = model._forward_mf_only(X)                       # [N,R]
    B, R = fire.shape
    top_val, top_idx = torch.topk(fire, k=topk, dim=1)   # [N,k]

    # 0-1 Indikator, welche Regel das Sample "besitzt"
    wta_mask = torch.zeros_like(fire).scatter_(1, top_idx, 1.)

    cov = wta_mask.sum(0).float()               # wie oft Regel Gewinner ist

    onehot = F.one_hot(y, model.num_classes).float().to(fire)
    mass   = wta_mask.T @ onehot               # [R,C] Gewinner‐Masse
    p_rc   = mass / mass.sum(1, keepdim=True).clamp_min(1e-9)
    H      = -(p_rc * p_rc.log()).sum(1) / np.log(model.num_classes)  # Entropie pro Regel
    return cov, H


def per_class_rule_scores(model, X, y, k_wta=3):
    """
    Liefert für jede Regel r:
       cov_r         – Gewinner-Coverage
       main_class_r  – Klasse mit größtem Anteil
       p_rc          – Reinheit der Hauptklasse
       score_rc      – cov * p
    """
    fire = model._forward_mf_only(X)                         # [N,R]
    top_idx = fire.topk(k_wta, dim=1).indices     # Gewinner-Regeln
    mask    = torch.zeros_like(fire).scatter_(1, top_idx, 1.)

    cov = mask.sum(0).float()                     # [R]

    onehot = F.one_hot(y, model.num_classes).float().to(fire)
    mass   = mask.T @ onehot                      # [R,C] Gewinner-Masse
    p_rc   = mass / mass.sum(1, keepdim=True).clamp_min(1e-9)

    main_cls = p_rc.argmax(1)                     # [R]
    purity   = p_rc.max(1).values                 # [R]

    score    = cov * purity                       # [R]  (= S_{r,main})
    return cov, purity, main_cls, score, p_rc



def create_weighted_sampler(X_np, y_np, batch_size, device="cpu"):
    y_np = np.asarray(y_np)                      # sicher numpy
    class_cnt = np.bincount(y_np)
    # vermeidet Division durch 0
    class_cnt[class_cnt == 0] = 1
    weights = 1.0 / class_cnt[y_np]

    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float32),
        num_samples=len(y_np),
        replacement=True)

    ds = TensorDataset(torch.as_tensor(X_np, dtype=torch.float32),
                       torch.as_tensor(y_np, dtype=torch.long))

    loader = DataLoader(ds,
                        batch_size=min(batch_size, len(ds)),
                        sampler=sampler,
                        drop_last=False)
    return loader


def _dbg(name, arr):
    """Lightweight diagnostic print for tensors / ndarrays."""
    import numpy as _np, torch as _torch
    if isinstance(arr, _torch.Tensor):
        nan = _torch.isnan(arr).sum().item()
        inf = _torch.isinf(arr).sum().item()
        arr_cpu = arr.detach().cpu()
        shape_str = str(tuple(arr.shape))
        print(f"[DBG] {name:<18} | shape {shape_str:>12} | "
              f"nan={nan:<4} inf={inf:<4} "
              f"min={arr_cpu.min().item(): .4g} max={arr_cpu.max().item(): .4g}")
    else:
        nan = _np.isnan(arr).sum()
        inf = _np.isinf(arr).sum()
        if hasattr(arr, 'size') and arr.size:
            shape_str = str(arr.shape)
            print(f"[DBG] {name:<18} | shape {shape_str:>12} | "
                  f"nan={nan:<4} inf={inf:<4} "
                  f"min={_np.nanmin(arr): .4g} max={_np.nanmax(arr): .4g}")
        else:
            print(f"[DBG] {name:<18} | EMPTY")
