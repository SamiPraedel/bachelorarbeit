import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import skfuzzy as fuzz
from torch import nn

def initialize_mfs_with_kmeans(model, data):
    """
    data: NumPy array of shape (num_samples, input_dim).
        Each column is one feature/dimension.

    Returns:
    centers: NumPy array of shape (input_dim, num_mfs)
    widths:  NumPy array of shape (input_dim, num_mfs)
    """
    
    input_dim = data.shape[1]
    num_mfs = model.num_mfs  # Anzahl der Membership Functions pro Dimension

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
    
    with torch.no_grad():
        model.centers[:] = torch.tensor(centers, dtype=torch.float32)
        model.widths[:]  = torch.tensor(widths,  dtype=torch.float32)

    print(f"K-Means-based centers:\n{centers}")
    print(f"K-Means-based widths:\n{widths}")

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
    num_mfs = model.num_mfs

    # run FCM
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        X, c=num_mfs, m=m, error=error, maxiter=maxiter, init=None)

    centers = cntr.T
    widths = np.zeros_like(centers)

    # compute widths per dimension and cluster
    for i in range(centers.shape[0]):  # each dimension
        Xi = data[:, i]  # now numpy array
        for k in range(num_mfs):
            u_k = u[k]  # membership of all samples to cluster k
            nume = np.sum((u_k**m) * (Xi - centers[i, k])**2)
            deno = np.sum(u_k**m)
            sigma = np.sqrt(nume/deno) if deno > 0 else 0.1
            widths[i, k] = max(sigma, 0.01)

    with torch.no_grad():
        model.centers[:] = torch.tensor(centers, dtype=torch.float32)
        model.widths[:]  = torch.tensor(widths,  dtype=torch.float32)

    print("FCM-based centers:\n", centers)
    print("FCM-based widths:\n", widths)
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
