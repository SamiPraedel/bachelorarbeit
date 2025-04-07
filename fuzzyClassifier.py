# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import umap.umap_ as umap
# import matplotlib.pyplot as plt

# # For FCM
# import skfuzzy as fuzz

# # For Iris dataset, splitting, and metrics
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from data_utils import load_K_chess_data, load_Kp_chess_data, load_wine_data

# import matplotlib.pyplot as plt


# ########################################################################
# # 1. Load the Iris dataset
# ########################################################################

# # # Iris has 150 samples, 4 features, and 3 classes (0, 1, 2)
# # iris = load_iris()
# # X_data = iris.data            # shape => [150, 4]
# # y_data = iris.target          # shape => [150]



# # # Optionally, shuffle and split into train/test
# # X_train, X_test, y_train, y_test = train_test_split(
# #     X_data, y_data, test_size=0.3, random_state=42
# # )

# # # Convert to PyTorch tensors
# # X_train_t = torch.from_numpy(X_train).float()  # [N_train, 4]
# # y_train_t = torch.from_numpy(y_train).long()   # [N_train]
# # X_test_t  = torch.from_numpy(X_test).float()   # [N_test, 4]
# # y_test_t  = torch.from_numpy(y_test).long()    # [N_test]

# X_train_t, y_train_t, X_test_t, y_test_t, X_train_np = load_K_chess_data()

# ########################################################################
# # 2. Apply FCM to cluster the training data
# ########################################################################

# # Number of fuzzy clusters (rules). 
# # For 4D data, you can experiment: e.g. 3, 5, 6, etc.
# n_clusters = 9

# # skfuzzy's cmeans expects data shape => [features, samples]
# X_train_np = X_train_np.T  # shape => [4, N_train]

# m = 2.0            # fuzziness exponent
# error = 1e-5
# maxiter = 1000
# init = None

# cntr, u, _, _, _, _, _ = fuzz.cmeans(
#     X_train_np, c=n_clusters, m=m, error=error, maxiter=maxiter, init=init
# )

# print("FCM cluster centers shape:", cntr.shape)
# print("FCM cluster centers:\n", cntr)
# # cntr => shape [n_clusters, 4]


# ########################################################################
# # 3. Estimate sigmas for each cluster
# ########################################################################
# sigmas = []
# for i in range(n_clusters):
#     # We'll pick points that have membership to cluster i > 0.5
#     mask = (u[i] > 0.5)
#     cluster_points = X_train_np.T[mask]  # shape => [N_sub, 4]
#     if len(cluster_points) < 2:
#         sigmas.append(1.0)  # fallback
#     else:
#         dists = np.linalg.norm(cluster_points - cntr[i], axis=1)
#         sigmas.append(np.mean(dists) + 1e-6)

# sigmas = np.array(sigmas, dtype=np.float32)
# print("Estimated sigmas:", sigmas)


# ########################################################################
# # 4. Define the TSK-like Fuzzy Classifier in PyTorch
# ########################################################################

# class FCMFuzzyClassifier(nn.Module):
#     def __init__(self, input_dim, n_clusters, n_classes, 
#                  init_centers, init_sigmas):
#         super().__init__()
#         self.n_clusters = n_clusters
#         self.n_classes = n_classes
#         self.input_dim = input_dim
        
#         # Cluster centers as parameters
#         self.centers = nn.Parameter(
#             torch.tensor(init_centers, dtype=torch.float32),
#             requires_grad=True
#         )  # shape => [n_clusters, input_dim]
        
#         # Sigmas as parameters (one sigma per cluster for multi-dim Gaussian)
#         # If you want separate sigmas per dimension => shape [n_clusters, input_dim]
#         # but we'll keep it simple: [n_clusters, 1]
#         self.sigmas = nn.Parameter(
#             torch.tensor(init_sigmas, dtype=torch.float32).view(-1,1),
#             requires_grad=True
#         )
        
#         # Each cluster => linear map from input_dim -> n_classes
#         self.rule_logits = nn.ModuleList([
#             nn.Linear(input_dim, n_classes) for _ in range(n_clusters)
#         ])
        
#     def forward(self, x):
#         """
#         x: [batch_size, input_dim]
#         returns: [batch_size, n_classes] (logits)
#         """
#         x_expanded = x.unsqueeze(1)                   # [B, 1, D]
#         c_expanded = self.centers.unsqueeze(0)        # [1, n_clusters, D]
#         diff = x_expanded - c_expanded                # [B, n_clusters, D]
        
#         dist_sq = torch.sum(diff**2, dim=2)           # [B, n_clusters]
#         sigma_sq = (self.sigmas**2).view(1, self.n_clusters)
        
#         mu = torch.exp(-dist_sq / (2.0 * sigma_sq))   # [B, n_clusters]
        
#         # Normalize firing strengths
#         mu_sum = torch.sum(mu, dim=1, keepdim=True) + 1e-9
#         w = mu / mu_sum  # [B, n_clusters]
        
#         # Get rule-based logits from each cluster
#         cluster_logits = []
#         for i in range(self.n_clusters):
#             out_i = self.rule_logits[i](x)  # [B, n_classes]
#             cluster_logits.append(out_i)
        
#         cluster_logits = torch.stack(cluster_logits, dim=1)  # [B, n_clusters, n_classes]
        
#         w_expanded = w.unsqueeze(2)                          # [B, n_clusters, 1]
#         logits = torch.sum(w_expanded * cluster_logits, dim=1)  # [B, n_classes]
        
#         return logits, w

# def plot_rule_firing_strengths_umap(model, X, w, rule_index=0):
#     """
#     1) Compute firing strengths for all rules. 
#     2) Apply UMAP to reduce X to 2D.
#     3) Scatter-plot X in 2D, colored by the firing strength for a specific rule.
#     """

    

   
#     w_np = w[:, rule_index].cpu().numpy()  # shape [N], membership for chosen rule
    
#     # 2) Apply UMAP on raw X data (convert to numpy if needed)
#     X_np = X.cpu().numpy()  # shape [N, input_dim]
#     reducer = umap.UMAP(n_components=2, random_state=42)
#     embedding = reducer.fit_transform(X_np)  # shape [N, 2]
    
#     # 3) Scatter-plot colored by firing strength
#     plt.figure(figsize=(8,6))
#     scatter = plt.scatter(embedding[:,0],
#                           embedding[:,1],
#                           c=w_np,
#                           cmap='viridis')
#     plt.colorbar(scatter, label=f'Firing Strength - Rule {rule_index}')
#     plt.title(f'UMAP of Input Data, colored by Rule #{rule_index} firing strength')
#     plt.xlabel('UMAP-1')
#     plt.ylabel('UMAP-2')
#     plt.show()


# ########################################################################
# # 5. Instantiate and train
# ########################################################################
# # input_dim = X_train.shape[1]  # = 4 for Iris
# # n_classes = len(np.unique(y_data))  # = 3 for Iris

# # model = FCMFuzzyClassifier(
# #     input_dim=input_dim, 
# #     n_clusters=n_clusters, 
# #     n_classes=n_classes, 
# #     init_centers=cntr, 
# #     init_sigmas=sigmas
# # )

# # criterion = nn.CrossEntropyLoss()
# # optimizer = optim.Adam(model.parameters(), lr=0.01)

# # n_epochs = 100
# # batch_size = 16

# # for epoch in range(n_epochs):
# #     # Shuffle
# #     perm = torch.randperm(X_train_t.size(0))
# #     X_train_t = X_train_t[perm]
# #     y_train_t = y_train_t[perm]
    
# #     epoch_loss = 0.0
# #     for i in range(0, X_train_t.size(0), batch_size):
# #         x_batch = X_train_t[i : i+batch_size]
# #         y_batch = y_train_t[i : i+batch_size]
        
# #         optimizer.zero_grad()
# #         logits = model(x_batch)
# #         loss = criterion(logits, y_batch)
        
# #         loss.backward()
# #         optimizer.step()
        
# #         epoch_loss += loss.item()
    
# #     epoch_loss /= (X_train_t.size(0) // batch_size)
    
# #     if (epoch+1) % 10 == 0:
# #         print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.4f}")


# input_dim  = 23  # (wk_file, wk_rank, wr_file, wr_rank, bk_file, bk_rank)
# n_classes  = 18 #len(np.unique(y_encoded))  # e.g. 2 if it's "won"/"draw"
# model      = FCMFuzzyClassifier(input_dim, n_clusters, n_classes, cntr, sigmas)

# criterion  = nn.CrossEntropyLoss()
# optimizer  = optim.Adam(model.parameters(), lr=0.01)

# n_epochs   = 1000
# batch_size = 32

# for epoch in range(n_epochs):
#     # Shuffle training data
#     perm = torch.randperm(X_train_t.size(0))
#     X_train_shuffled = X_train_t[perm]
#     y_train_shuffled = y_train_t[perm]
    
#     epoch_loss = 0.0
#     for i in range(0, X_train_shuffled.size(0), batch_size):
#         x_batch = X_train_shuffled[i : i+batch_size]
#         y_batch = y_train_shuffled[i : i+batch_size]
        
#         optimizer.zero_grad()
#         logits = model(x_batch)
#         loss = criterion(logits, y_batch)
#         loss.backward()
#         optimizer.step()
        
#         epoch_loss += loss.item()
    
#     epoch_loss /= (X_train_shuffled.size(0) // batch_size)
    
#     if (epoch+1) % 10 == 0:
#         print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {epoch_loss:.4f}")


# ########################################################################
# # 6. Evaluate on Test Data
# ########################################################################
# model.eval()
# with torch.no_grad():
#     logits_test = model(X_test_t)
#     preds_test = torch.argmax(logits_test, dim=1).numpy()
#     acc = accuracy_score(y_test_t, preds_test)
    
# print(f"Test Accuracy on Iris: {acc*100:.2f}%")

# # Plot firing strength for rule #0 on the training set:
# plot_rule_firing_strengths_umap(model, X_train_t, rule_index=0)

# # Maybe also rule #1 or others
# plot_rule_firing_strengths_umap(model, X_train_t, rule_index=1)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import umap.umap_ as umap
import matplotlib.pyplot as plt

# For FCM
import skfuzzy as fuzz

# For splitting and metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from data_utils import load_K_chess_data, load_Kp_chess_data, load_wine_data
#######################################################################
# 1) Load your data (here, the K-chess data) 
#######################################################################
X_train_t, y_train_t, X_test_t, y_test_t, X_train_np = load_K_chess_data()

#######################################################################
# 2) Fuzzy c-Means 
#######################################################################
n_clusters = 9

# scikit-fuzzy wants data as [features, samples]
X_train_np = X_train_np.T  
cntr, u, _, _, _, _, _ = fuzz.cmeans(
    X_train_np, c=n_clusters, m=2.0, error=1e-5, maxiter=1000, init=None
)
print("FCM cluster centers:\n", cntr)

# Estimate sigmas
sigmas = []
for i in range(n_clusters):
    mask = (u[i] > 0.5)
    cluster_points = X_train_np.T[mask]  # shape => [N_sub, features]
    if len(cluster_points) < 2:
        sigmas.append(1.0)
    else:
        dists = np.linalg.norm(cluster_points - cntr[i], axis=1)
        sigmas.append(np.mean(dists) + 1e-6)
sigmas = np.array(sigmas, dtype=np.float32)
print("sigmas:", sigmas)

#######################################################################
# 3) Define the TSK-like Fuzzy Classifier
#######################################################################
class FCMFuzzyClassifier(nn.Module):
    def __init__(self, input_dim, n_clusters, n_classes, init_centers, init_sigmas):
        super().__init__()
        self.n_clusters = n_clusters
        self.n_classes  = n_classes
        self.input_dim  = input_dim
        
        # cluster centers -> [n_clusters, input_dim]
        self.centers = nn.Parameter(
            torch.tensor(init_centers, dtype=torch.float32),
            requires_grad=True
        )
        # sigmas -> [n_clusters, 1]
        self.sigmas = nn.Parameter(
            torch.tensor(init_sigmas, dtype=torch.float32).view(-1,1),
            requires_grad=True
        )
        
        # Each cluster => linear map
        self.rule_logits = nn.ModuleList([
            nn.Linear(input_dim, n_classes) for _ in range(n_clusters)
        ])
        
    def forward(self, x):
        """
        returns:
          logits => [batch_size, n_classes]
          w => [batch_size, n_clusters], normalized firing strengths
        """
        B = x.size(0)
        x_expanded = x.unsqueeze(1)   # [B, 1, D]
        c_expanded = self.centers.unsqueeze(0)  # [1, n_clusters, D]
        
        diff = x_expanded - c_expanded          # [B, n_clusters, D]
        dist_sq = torch.sum(diff**2, dim=2)     # [B, n_clusters]
        
        sigma_sq = (self.sigmas**2).view(1, self.n_clusters)
        mu = torch.exp(-dist_sq / (2.0 * sigma_sq))  # [B, n_clusters]
        
        # normalize
        mu_sum = mu.sum(dim=1, keepdim=True) + 1e-9
        w = mu / mu_sum  # [B, n_clusters]
        
        # compute linear outputs from each cluster
        cluster_outs = []
        for i in range(self.n_clusters):
            out_i = self.rule_logits[i](x)  # [B, n_classes]
            cluster_outs.append(out_i)
        
        cluster_outs = torch.stack(cluster_outs, dim=1)  # [B, n_clusters, n_classes]
        
        w_expanded = w.unsqueeze(2)  # [B, n_clusters, 1]
        logits = torch.sum(w_expanded * cluster_outs, dim=1)  # => [B, n_classes]
        
        return logits, w

#######################################################################
# 4) Create model, train with CrossEntropy
#######################################################################
input_dim = 23
n_classes = 18
model = FCMFuzzyClassifier(input_dim, n_clusters, n_classes, cntr, sigmas)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

n_epochs   = 100
batch_size = 32

for epoch in range(n_epochs):
    # shuffle data each epoch
    perm = torch.randperm(X_train_t.size(0))
    X_train_shuffled = X_train_t[perm]
    y_train_shuffled = y_train_t[perm]
    
    epoch_loss = 0.0
    for i in range(0, X_train_shuffled.size(0), batch_size):
        x_batch = X_train_shuffled[i : i+batch_size]
        y_batch = y_train_shuffled[i : i+batch_size]
        
        optimizer.zero_grad()
        logits, w_batch = model(x_batch)  # forward pass
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    epoch_loss /= (X_train_shuffled.size(0) // batch_size)
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {epoch_loss:.4f}")

#######################################################################
# 5) Evaluate on Test Data
#######################################################################
model.eval()
with torch.no_grad():
    logits_test, w_test = model(X_test_t)
    preds_test = torch.argmax(logits_test, dim=1)
    test_acc   = accuracy_score(y_test_t.numpy(), preds_test.numpy())

print(f"Test Accuracy: {test_acc*100:.2f}%")


def plot_rule_firing_strengths_umap(X, w, rule_index=0):
    """
    X: [N, input_dim] torch tensor
    w: [N, n_clusters] torch tensor (firing strengths)
    rule_index: which cluster/rule to highlight
    """
    # 1) convert to numpy
    X_np = X.cpu().numpy()
    w_np = w[:, rule_index].cpu().numpy()
    
    # 2) UMAP => 2D embedding
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(X_np)  # shape [N, 2]
    
    # 3) Scatter color-coded by w_np
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(
        embedding[:,0],
        embedding[:,1],
        c=w_np,
        cmap='viridis'
    )
    plt.colorbar(scatter, label=f'Firing Strength - Rule {rule_index}')
    plt.title(f'UMAP of Input Data, colored by Rule #{rule_index} firing strength')
    plt.xlabel('UMAP-1')
    plt.ylabel('UMAP-2')
    plt.show()

# We'll do it on the training set (or test set if you prefer)
model.eval()
with torch.no_grad():
    _, w_train = model(X_train_t)

# Plot for rule #0
plot_rule_firing_strengths_umap(X_train_t, w_train, rule_index=0)

# Plot for rule #1, etc.
plot_rule_firing_strengths_umap(X_train_t, w_train, rule_index=1)