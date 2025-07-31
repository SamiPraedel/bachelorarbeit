import numpy as np
import torch
from sklearn.semi_supervised import LabelPropagation
from sklearn.neighbors import kneighbors_graph, KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn import datasets
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from data_utils import load_K_chess_data_splitted, load_pmd_data
from sklearn.metrics import accuracy_score
from anfis_nonHyb import NoHybridANFIS
from trainAnfis import train_anfis_noHyb
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

def labelprop_on_features():
    
    kernel = 'knn'
    n_neighbors = 15
    max_iter = 1000
    tol = 1e-3
    
    
  

    
    Xtr, ytr, Xte, yte = load_K_chess_data_splitted()
    #Xtr, ytr, Xte, yte = load_pmd_data()
    
    rng = np.random.RandomState(42)
    random_unlabeled_points = rng.rand(len(ytr)) < 0.9
    
    labels = np.copy(ytr)
    labels[random_unlabeled_points] = -1
    X_l = Xtr[~random_unlabeled_points]
    y_l = ytr[~random_unlabeled_points]
    
    
    #label_prop_model = LabelPropagation(kernel=kernel, n_neighbors=n_neighbors, max_iter=max_iter, tol=tol)
    label_prop_model = LabelSpreading(
    kernel='rbf',        # oder 'rbf'
    n_neighbors=15,       # nur bei kernel='knn'
    gamma=20,            # nur bei kernel='rbf'
    alpha=0.8,           # Mischgewicht zwischen ursprünglichen und propagierten Labels
    max_iter=3000,       # mehr Iterationen für bessere Konvergenz
    tol=1e-3             # Toleranz für das Abbruchkriterium
    )
    label_prop_model.fit(Xtr, labels)
    y_pred = label_prop_model.predict(Xte)
    print(accuracy_score(yte, y_pred))
    
    
    
    
    
    anfis_model = NoHybridANFIS(
        input_dim=Xtr.shape[1],
        num_classes=len(ytr.unique()), num_mfs=4, max_rules=1000,
        seed=42
    ).to(device)
    

    train_anfis_noHyb(
        anfis_model,
        X_l.to(device), y_l.to(device),
        Xtr,
        num_epochs=500, lr=0.01
    )
    
 
    
    _, rule_activations_train, _ = anfis_model(Xtr.to(device))
    
    label_prop_model = LabelPropagation(kernel=kernel, n_neighbors=n_neighbors, max_iter=max_iter, tol=tol)
    rule_activations_train = rule_activations_train.cpu().detach().numpy()
    scaler = StandardScaler()
    rule_activations_train = scaler.fit_transform(rule_activations_train)

    label_prop_model.fit(rule_activations_train, labels)
    y_pred = label_prop_model.predict(rule_activations_train)
    print(accuracy_score(labels, y_pred))
    
    
    y_pred, _,_ = anfis_model(Xte.to(device))
    y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()
    print(accuracy_score(yte, y_pred))
    
    mf_values_train = anfis_model._fuzzify(Xtr.to(device))
    
    label_prop_model = LabelPropagation(kernel=kernel, n_neighbors=n_neighbors, max_iter=max_iter, tol=tol)
    mf_values_train = mf_values_train.view(mf_values_train.size(0), -1)
    mf_values_train = mf_values_train.cpu().detach().numpy()
    
    scaler = StandardScaler()
    mf_values_train = scaler.fit_transform(mf_values_train)
    
    label_prop_model.fit(mf_values_train, labels)
    y_pred = label_prop_model.predict(mf_values_train)
    print(accuracy_score(labels, y_pred))

    return label_prop_model.transduction_, label_prop_model

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    labelprop_on_features()