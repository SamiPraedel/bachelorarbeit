import torch
from create_plots import plot_TopK
from anfis_hybrid import HybridANFIS, train_hybrid_anfis
from mlp_iris import FullyConnected, fit_mlp
from anfis_nonHyb import NoHybridANFIS, train_anfis, plot_firing_strength_heatmap, plot_umap_fixed_rule, plot_sample_firing_strengths
from data_utils import load_iris_data, load_heart_data, load_wine_data, load_abalon_data, load_K_chess_data, load_Kp_chess_data, load_K_chess_data_splitted, load_K_chess_data_OneHot, load_Poker_data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import seaborn as sns
from torchmetrics.functional import matthews_corrcoef
from sklearn import metrics
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import silhouette_score

def run_experiment(
    type,
    dataset,
    num_epochs,
    input_dim,
    num_classes,
    num_mfs,
    max_rules,
    seed,
    lr=1e-3
):
    device = torch.device("mps")
    if dataset == "iris":
        X_train, y_train, X_test, y_test = load_iris_data()
        
    elif dataset == "heart":
        X_train, y_train, X_test, y_test = load_heart_data()
    elif dataset == "wine":
        X_train, y_train, X_test, y_test, mean = load_wine_data()
    elif dataset == "abalone":
        X_train, y_train, X_test, y_test = load_abalon_data()
    elif dataset == "ChessK":
        X_train, y_train, X_test, y_test = load_K_chess_data_splitted()
    elif dataset == "ChessKp":
        X_train, y_train, X_test, y_test, s = load_Kp_chess_data()
    elif dataset == "Poker":
        X_train, y_train, X_test, y_test = load_Poker_data()
    else:
        raise ValueError("Unknown dataset: " + dataset)

    if type == "anfis":
        X_train_np = X_train.cpu().numpy()
        print("Anf_hybr")
        X_train  = X_train.to(device)
        y_train  = y_train.to(device)
        X_test   = X_test.to(device)
        y_test   = y_test.to(device)
        
        
        model = HybridANFIS(
            input_dim=X_train.shape[1], 
            num_classes=y_train.shape[0],
            num_mfs=num_mfs,
            max_rules=max_rules,
            seed=seed
        )
        model.to(device)
        centers_kmeans, widths_kmeans = model.initialize_mfs_with_kmeans(X_train_np)
        centers_t = torch.from_numpy(centers_kmeans).to(device=device, dtype=torch.float32)
        widths_t  = torch.from_numpy(widths_kmeans).to(device=device, dtype=torch.float32)

        # # # # # Assign the centers/widths to the model’s parameters:
        with torch.no_grad():
            model.centers[:] = torch.tensor(centers_t, dtype=torch.float32)
            model.widths[:]  = torch.tensor(widths_t,  dtype=torch.float32)
            

        
        train_hybrid_anfis(model, X_train, y_train, num_epochs=num_epochs, lr=lr)

        

    elif type == "RandomF":
        print("RandomF")
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train,y_train)
    elif type == "noHyb":
        print("noHyb")
        X_train_np = X_train.cpu().numpy()
        X_train_t = X_train.clone().detach().float()
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_val_t   = torch.tensor(X_test,   dtype=torch.float32)

        model = NoHybridANFIS(
            input_dim=input_dim, 
            num_classes=num_classes,
            num_mfs=num_mfs,
            max_rules=max_rules,
            seed=seed,
            zeroG=False
        )

        #train_anfis(model, X_train, y_train, num_epochs=num_epochs, lr=lr)
        centers_k, widths_k = model.initialize_mfs_with_kmeans(X_train)  # X_train as np array
        with torch.no_grad():
           model.centers[:] = torch.tensor(centers_k, dtype=torch.float32)
           model.widths[:]  = torch.tensor(widths_k,  dtype=torch.float32)


        # WeightedRandomSampler example
        class_sample_count = np.array([len(np.where(y_train == t)[0]) 
                                     for t in np.unique(y_train)])
        weight_per_class = 1.0 / class_sample_count
        sample_weights = np.array([weight_per_class[int(label)] for label in y_train])
        sample_weights = torch.from_numpy(sample_weights).float()
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        # # DataLoader with sampler
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)

        # # Train
        train_anfis(model, X_train_t, y_train_t, num_epochs=num_epochs, lr=lr, dataloader=train_loader)

        # # Predict on validation
        with torch.no_grad():
            model.eval()
            outputs, _, _ = model(X_val_t)
            preds_val = torch.argmax(outputs, dim=1).cpu().numpy()
            
        
        plot_umap_fixed_rule(model, X_val_t, rule_index=10, cmap='viridis')
        sample = X_val_t[1]
        plot_sample_firing_strengths(model, sample)
    elif type == "neuralNet":
       model = FullyConnected(input_dim=input_dim, hidden1=30, hidden2=50, num_classes=num_classes)
       fit_mlp(model, X_train, y_train)

    else:
        raise ValueError("Unknown model type: " + type)

    # 3) Testphase
    
    with torch.no_grad():
        
        if type == "anfis" or type == "noHyb":
            model.eval()
            outputs, firing_strengths, _= model(X_test) 
            preds = torch.argmax(outputs, dim=1)
            preds_np   = preds.detach().cpu().numpy()
            y_test_np  = y_test.detach().cpu().numpy()
            accuracy = metrics.accuracy_score(y_test_np, preds_np) * 100
            print(f"Test Accuracy on {dataset} with {type}: {accuracy:.2f}%")
            mcc = matthews_corrcoef(preds, y_test,"multiclass",num_classes=int(torch.unique(y_test).numel()))
            
            print("mcc: ", mcc)
            firing_strengths = firing_strengths.detach().cpu().numpy()  # [batch, num_rules]
            
            sortedPreds, preds_ind,  = torch.sort(preds)
            
            #class_boundaries = np.nonzero(np.diff(sortedPreds))[0] + 1
            
            diffs = sortedPreds[1:] != sortedPreds[:-1]   # Boolean‑Mask, wo Klassen wechseln
            boundaries = torch.nonzero(diffs).squeeze() + 1 
            
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            sorted_Fs = firing_strengths[preds_ind]

            # 1) Transpose if needed
            sorted_Fs = sorted_Fs.T  # shape [num_rules, N]

            # 2) Plot
            im = ax.imshow(sorted_Fs, aspect='auto', cmap='viridis', vmin=0, vmax=0.01)

            # 3) Draw vertical class boundaries in data coords
            for boundary in boundaries[1:-1]:
                ax.axvline(boundary - 0.5, color='white', linewidth=0.5)
            
            plot_TopK(0.1, sorted_Fs)

            ax.set_xlabel("Testbeispiel Index (sortiert nach Klasse)")
            ax.set_ylabel("Regel-Index")

            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Firing Strength")

            plt.title("Heatmap der Regelfiring Strengths mit Klassengrenzen")
            plt.show()
            
            # silhouette_avg = silhouette_score(sorted_Fs.T, sortedPreds)
            # print("Silhouette Score:", silhouette_avg)

            
        elif type == "neuralNet":
            outputs = model(X_test)
            preds = torch.argmax(outputs, dim=1)
            accuracy = metrics.accuracy_score(y_test, preds)

        else:
            y_pred = model.predict(X_test)
            print('Accuracy: ', metrics.accuracy_score(y_test,y_pred))
            # mcc = matthews_corrcoef(torch.tensor(y_pred), y_test,"multiclass",num_classes=5)
            # print(mcc)



def get_important_rule_per_class_torch(firing_strengths, labels):


    unique_classes = torch.unique(labels)
    important_rules = {}
    
    # Für jede Klasse:
    for cls in unique_classes:
      
        indices = (labels == cls).nonzero(as_tuple=True)[0]
        
        # Berechne den Mittelwert der Firing Strengths für jede Regel für diese Instanzen:
        avg_strengths = torch.mean(firing_strengths[indices, :], dim=0)
        
        # Finde den Regelindex mit dem höchsten Mittelwert:
        important_rule_idx = torch.argmax(avg_strengths).item()
        
        # Speichere das Ergebnis im Dictionary
        important_rules[int(cls.item())] = important_rule_idx
        
    return important_rules



    

if __name__ == "__main__":

    """run_experiment(type="RandomF", dataset="ChessK", 
             num_epochs=200, lr=1e-3,
             input_dim=13, num_classes=5, num_mfs=2, max_rules=50, seed=50)"""

    """run_experiment(type="RandomF", dataset="Poker", 
                   num_epochs=200, lr=0.001, 
                   input_dim=4, num_classes=3, 
                   num_mfs=2, max_rules=50, seed=50)"""
    
    """run_experiment(type="anfis", dataset="iris", 
                num_epochs=200, lr=0.01, 
                input_dim=4, num_classes=3,
                num_mfs=2, max_rules=100, seed=47)"""
    
    """run_experiment(type="noHyb", dataset="ChessK", 
                num_epochs=100, lr=0.001, 
                input_dim=6, num_classes=18, 
                num_mfs=3, max_rules=729, seed=48)"""
    # ordinal, dim = 6, mfs = 8, epochs = 100, rules = 1000 --> 70%
    # one hot
    
    """run_experiment(type="noHyb", dataset="Poker", 
            num_epochs=10, lr=0.001, 
            input_dim=10, num_classes=10, 
            num_mfs=2, max_rules=1024, seed=48)"""
    
    run_experiment(type="noHyb", dataset="ChessKp", 
                num_epochs=300, lr=0.001, 
                input_dim=71, num_classes=2, 
                num_mfs=2, max_rules=100, seed=45)

    """run_experiment(type="noHyb", dataset="heart", 
                    num_epochs=10, lr=0.001,
                    input_dim=13, num_classes=5, num_mfs=2, max_rules=8192, seed=50)"""

    """run_experiment(type="anfis", dataset="wine", num_epochs=150,
                    input_dim=13, num_classes=5, num_mfs=2, max_rules=300, seed=50, lr=0.001)
    
    run_experiment(type="noHyb", dataset="abalone", num_epochs=100, lr=0.001, 
                input_dim=8, num_classes=3,num_mfs=2, max_rules=256, seed=50)"""
    
    """run_experiment(type="anfis", dataset="abalone", num_epochs=300, lr=0.001, 
                   input_dim=8, num_classes=3,num_mfs=2, max_rules=256, seed=50)"""
    

