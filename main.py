import torch
from anfis_hybrid import HybridANFIS, train_hybrid_anfis, plot_firing_strengths
from mlp_iris import FullyConnected, fit_mlp
from anfis_nonHyb import NoHybridANFIS, train_anfis
from data_utils import load_iris_data, load_heart_data, load_wine_data, load_abalon_data, load_K_chess_data, load_Kp_chess_data
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


def run_experiment(
    type,
    dataset,
    num_epochs,
    input_dim,
    num_classes,
    num_mfs,
    max_rules,
    seed,
    lr=1e-4
):

    if dataset == "iris":
        X_train, y_train, X_test, y_test = load_iris_data()
    elif dataset == "heart":
        X_train, y_train, X_test, y_test = load_heart_data()
    elif dataset == "wine":
        X_train, y_train, X_test, y_test, mean = load_wine_data()
    elif dataset == "abalone":
        X_train, y_train, X_test, y_test = load_abalon_data()
    elif dataset == "ChessK":
        X_train, y_train, X_test, y_test = load_K_chess_data()
    elif dataset == "ChessKp":
        X_train, y_train, X_test, y_test = load_Kp_chess_data()
    else:
        raise ValueError("Unknown dataset: " + dataset)

    if type == "anfis":
        print("Anf_hybr")
        model = HybridANFIS(
            input_dim=X_train.shape[1], 
            num_classes=y_train.shape[0],
            num_mfs=num_mfs,
            max_rules=max_rules,
            seed=seed
        )
        #model.initialize_mfs_with_kmeans(X_train.numpy())
        
        train_hybrid_anfis(model, X_train, y_train, num_epochs=num_epochs, lr=lr)

        

    elif type == "RandomF":
        print("RandomF")
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train,y_train)
    elif type == "noHyb":
        print("noHyb")
        model = NoHybridANFIS(
            input_dim=input_dim, 
            num_classes=num_classes,
            num_mfs=num_mfs,
            max_rules=max_rules,
            seed=seed
        )
        train_anfis(model, X_train, y_train, num_epochs=num_epochs, lr=lr)

    elif type == "neuralNet":
       model = FullyConnected(input_dim=input_dim, hidden1=30, hidden2=50, num_classes=num_classes)
       fit_mlp(model, X_train, y_train)

    else:
        raise ValueError("Unknown model type: " + type)

    # 3) Testphase
    
    with torch.no_grad():
        
        if type == "anfis" or type == "noHyb":
            model.eval()
            outputs, firing_strengths, _ = model(X_test) 
            preds = torch.argmax(outputs, dim=1)
            accuracy = metrics.accuracy_score(y_test, preds) * 100
            print(f"Test Accuracy on {dataset} with {type}: {accuracy:.2f}%")
            mcc = matthews_corrcoef(preds, y_test,"multiclass",num_classes=y_test.shape[0])
            
            print("mcc: ", mcc)
            firing_strengths = firing_strengths.detach().cpu().numpy()  # [batch, num_rules]
            #plot_firing_strengths(model, X_test)
            

            plt.figure(figsize=(12, 8))
            sns.heatmap(firing_strengths.T, cmap='viridis', cbar=True)
            plt.title('Heatmap der Regelfiring Strengths')
            plt.xlabel('Testbeispiel Index')
            plt.ylabel('Regel')
            #plt.show()
        elif type == "neuralNet":
            outputs = model(X_test)
            preds = torch.argmax(outputs, dim=1)
            accuracy = metrics.accuracy_score(y_test, preds)

        else:
            y_pred = model.predict(X_test)
            print('Accuracy: ', metrics.accuracy_score(y_test,y_pred))
            # mcc = matthews_corrcoef(torch.tensor(y_pred), y_test,"multiclass",num_classes=5)
            # print(mcc)
        
    
        

if __name__ == "__main__":

    run_experiment(type="RandomF", dataset="ChessKp", 
             num_epochs=200, lr=1e-3,
             input_dim=13, num_classes=5, num_mfs=2, max_rules=50, seed=50)

    """run_experiment(type="RandomF", dataset="abalone", 
                   num_epochs=200, lr=0.001, 
                   input_dim=4, num_classes=3, 
                   num_mfs=2, max_rules=50, seed=50)"""
    
    """run_experiment(type="anfis", dataset="iris", 
                num_epochs=100, lr=0.0001, 
                input_dim=4, num_classes=3, 
                num_mfs=2, max_rules=100, seed=47)"""
    """run_experiment(type="noHyb", dataset="ChessK", 
                num_epochs=100, lr=0.001, 
                input_dim=6, num_classes=18, 
                num_mfs=2, max_rules=100, seed=45)"""
    
    run_experiment(type="noHyb", dataset="ChessKp", 
                num_epochs=100, lr=0.001, 
                input_dim=36, num_classes=2, 
                num_mfs=2, max_rules=40, seed=45)

    """run_experiment(type="anfis", dataset="heart", 
                    num_epochs=100, lr=0.01,
                    input_dim=13, num_classes=5, num_mfs=2, max_rules=200, seed=50)

    run_experiment(type="anfis", dataset="wine", num_epochs=150,
                    input_dim=13, num_classes=5, num_mfs=2, max_rules=300, seed=50, lr=0.001)
    
    run_experiment(type="noHyb", dataset="abalone", num_epochs=100, lr=0.001, 
                input_dim=8, num_classes=3,num_mfs=2, max_rules=256, seed=50)"""
    
    """run_experiment(type="anfis", dataset="abalone", num_epochs=100, lr=0.001, 
                   input_dim=8, num_classes=3,num_mfs=2, max_rules=256, seed=50)"""
    

