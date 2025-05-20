import numpy as np
import torch
import argparse

from torchmetrics.functional import matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import silhouette_score

from anfis_hybrid import HybridANFIS
from mlp_iris import FullyConnected, fit_mlp
from anfis_nonHyb import NoHybridANFIS
from trainAnfis import train_anfis_noHyb, train_anfis_hybrid
from data_utils import load_iris_data, load_heart_data, load_wine_data, load_abalon_data, load_Kp_chess_data, load_K_chess_data_splitted, load_K_chess_data_OneHot, load_Poker_data, load_Kp_chess_data_ord
from create_plots import plot_TopK, plot_sorted_Fs, plot_umap_fixed_rule, plot_sample_firing_strengths
from anfisHelper import weighted_sampler
# for training on cuda
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)


def run_experiment(
    type,
    dataset,
    num_epochs,
    num_mfs,
    max_rules,
    seed,
    lr=1e-3
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    DATASETS = {
        "iris":    (load_iris_data,3),
        "heart":   (load_heart_data,5),
        "wine":    (load_wine_data,11),
        "abalone": (load_abalon_data,3),
        "ChessK":  (load_K_chess_data_splitted,18),
        "ChessKp": (load_Kp_chess_data_ord,2),
        "Poker":   (load_Poker_data,10),
    }
    try:
        loader, num_classes = DATASETS[dataset]
    except KeyError:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    X_train, y_train, X_test, y_test, *rest = loader()

    if type == "anfis":
        X_train_np = X_train.numpy()
        X_train  = X_train.to(device)
        y_train  = y_train.to(device)
        X_test   = X_test.to(device)
        y_test   = y_test.to(device)
        
        model = HybridANFIS(
            input_dim=X_train.shape[1], 
            num_classes=num_classes,
            num_mfs=num_mfs,
            max_rules=max_rules,
            seed=seed
        )
        model.to(device)
        
        train_anfis_hybrid(model, X_train, y_train, num_epochs=num_epochs, lr=lr)
    elif type == "noHyb":

        X_train_t = X_train.clone().detach().float()
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_val_t   = torch.tensor(X_test,   dtype=torch.float32)

        model = NoHybridANFIS(
            input_dim=X_train.shape[1], 
            num_classes=num_classes,
            num_mfs=num_mfs,
            max_rules=max_rules,
            seed=seed,
            zeroG=False
        ).to(device)

        train_loader = weighted_sampler(X_train_t, y_train_t, y_train)
        train_anfis_noHyb(model, X_train_t, y_train_t, num_epochs=num_epochs, lr=lr, dataloader=train_loader)
            
        #plot_umap_fixed_rule(model, X_val_t, rule_index=10, cmap='viridis')
        #plot_sample_firing_strengths(model, X_val_t[1])

    elif type == "RandomF":
        model = RandomForestClassifier(n_estimators=25)
        model.fit(X_train,y_train)
    elif type == "neuralNet":
       model = FullyConnected(input_dim=X_train.shape[1], hidden1=30, hidden2=50, num_classes=num_classes)
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
                        
            mcc = matthews_corrcoef(preds, y_test,"multiclass",num_classes=num_classes)
            print("mcc: ", mcc)
            firing_strengths = firing_strengths.detach().cpu().numpy()  # [batch, num_rules]
            
            plot_sorted_Fs(preds, firing_strengths)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["anfis","noHyb","RandomF","neuralNet"], required=True)
    parser.add_argument("--dataset", choices=["iris","heart","ChessK", "ChessKp","abalone", "Poker"], required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_mfs", type=int, default=33)
    parser.add_argument("--max_rules", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=45)
    args = parser.parse_args()

    run_experiment(
        type=args.type,
        dataset=args.dataset,
        num_epochs=args.epochs,
        lr=args.lr,
        num_mfs=args.num_mfs,
        max_rules=args.max_rules,
        seed=args.seed,
    )

    

