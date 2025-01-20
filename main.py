
import torch
from anfis_hybrid import HybridANFIS, train_hybrid_anfis
from data_utils import load_iris_data, load_heart_data

def run_experiment(dataset="iris", num_epochs=100, lr=1e-4, 
                   input_dim=4, num_classes=3, num_mfs=2, max_rules=50):
    # 1) Data load
    if dataset == "iris":
        X_train, y_train, X_test, y_test = load_iris_data()
    elif dataset == "heart":
        X_train, y_train, X_test, y_test = load_heart_data()
        input_dim = 13
        num_classes= 5
    elif dataset == "":
        X_train, y_train, X_test, y_test = load_iris_data()
    else:
        raise ValueError("Unknown dataset")
    






    

    # # 2) Modell
    # model = HybridANFIS(input_dim=input_dim, 
    #                     num_classes=num_classes,
    #                     num_mfs=num_mfs,
    #                     max_rules=max_rules)

    # # 3) Train
    # train_hybrid_anfis(model, X_train, y_train, num_epochs=num_epochs, lr=lr)

    # # 4) Test
    # model.eval()
    # with torch.no_grad():
    #     outputs, _, _ = model(X_test)
    #     preds = torch.argmax(outputs, dim=1)
    #     accuracy = (preds == y_test).float().mean().item() * 100
    # print(f"Test Accuracy on {dataset}: {accuracy:.2f}%")

if __name__ == "__main__":
   
   
    run_experiment(dataset="iris", num_epochs=100, lr=1e-3, 
                   input_dim=4, num_classes=3, num_mfs=2, max_rules=50)

  
    run_experiment(dataset="heart", num_epochs=100, lr=1e-4, 
                   input_dim=13, num_classes=5, num_mfs=2, max_rules=100)
