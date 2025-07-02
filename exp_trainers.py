# trainers.py
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score

# Import model-specific classes and trainers for specialized handling
from anfis_hybrid import HybridANFIS
from PopFnn import POPFNN
from trainAnfis import train_anfis_hybrid, train_anfis_noHyb
from trainPF import train_popfnn
from anfis_nonHyb import NoHybridANFIS
# You will need to make your GraphSSL class available for import
# For example, by placing it in a file like `graph_ssl.py`
from lb_scratch import GraphSSL 

def _train_supervised_step(model, X_l, y_l, X_tr,device, lr=0.01, epochs=100, **kwargs):
    """
    A helper function to perform a standard supervised training run.
    This is a building block for more complex SSL trainers.
    It dispatches to model-specific trainers where necessary.
    """
    print(f"      Running initial supervised training for {epochs} epochs...")

    if isinstance(model, HybridANFIS):
        print("      Using specialized HybridANFIS trainer...")
        train_anfis_hybrid(model, X_l, y_l, X_tr,num_epochs=epochs, lr=lr)
    elif isinstance(model, POPFNN):
        print("      Using specialized POPFNN trainer...")
        train_popfnn(model, X_l, y_l, epochs=epochs, lr=lr)
    elif isinstance(model, NoHybridANFIS):
        print("      Using specialized NoHybridANFIS trainer...")
        # Assuming NoHybridANFIS has a specific training method
        train_anfis_noHyb(model, X_l, y_l, X_tr, num_epochs=epochs, lr=lr)
    else:
        # Generic trainer for end-to-end differentiable models like NoHybridANFIS.
        print("      Using generic supervised trainer...")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(X_l.to(device))
            logits = output[0] if isinstance(output, tuple) else output
            loss = F.cross_entropy(logits, y_l.to(device))
            loss.backward()
            optimizer.step()

    print("      Supervised training complete.")
    return model

def train_rulespace_ssl(
    model_instance,
    X_l, y_l, X_tr, y_tr, y_semi_sup, X_te,
    device,
    lr=0.01, epochs=500,
    ssl_method='grf', k=15, sigma=0.2, **kwargs
):
    # 1. Initial supervised training on labeled data
    model_instance = _train_supervised_step(model_instance, X_l, y_l, X_tr, device, lr=lr, epochs=epochs, **kwargs)

    # 2. Feature Extraction from the Rule Space
    print("      Extracting rule space features for GraphSSL...")
    model_instance.eval()
    with torch.no_grad():
        # Assumes model's forward pass returns: (logits, firing_strengths, other_params)
        _, rule_activations_train, _ = model_instance(X_tr.to(device))
        _, rule_activations_test, _ = model_instance(X_te.to(device))
    print(f"      Shape of rule space features: {rule_activations_train.shape}")

    # 3. Graph-based SSL fitting and prediction
    graph_ssl_model = GraphSSL(method=ssl_method, device=device, k=k, sigma=sigma)
    
    print(f"      Fitting GraphSSL model ({ssl_method.upper()})...")
    graph_ssl_model.fit(rule_activations_train, y_semi_sup)
    
    # 4. Evaluate pseudo-label accuracy on the training set (for diagnostics)
    unlabeled_mask = (y_semi_sup == -1)
    pseudo_label_acc = accuracy_score(y_tr.numpy()[unlabeled_mask], graph_ssl_model.transduction_[unlabeled_mask])
    print(f"      Diagnostic Pseudo-Label Accuracy: {pseudo_label_acc * 100:.2f}%")
    
    # 5. Predict on the test set's rule space and return the predictions
    y_pred_np = graph_ssl_model.predict(rule_activations_test)
    return y_pred_np


def train_ifgst(
    model_instance,
    X_l, y_l, X_u, X_te,
    device,
    num_rounds=5, epochs_per_round=15, confidence_threshold=0.95, **kwargs
):

    # Initial supervised training
    model_instance = _train_supervised_step(model_instance, X_l, y_l, device, epochs=200, **kwargs) # Warm-up

    # Start the iterative self-training loop
    for r in range(num_rounds):
        print(f"\n      --- Self-Training Round {r+1}/{num_rounds} ---")
        if len(X_u) == 0:
            print("      Unlabeled pool is empty. Stopping.")
            break

        # 1. Feature Extraction on all current data (labeled + unlabeled)
        X_all = torch.cat([X_l, X_u])
        model_instance.eval()
        with torch.no_grad():
            _, rule_activations_all, _ = model_instance(X_all.to(device))

        # 2. Pseudo-Label Generation with a GraphSSL teacher
        y_round_semi_sup = np.concatenate([y_l.numpy(), np.full(len(X_u), -1, dtype=np.int64)])
        ssl_teacher = GraphSSL(method='iterative', k=15, sigma=0.2, device=device)
        ssl_teacher.fit(rule_activations_all, y_round_semi_sup)

        # 3. Select high-confidence pseudo-labels
        confidences = torch.from_numpy(ssl_teacher.label_distributions_).max(1).values
        pseudo_labels = torch.from_numpy(ssl_teacher.transduction_)
        
        unlabeled_confidences = confidences[len(y_l):]
        unlabeled_pseudo_labels = pseudo_labels[len(y_l):]
        
        high_conf_mask = (unlabeled_confidences > confidence_threshold)
        
        if high_conf_mask.sum() == 0:
            print("      No new pseudo-labels passed the confidence threshold. Stopping.")
            break
            
        # 4. Augment the dataset
        X_new_pseudo = X_u[high_conf_mask]
        y_new_pseudo = unlabeled_pseudo_labels[high_conf_mask].long()
        
        print(f"      Adding {len(X_new_pseudo)} new high-confidence pseudo-labels.")
        X_l = torch.cat([X_l, X_new_pseudo])
        y_l = torch.cat([y_l, y_new_pseudo])
        X_u = X_u[~high_conf_mask]

        # 5. Retrain the student model on the augmented dataset
        model_instance = _train_supervised_step(model_instance, X_l, y_l, device, lr=0.005, epochs=epochs_per_round, **kwargs)

    print("\n--- IFGST training complete! ---")
    
    # Final prediction on the test set
    model_instance.eval()
    with torch.no_grad():
        output = model_instance(X_te.to(device))
        logits = output[0] if isinstance(output, tuple) else output
        y_pred_np = logits.argmax(dim=1).cpu().numpy()
        
    return y_pred_np
