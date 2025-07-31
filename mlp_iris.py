import torch
import torch.nn as nn
from sklearn import metrics
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.metrics import accuracy_score

from data_utils import load_iris_data, load_abalon_data, load_heart_data, load_Kp_chess_data, load_K_chess_data_splitted, load_htru_data, load_pen_data, load_block_data, load_letter_data


class FullyConnected(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, hidden3, num_classes):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden1)
        self.act1 = nn.ReLU()
        self.l2 = nn.Linear(hidden1, hidden2)
        self.act2 = nn.ReLU()
        self.l3 = nn.Linear(hidden2, hidden3)
        # self.act3 = nn.Softmax()
        # self.l4 = nn.Linear(hidden3, num_classes)

    def forward(self, x):
        x = self.act1(self.l1(x))
        x = self.act2(self.l2(x))
        #x = self.act3(self.l3(x))  # => [batch_size, num_classes]
        #x = self.l4(x)
        return x

def fit_mlp(model, X_train, y_train, epochs=500, lr=0.001):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    
    #X_train, y_train, X_test, y_test = load_K_chess_data_splitted()
    #X_train, y_train, X_test, y_test = load_pen_data()
    X_train, y_train, X_test, y_test = load_block_data()
    #X_train, y_train, X_test, y_test = load_letter_data()
    #X_train, y_train, X_test, y_test = load_abalon_data()
    #X_train, y_train, X_test, y_test = load_heart_data()
    #X_train, y_train, X_test, y_test = load_Kp_chess_data()
    #X_train, y_train, X_test, y_test = load_htru_data()
    print(y_train)
    
    n_labeled = int(0.1 * len(y_train))
    np.random.seed(42)
    labeled_indices = np.random.choice(np.arange(len(y_train)), size=n_labeled, replace=False)
    y_semi_sup = np.full(len(y_train), -1, dtype=np.int64)
    y_semi_sup[labeled_indices] = y_train[labeled_indices]
    
    X_l, y_l = X_train[labeled_indices], y_train[labeled_indices]
    print(X_l.shape)
    
    number_classes = torch.unique(y_train).size(0)

    
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(32, 48, 32, 16), random_state=1, max_iter=1000)
    clf.fit(X_l, y_l)
    preds = clf.predict(X_test)
    preds_np   = preds
    y_test_np  = y_test.detach().cpu().numpy()
    accuracy = metrics.accuracy_score(y_test_np, preds_np) * 100
    print(accuracy)
    
    unlabeled_mask = (y_semi_sup == -1)
    
    lp_raw_model = LabelPropagation(kernel='knn', n_neighbors=10, max_iter=1000)
    #lp_raw_model = LabelSpreading(kernel='knn', n_neighbors=30, alpha=0.3, max_iter=2000)
    #lp_raw_model = LabelPropagation(kernel='rbf', gamma=0.3, max_iter=1000)
    #lp_raw_model = LabelSpreading(kernel='rbf', gamma=0.3, max_iter=1000)
    lp_raw_model.fit(X_train.cpu().numpy(), y_semi_sup)
    pseudo_label_acc_raw = accuracy_score(y_train.numpy()[unlabeled_mask], lp_raw_model.transduction_[unlabeled_mask])
    print(f"  Pseudo-Label Accuracy: {pseudo_label_acc_raw * 100:.2f}%")
    y_pred_raw = lp_raw_model.predict(X_test)
    test_acc_raw = accuracy_score(y_test.numpy(), y_pred_raw)
    print(f"  Final Test Accuracy (Raw-Space): {test_acc_raw * 100:.2f}%")
    
    from sklearn.neighbors import NearestNeighbors
    import numpy as np

    X_labeled = X_train[labeled_indices]
    y_labeled = y_train[labeled_indices]

    y_labeled = np.array(y_labeled)  # oder y_labeled.numpy()

    X_labeled = X_train[labeled_indices]  # bleib bei NumPy-Arrays
    k = 10
    nn = NearestNeighbors(n_neighbors=k+1).fit(X_labeled)
    distances, neighbors = nn.kneighbors(X_labeled)

    agreements = []
    for i, nbrs in enumerate(neighbors):
        true_nbrs = nbrs[1:]
        same_label = np.sum(y_labeled[true_nbrs] == y_labeled[i])
        agreements.append(same_label / k)

    print("Durchschnittliche Nachbarschafts-Übereinstimmung:",
        np.round(np.mean(agreements), 3))