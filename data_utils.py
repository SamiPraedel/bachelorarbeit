# data_utils.py

import torch
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OrdinalEncoder
from ucimlrepo import fetch_ucirepo
import pandas as pd
from scipy import stats
from data.openD import loadK, loadKP, loadPoker
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.cluster import KMeans

def load_Poker_data(test_size=0.3, random_state=42):
    # fetch dataset 
    poker_hand = fetch_ucirepo(id=158) 
    
    # data (as pandas dataframes) 
    X = poker_hand.data.features 
    y = poker_hand.data.targets 
    
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(X)
    
    if isinstance(y, pd.DataFrame):
        y = y.to_numpy().squeeze()
    
    # Aufteilen in Trainings- und Testdaten
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        x_scaled, y, test_size=test_size, shuffle=True, random_state=random_state
    )
    
    # Eventuelle NaN-Werte ersetzen
    X_train_np = np.nan_to_num(X_train_np, nan=0.0)
    X_test_np  = np.nan_to_num(X_test_np,  nan=0.0)
    
    # Konvertiere in PyTorch-Tensoren
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    X_test  = torch.tensor(X_test_np, dtype=torch.float32)
    y_test  = torch.tensor(y_test_np, dtype=torch.long)
    
    #print("y_train shape:", y_train.shape)
    
    return X_train, y_train, X_test, y_test

def load_K_chess_data_splitted(test_size=0.3, random_state=42):
    df = loadK()
    #     # Annahme: Die letzte Spalte ist die Zielvariable
    #    # Features und Label trennen
    df.rename(columns={
        'V1': 'white-king-file',
        'V2': 'white-rook-file',
        'V3': 'black-king-file'
    }, inplace=True)
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    chess_king_rook_vs_king = fetch_ucirepo(id=23) 

    
    # # data (as pandas dataframes) 
    X = chess_king_rook_vs_king.data.features 
    y = chess_king_rook_vs_king.data.targets 
    
    encoder = OrdinalEncoder()
    
    xWkf = np.array(X['white-king-file']).reshape(-1,1)
    xWrf = np.array(X['white-rook-file']).reshape(-1,1)
    xBkf = np.array(X['black-king-file']).reshape(-1,1)
    
    

    X.loc[:, 'white-king-file'] = encoder.fit_transform(xWkf)
    X.loc[:, 'white-rook-file'] = encoder.fit_transform(xWrf)
    X.loc[:, 'black-king-file'] = encoder.fit_transform(xBkf)

    # Numerische Features skalieren
    scaler = MinMaxScaler()
    X_processed = scaler.fit_transform(X)
    
    X_processed = X_processed.astype(np.float32)

    # Label kodieren (falls nötig)
    le = LabelEncoder()
    #y_encoded = le.fit_transform(y)
    y_encoded = le.fit_transform(y.values.ravel())

    
    # Aufteilen in Trainings- und Testdaten
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_processed, y_encoded, test_size=test_size, shuffle=True, random_state=random_state
    )
    
    # Eventuelle NaN-Werte ersetzen
    X_train_np = np.nan_to_num(X_train_np, nan=0.0)
    X_test_np  = np.nan_to_num(X_test_np,  nan=0.0)
    
    # Konvertiere in PyTorch-Tensoren
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    X_test  = torch.tensor(X_test_np, dtype=torch.float32)
    y_test  = torch.tensor(y_test_np, dtype=torch.long)
    
    #print("y_train shape:", y_train.shape)
    
    return X_train, y_train, X_test, y_test


def load_K_chess_data_OneHot(test_size=0.3, random_state=42):
    # df = loadK()
    #     # Annahme: Die letzte Spalte ist die Zielvariable
    #    # Features und Label trennen
    # X = df.drop(columns=['Class'])
    # y = df['Class']
    
    chess_king_rook_vs_king = fetch_ucirepo(id=23) 

    
    # # data (as pandas dataframes) 
    X = chess_king_rook_vs_king.data.features 
    y = chess_king_rook_vs_king.data.targets 
    
    # Numerische und kategoriale Spalten bestimmen
    numeric_cols = X.select_dtypes(include=['float64', 'int']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    # Numerische Features skalieren
    scaler = MinMaxScaler()
    if len(numeric_cols) > 0:
        X_numeric = pd.DataFrame(scaler.fit_transform(X[numeric_cols]),
                                 columns=numeric_cols,
                                 index=X.index)
    else:
        X_numeric = pd.DataFrame(index=X.index)
    
    # Kategoriale Features mittels One-Hot-Encoding umwandeln
    if len(categorical_cols) > 0:
        X_categorical = pd.get_dummies(X[categorical_cols])
    else:
        X_categorical = pd.DataFrame(index=X.index)
    
    # Kombiniere numerische und kategoriale Features
    X_processed = pd.concat([X_numeric, X_categorical], axis=1)
    
    X_processed = X_processed.astype(np.float32)

    # Label kodieren (falls nötig)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Aufteilen in Trainings- und Testdaten
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_processed, y_encoded, test_size=test_size, shuffle=True, random_state=random_state
    )
    
    # Eventuelle NaN-Werte ersetzen
    X_train_np = np.nan_to_num(X_train_np, nan=0.0)
    X_test_np  = np.nan_to_num(X_test_np,  nan=0.0)
    
    # Konvertiere in PyTorch-Tensoren
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    X_test  = torch.tensor(X_test_np, dtype=torch.float32)
    y_test  = torch.tensor(y_test_np, dtype=torch.long)
    
    #print("y_train shape:", y_train.shape)
    
    return X_train, y_train, X_test, y_test

def load_K_chess_data(test_size=0.3, random_state=42):
    # df = loadK()

    # X = df.drop(columns=['Class'])
    # y = df['Class']
    # fetch dataset 
    chess_king_rook_vs_king = fetch_ucirepo(id=23) 

    
    # data (as pandas dataframes) 
    X = chess_king_rook_vs_king.data.features 
    y = chess_king_rook_vs_king.data.targets 
    
    # Numerische und kategoriale Spalten bestimmen
    numeric_cols = X.select_dtypes(include=['float64', 'int']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    # Aufteilen in Trainings- und Testdaten
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X, y, test_size=test_size, shuffle=True, random_state=random_state
    )
    
    # Numerische Features skalieren
    scaler = MinMaxScaler()

    X_train_np = scaler.fit_transform(X_train_np)
    X_test_np  = scaler.transform(X_test_np)
    

    
    # Eventuelle NaN-Werte ersetzen
    X_train_np = np.nan_to_num(X_train_np, nan=0.0)
    X_test_np  = np.nan_to_num(X_test_np,  nan=0.0)
    
    # Konvertiere in PyTorch-Tensoren
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    X_test  = torch.tensor(X_test_np, dtype=torch.float32)
    y_test  = torch.tensor(y_test_np, dtype=torch.long)
    
    #print("X_train shape:", X_train.shape)
    
    return X_train, y_train, X_test, y_test, X_train_np, X, y

def get_chessK_full():
    """
    Returns the *entire* ChessK dataset in NumPy form: (X_np, y_np).
    No train/test split here, so cross-validation can do its own splits.
    """
    chess_king_rook_vs_king = fetch_ucirepo(id=23)
    X_df = chess_king_rook_vs_king.data.features
    y_df = chess_king_rook_vs_king.data.targets

    # Example: do numeric scaling if needed
    numeric_cols = X_df.select_dtypes(include=['float64', 'int']).columns
    if len(numeric_cols) > 0:
        scaler = MinMaxScaler()
        X_df[numeric_cols] = scaler.fit_transform(X_df[numeric_cols])

    cat_cols = X_df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        X_df = pd.get_dummies(X_df, columns=cat_cols)

    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_df)
    print(X_df.shape, "jkl")


    # Convert to numpy
    X_np = X_df.to_numpy(dtype=np.float32)
    # If y_df is a Series, convert to int array
    y_np = y_encoded.astype(int)

    return X_np, y_np

def get_iris_full():
    data = load_iris()
    X_np = data['data'].astype(np.float32)
    y_np = data['target'].astype(int)

    # Possibly scale
    scaler = MinMaxScaler()
    X_np = scaler.fit_transform(X_np)

    return X_np, y_np



def get_chessKp_full():
    chess_king_rook_vs_king_pawn = fetch_ucirepo(id=22)
    X_df = chess_king_rook_vs_king_pawn.data.features
    y_df = chess_king_rook_vs_king_pawn.data.targets

    numeric_cols = X_df.select_dtypes(include=['float64', 'int']).columns
    if len(numeric_cols) > 0:
        scaler = MinMaxScaler()
        X_df[numeric_cols] = scaler.fit_transform(X_df[numeric_cols])

    cat_cols = X_df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        X_df = pd.get_dummies(X_df, columns=cat_cols)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_df)

    

    X_np = X_df.to_numpy(dtype=np.float32)
    if not (isinstance(y_encoded, np.ndarray)):
        y_np = y_encoded.to_numpy(dtype=int)
    else:
        y_np = y_encoded
    
    return X_np, y_np



def load_Kp_chess_data(test_size=0.3, random_state=42):
    # df = loadKP()
    #     # Annahme: Die letzte Spalte ist die Zielvariable
    #    # Features und Label trennen
    # X = df.drop(columns=['Class'])
    # y = df['Class']

    # fetch dataset 
    chess_king_rook_vs_king_pawn = fetch_ucirepo(id=22) 
    
    # data (as pandas dataframes) 
    X = chess_king_rook_vs_king_pawn.data.features 
    y = chess_king_rook_vs_king_pawn.data.targets 
    
    # Numerische und kategoriale Spalten bestimmen
    numeric_cols = X.select_dtypes(include=['float64', 'int']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    # Numerische Features skalieren
    scaler = MinMaxScaler()
    if len(numeric_cols) > 0:
        X_numeric = pd.DataFrame(scaler.fit_transform(X[numeric_cols]),
                                 columns=numeric_cols,
                                 index=X.index)
    else:
        X_numeric = pd.DataFrame(index=X.index)
    
    # Kategoriale Features mittels One-Hot-Encoding umwandeln
    if len(categorical_cols) > 0:
        X_categorical = pd.get_dummies(X[categorical_cols])
    else:
        X_categorical = pd.DataFrame(index=X.index)
    
    # Kombiniere numerische und kategoriale Features
    X_processed = pd.concat([X_numeric, X_categorical], axis=1)
    
    X_processed = X_processed.astype(np.float32)

    # Label kodieren (falls nötig)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Aufteilen in Trainings- und Testdaten
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_processed, y_encoded, test_size=test_size, shuffle=True, random_state=random_state
    )
    
    # Eventuelle NaN-Werte ersetzen
    X_train_np = np.nan_to_num(X_train_np, nan=0.0)
    X_test_np  = np.nan_to_num(X_test_np,  nan=0.0)
    
    # Konvertiere in PyTorch-Tensoren
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    X_test  = torch.tensor(X_test_np, dtype=torch.float32)
    y_test  = torch.tensor(y_test_np, dtype=torch.long)
    
    print("X_train shape:", X_train.shape)
    
    return X_train, y_train, X_test, y_test, X_train_np

def load_heart_data(test_size=0.3, random_state=42):
    heart = fetch_ucirepo(id=45)
    X_df = heart.data.features
    y_df = heart.data.targets.squeeze()
    

    # => Split
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_df, y_df, test_size=test_size, shuffle=True, random_state=random_state
    )
    # NaN -> 0
    X_train_np = np.nan_to_num(X_train_np, nan=0.0)
    X_test_np  = np.nan_to_num(X_test_np,  nan=0.0)

    # scale
    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train_np)
    X_test_np  = scaler.transform(X_test_np)

    # to torch
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np.to_numpy(), dtype=torch.long)
    X_test  = torch.tensor(X_test_np,  dtype=torch.float32)
    y_test  = torch.tensor(y_test_np.to_numpy(),  dtype=torch.long)

    return X_train, y_train, X_test, y_test

def load_wine_data(test_size = 0.2, random_state = 42): 
    # fetch dataset 
    wine_quality = fetch_ucirepo(id=186) 
    
    # data (as pandas dataframes) 
    X_df = wine_quality.data.features 
    y_df = wine_quality.data.targets 

    df = pd.DataFrame(X_df, columns=wine_quality.feature_names)
    df['quality'] = y_df

    z = np.abs(stats.zscore(df))

    df_o = df[(z < 3).all(axis=1)]

    X = df_o.drop(columns = 'quality')
    y = df_o['quality']
    mean = X.mean()


    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    
    label_map = {4:0, 5:1, 6:2, 7:3, 8:4}
    y_mapped = y_train.map(label_map)

    return X_train, y_mapped, X_test, y_test, mean

def load_abalon_data():  
    # fetch dataset 
    abalone = fetch_ucirepo(id=1) 
    

    
    # data (as pandas dataframes) 
    X = abalone.data.features 
    y = abalone.data.targets 
    print(X.head(10))

    X["Target Variable"] = y  # "Target Variable" ist der Name für y
    print(X.head(10))
    encodings = X.groupby('Sex')['Target Variable'].mean().reset_index()
    encoding_mapping = encodings.set_index('Sex')['Target Variable'].to_dict()
    X['Sex_target_encoded'] = X['Sex'].map(encoding_mapping)
    print(X.head(10))

    X["AgeGroup"] = pd.cut(X["Target Variable"], bins=[0, 8, 11, np.inf], labels=[0, 1, 2])
    X["AgeGroup"] = X["AgeGroup"].astype(int)
    y = X["AgeGroup"]
    
    #X = X.merge(encodings, how='left', on='Sex')
    X = X.drop('Sex', axis=1)
    X = X.drop('Target Variable', axis=1)
    X = X.drop('AgeGroup', axis = 1)
    print(X.head(10))
    print(y.head(10))
    

    # Normalisieren
    #scaler = StandardScaler()
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-Test-Split (stratifiziert)
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_scaled, y, test_size=0.2, random_state=8, stratify=y
    )

    # Umwandlung in Torch-Tensoren
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np.values, dtype=torch.long)
    #print(X_train.shape)

    return X_train, y_train, X_test, y_test, X_train_np




def load_iris_data(test_size=0.2, random_state=42):
    iris = load_iris()
    X_iris = iris.data
    y_iris = iris.target

    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_iris, y_iris, test_size=test_size, shuffle=True, random_state=random_state
    )
    # Optionale NaN-Behandlung
    X_train_np = np.nan_to_num(X_train_np, nan=0.0)
    X_test_np  = np.nan_to_num(X_test_np,  nan=0.0)

    scaler = MinMaxScaler()
    X_train_np = scaler.fit_transform(X_train_np)
    X_test_np  = scaler.transform(X_test_np)

    # PyTorch Tensor
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    X_test  = torch.tensor(X_test_np,  dtype=torch.float32)
    y_test  = torch.tensor(y_test_np,  dtype=torch.long)

    return X_train, y_train, X_test, y_test


def show_ChessK():
    output_dir = 'plots'
    chess_king_rook_vs_king = fetch_ucirepo(id=23)  
    X = chess_king_rook_vs_king.data.features 
    y = chess_king_rook_vs_king.data.targets 

    for col in X.features:
        plt.figure(figsize=(8,6))
        #for cls in y.unice()
        
        

if __name__ == "__main__":
    X_train1, y_train, X_test, y_test = load_K_chess_data_splitted()
    X_train2, y_train, X_test, y_test = load_K_chess_data_OneHot()
    X_train3, y_train, X_test, y_test = load_iris_data()
    X_train4, y_train, X_test, y_test = load_heart_data()
    X_train5, y_train, X_test, y_test, s = load_abalon_data()
    X_train6, y_train, X_test, y_test, s = load_Kp_chess_data()
    X_train7, y_train, X_test, y_test, = load_Poker_data()
    
    data_list = [X_train1, X_train2, X_train3, X_train4, X_train5, X_train6, X_train7]
    
    from sklearn.cluster import DBSCAN
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    
    for X_train in data_list:
        
        
        kmax = 8
        # X_train in NumPy umwandeln
        X_train_np = X_train.detach().cpu().numpy()
        n_samples, n_features = X_train_np.shape
        wss_all_features = {}  # Dictionary: key = Feature-Index, value = Liste der WSS für k=1 bis kmax

        # Für jedes Feature separat:
        for feature_idx in range(n_features):
            # Daten des aktuellen Features als 2D-Array: (n_samples, 1)
            data = X_train_np[:, feature_idx].reshape(-1, 1)
            wss_feature = []
            # k von 1 bis kmax
            for k in range(1, kmax + 1):
                km = KMeans(n_clusters=k, random_state=0, n_init='auto')
                km.fit(data)
                # km.inertia_ liefert die Summe der quadratischen Abstände (WSS)
                wss_feature.append(km.inertia_)
            wss_all_features[f"feature_{feature_idx}"] = wss_feature
            
            k_values = list(range(1, kmax + 1))
            for feature, wss in wss_all_features.items():
                print(f"{feature}: {wss}")
                plt.plot(k_values, wss, marker='o', label=feature)
            
            plt.xlabel("Anzahl der Cluster (k)")
            plt.ylabel("WSS (Within-Cluster Sum of Squares)")
            plt.title("Elbow Plot pro Feature")
            plt.legend()
            plt.grid(True)
            plt.show()
                
                # # Fit the DBSCAN model to the data
                # dbscan = DBSCAN(eps=0.3, min_samples=5)
                # clusters = dbscan.fit_predict(data.reshape(-1, 1))

                # # Get the cluster assignments for each data point
                # print("------------")
                # print(clusters)
                
                # # Perform hierarchical clustering
                # Z = linkage(data, method='ward')

                # # Create a dendrogram
                # dendrogram(Z)

                # # Determine the clusters by cutting the dendrogram at a threshold
                # clusters = fcluster(Z, t=1, criterion='distance')

                # # Print the cluster assignments
                # print(clusters)

                # # Show the dendrogram
                # plt.show()
        
    
        


    


    

    

    

       
       
