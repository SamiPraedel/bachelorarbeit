# data_utils.py

import torch
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OrdinalEncoder, OneHotEncoder
from ucimlrepo import fetch_ucirepo
import pandas as pd
from scipy import stats
from data.openD import loadK, loadKP, loadPoker
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.cluster import KMeans
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import umap.umap_ as umap

# poker_loader.py
# ----------------------------------------------------------
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, WeightedRandomSampler
from ucimlrepo import fetch_ucirepo        # <- falls du uciml-repo nutzt

def load_poker_data(test_size=0.3, max_samples=90000,
                    scale_numeric=True, random_state=42):

    # ---------- 1) Laden + optional begrenzen ----------------------------
    ds  = fetch_ucirepo(id=158)
    Xdf = ds.data.features.copy()
    y   = ds.data.targets.squeeze().to_numpy()

    if len(Xdf) > max_samples:
        Xdf, _, y, _ = train_test_split(
            Xdf, y, train_size=max_samples, stratify=y, # y here is the original full y
            random_state=random_state)
        # After this, Xdf and y are the sampled versions.

    # Filter out classes with fewer than 2 instances from the (potentially sampled) y
    # This is done BEFORE the main train/test split to prevent the ValueError
    unique_classes_in_y, counts_in_y = np.unique(y, return_counts=True)
    classes_to_remove = unique_classes_in_y[counts_in_y < 2]

    if len(classes_to_remove) > 0:
        counts_of_removed_classes = counts_in_y[counts_in_y < 2]
        print(f"Warning: Poker dataset (after sampling to {max_samples} if active): "
              f"Removing samples from classes with < 2 instances. "
              f"Classes removed: {classes_to_remove}. "
              f"Their counts were: {counts_of_removed_classes}.")
        
        # Create a boolean mask to keep only samples from classes with >= 2 instances
        keep_mask = ~np.isin(y, classes_to_remove)
        
        Xdf = Xdf[keep_mask]
        y = y[keep_mask]

        if len(y) == 0:
            raise ValueError("Poker dataset: After removing classes with < 2 instances, no data remains.")
        
        # Update unique_classes_in_y after filtering for the next check
        unique_classes_in_y = np.unique(y) 
        if len(unique_classes_in_y) < 2 and len(y) > 0:
            print(f"Warning: Poker dataset: After filtering, only {len(unique_classes_in_y)} class(es) remain. "
                  "Stratification in the final train/test split will be disabled.")

    # ---------- 2) Skalieren (nur Rang-Spalten) --------------------------
    if scale_numeric:
        # Correctly identify rank columns (e.g., C1, C2, ..., C5 for Poker Hand dataset)
        # These columns represent the rank of the cards.
        rank_cols = [col for col in Xdf.columns if col.startswith('C')]
        if rank_cols: # Ensure rank_cols is not empty before attempting to scale
            scaler = MinMaxScaler()
            Xdf[rank_cols] = scaler.fit_transform(Xdf[rank_cols])
        else:
            print("Warning: No rank columns (starting with 'C') found for scaling in Poker dataset. Skipping scaling of rank columns.")

    X_np = Xdf.to_numpy(dtype=np.float32)

    # ---------- 3) Train/Test-Split -------------------------------------
    # The 'y' here is now the filtered y.
    if len(y) == 0: # Should be caught by the earlier check, but as a safeguard
        raise ValueError("Poker dataset: 'y' is empty before final train/test split.")
    
    can_stratify = len(np.unique(y)) >= 2

    if can_stratify:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_np, y, test_size=test_size,
            stratify=y, random_state=random_state)
    else: # Not enough unique classes to stratify (e.g., only 0 or 1 class left)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_np, y, test_size=test_size, random_state=random_state) # No stratify

    # ---------- 5) Torch-Tensoren ---------------------------------------
    X_tr = torch.as_tensor(X_tr, dtype=torch.float32)
    y_tr = torch.as_tensor(y_tr, dtype=torch.long)

    X_te = torch.as_tensor(X_te, dtype=torch.float32)
    y_te = torch.as_tensor(y_te, dtype=torch.long)

    return X_tr, y_tr, X_te, y_te

def load_shuttle_data(test_size=0.3, random_state=42):
  
    # fetch dataset 
    statlog_shuttle = fetch_ucirepo(id=148) 
    
    # data (as pandas dataframes) 
    X_df = statlog_shuttle.data.features 
    y_df = statlog_shuttle.data.targets # y_df is a DataFrame
     
    scaler = MinMaxScaler()
    # X_processed_np will be a NumPy array
    X_processed_np = scaler.fit_transform(X_df) 
    
    # Ensure X_processed_np is float32
    X_processed_np = X_processed_np.astype(np.float32)
    
    # Convert y_df (DataFrame) to a 1D NumPy array for y_for_split_np
    y_for_split_np = y_df.to_numpy().squeeze()
    # Convert 1-indexed labels (1-7) to 0-indexed labels (0-6)
    y_for_split_np = y_for_split_np - 1
    
    # Now, X_processed_np and y_for_split_np are both NumPy arrays.
    # So, y_train_np and y_test_np will also be NumPy arrays.
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_processed_np, y_for_split_np, 
        test_size=test_size, 
        shuffle=True, 
        random_state=random_state,
        stratify=y_for_split_np # Stratification is good for imbalanced datasets
    )
    
    # Convert NumPy arrays to PyTorch tensors
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.long) # This will now work
    X_test  = torch.tensor(X_test_np, dtype=torch.float32)
    y_test  = torch.tensor(y_test_np, dtype=torch.long)  # This will now work
    
    return X_train, y_train, X_test, y_test


    
    
    
    



def load_K_chess_data_splitted(test_size=0.3, random_state=42):
    
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
    
    chess_king_rook_vs_king = fetch_ucirepo(id=23) 

    

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


def load_Kp_chess_data(test_size=0.3, random_state=42):
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
    
    #print("X_train shape:", X_train.shape)
    
    return X_train, y_train, X_test, y_test

def load_heart_data(test_size=0.3, random_state=42):
    heart = fetch_ucirepo(id=45)
    X_df = heart.data.features
    y_df = heart.data.targets.squeeze()
    print(X_df.head)
    y_df = (y_df > 0).astype(int)
    # => Split
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_df, y_df, test_size=test_size, shuffle=True, random_state=random_state
    )
    # NaN -> 0
    X_train_np = np.nan_to_num(X_train_np, nan=0.0)
    X_test_np  = np.nan_to_num(X_test_np,  nan=0.0)

    # scale
    scaler = MinMaxScaler()
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

    return X_train, y_mapped, X_test, y_test

def load_abalon_data():  
    # fetch dataset 
    abalone = fetch_ucirepo(id=1) 
    

    
    # data (as pandas dataframes) 
    X = abalone.data.features 
    y = abalone.data.targets 


    X["Target Variable"] = y  # "Target Variable" ist der Name für y
    #print(X.head(10))
    encodings = X.groupby('Sex')['Target Variable'].mean().reset_index()
    encoding_mapping = encodings.set_index('Sex')['Target Variable'].to_dict()
    X['Sex_target_encoded'] = X['Sex'].map(encoding_mapping)
    #print(X.head(10))

    X["AgeGroup"] = pd.cut(X["Target Variable"], bins=[0, 8, 11, np.inf], labels=[0, 1, 2])
    X["AgeGroup"] = X["AgeGroup"].astype(int)
    y = X["AgeGroup"]
    
    #X = X.merge(encodings, how='left', on='Sex')
    X = X.drop('Sex', axis=1)
    X = X.drop('Target Variable', axis=1)
    X = X.drop('AgeGroup', axis = 1)
    #print(X.head(10))
    #print(y.head(10))
    

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



    return X_train, y_train, X_test, y_test


def load_abalone_data( test_size = 0.3, random_state = 8):

    abalone = fetch_ucirepo(id=1)
    X = abalone.data.features.copy()
    y = abalone.data.targets.squeeze()
    
    X_num = X.select_dtypes(exclude='object')
    X_cat = X.select_dtypes(include='object')

    X_tranf_cat = pd.get_dummies(X_cat)
    
    X = pd.concat([X_num, X_tranf_cat], axis=1) 
    
    X_processed = X.astype(np.float32)
    
    print(torch.bincount(torch.from_numpy(y.values)))
    

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y,
        test_size=test_size, random_state=random_state
    )

    scaler_cls = MinMaxScaler()
    fitted_scaler = scaler_cls().fit(X_train)
    X_train_scaled = fitted_scaler.transform(X_train)
    X_test_scaled = fitted_scaler.transform(X_test)

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.long)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.long)

    return X_train_t, y_train_t, X_test_t, y_test_t


def load_Kp_chess_data_ord(test_size=0.3, random_state=42):
    chess_king_rook_vs_king_pawn = fetch_ucirepo(id=22) 
    
    # data (as pandas dataframes) 
    X = chess_king_rook_vs_king_pawn.data.features 
    y = chess_king_rook_vs_king_pawn.data.targets 
    
    encoder = OrdinalEncoder()

    X = encoder.fit_transform(X)

    
    # Numerische Features skalieren
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    # Kombiniere numerische und kategoriale Features

    # Label kodieren (falls nötig)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Aufteilen in Trainings- und Testdaten
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X, y_encoded, test_size=test_size, shuffle=True, random_state=random_state
    )
    
    # Eventuelle NaN-Werte ersetzen
    X_train_np = np.nan_to_num(X_train_np, nan=0.0)
    X_test_np  = np.nan_to_num(X_test_np,  nan=0.0)
    
    # Konvertiere in PyTorch-Tensoren
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    X_test  = torch.tensor(X_test_np, dtype=torch.float32)
    y_test  = torch.tensor(y_test_np, dtype=torch.long)
    
    #print("X_train shape:", X_train.shape)
    #print(X_train)
    
    return X_train, y_train, X_test, y_test




def load_htru_data(test_size=0.3, random_state=42):
    """
    Loads and preprocesses the MAGIC Gamma Telescope dataset.
    Features are scaled using MinMaxScaler.
    Target labels ('g', 'h') are encoded to 0 and 1.
    """
    wine_quality = fetch_ucirepo(id=186) 
  
    # data (as pandas dataframes) 
    X = wine_quality.data.features 
    y = wine_quality.data.targets 
    
        
    # Preprocess features
    scaler = MinMaxScaler()
    X_processed_np = scaler.fit_transform(X) 
    X_processed_np = X_processed_np.astype(np.float32)
    
    # Preprocess targets
    # y_df.values.ravel() converts the DataFrame column to a 1D NumPy array
    le = LabelEncoder()
    y_encoded_np = le.fit_transform(y.values.ravel()) # Encodes 'g'/'h' to 0/1
    
    # Split data into training and testing sets
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_processed_np, y_encoded_np, 
        test_size=test_size, 
        shuffle=True, 
        random_state=random_state,
        stratify=y_encoded_np # Stratification is good for classification
    )
    
    # Convert NumPy arrays to PyTorch tensors
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    X_test  = torch.tensor(X_test_np, dtype=torch.float32)
    y_test  = torch.tensor(y_test_np, dtype=torch.long)
    
    return X_train, y_train, X_test, y_test


def load_pmd_data(test_size=0.3, random_state=42):
    """
    Loads and preprocesses the MAGIC Gamma Telescope dataset.
    Features are scaled using MinMaxScaler.
    Target labels ('g', 'h') are encoded to 0 and 1.
    """
    sepsis_survival_minimal_clinical_records = fetch_ucirepo(id=827) 
    
    # data (as pandas dataframes) 
    X = sepsis_survival_minimal_clinical_records.data.features 
    y = sepsis_survival_minimal_clinical_records.data.targets 
    
    X = X.astype(float)
  
    
        
    # Preprocess features
    scaler = MinMaxScaler()
    X_processed_np = scaler.fit_transform(X) 
    X_processed_np = X_processed_np.astype(np.float32)
    
    # Preprocess targets
    # y_df.values.ravel() converts the DataFrame column to a 1D NumPy array
    le = LabelEncoder()
    y_encoded_np = le.fit_transform(y.values.ravel()) # Encodes 'g'/'h' to 0/1
    
    # Split data into training and testing sets
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_processed_np, y_encoded_np, 
        test_size=test_size, 
        shuffle=True, 
        random_state=random_state,
        stratify=y_encoded_np # Stratification is good for classification
    )
    
    # Convert NumPy arrays to PyTorch tensors
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    X_test  = torch.tensor(X_test_np, dtype=torch.float32)
    y_test  = torch.tensor(y_test_np, dtype=torch.long)
    
    return X_train, y_train, X_test, y_test



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

def visualize_data_umap(X, y, title="UMAP projection of the dataset", n_neighbors=15, min_dist=0.1, random_state=42):
    """
    Visualizes the dataset X colored by y using UMAP.

    Args:
        X (torch.Tensor or np.ndarray): Feature data.
        y (torch.Tensor or np.ndarray): Label data.
        title (str): Title for the plot.
        n_neighbors (int): UMAP n_neighbors parameter.
        min_dist (float): UMAP min_dist parameter.
        random_state (int): Random state for UMAP.
    """
    if isinstance(X, torch.Tensor):
        X_np = X.cpu().numpy()
    else:
        X_np = X
    
    if isinstance(y, torch.Tensor):
        y_np = y.cpu().numpy()
    else:
        y_np = y

    if X_np.shape[0] == 0:
        print(f"Cannot visualize UMAP for '{title}': Input data X is empty.")
        return
    if y_np.shape[0] == 0:
        print(f"Cannot visualize UMAP for '{title}': Input data y is empty.")
        return
    if X_np.shape[0] != y_np.shape[0]:
        print(f"Cannot visualize UMAP for '{title}': X and y have mismatched sample numbers ({X_np.shape[0]} vs {y_np.shape[0]}).")
        return

    reducer = umap.UMAP(n_neighbors=min(n_neighbors, X_np.shape[0]-1) if X_np.shape[0] > 1 else 5, # Adjust n_neighbors if samples are few
                       min_dist=min_dist, 
                       n_components=2, 
                       random_state=random_state)
    
    embedding = reducer.fit_transform(X_np)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=y_np, cmap='Spectral', s=15, alpha=0.7)
    
    unique_labels = np.unique(y_np)
    if len(unique_labels) <= 10 and len(unique_labels) > 1: # Show legend for a reasonable number of classes
        handles, _ = scatter.legend_elements()
        plt.legend(handles=handles, labels=[str(l) for l in unique_labels], title="Classes")
    elif len(unique_labels) > 1: # Use colorbar if too many classes for a legend
        plt.colorbar(scatter, label='Class Labels')

    plt.gca().set_aspect('equal', 'datalim')
    plt.title(title, fontsize=16)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.grid(True)
    
    viz_dir = "visualizations"
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    # Sanitize title for filename
    safe_title = "".join(c if c.isalnum() or c in (' ', '_', '-') else '' for c in title).strip()
    plot_filename = "_".join(safe_title.lower().split()) + ".png"
    if not plot_filename: # Handle case where title was all special characters
        plot_filename = "umap_visualization.png"
        
    save_path = os.path.join(viz_dir, plot_filename)
    plt.savefig(save_path)
    print(f"UMAP visualization saved to {save_path}")
    plt.close() # Close the plot to free memory
    
def show_label_distribution(y, dataset_name="Dataset", plot=True):
    """
    Shows the distribution of labels in the dataset.

    Args:
        y (torch.Tensor or np.ndarray or pd.Series): Label data.
        dataset_name (str): Name of the dataset for printing and plot title.
        plot (bool): Whether to generate and save a bar plot of the distribution.
    """
    if isinstance(y, torch.Tensor):
        y_np = y.cpu().numpy()
    elif isinstance(y, pd.Series):
        y_np = y.to_numpy()
    else:
        y_np = np.asarray(y) # Ensure it's a numpy array

    if y_np.ndim > 1 and y_np.shape[1] > 1:
        print(f"Warning: Labels for '{dataset_name}' appear to be one-hot encoded or multi-label. Taking argmax along axis 1.")
        y_np = np.argmax(y_np, axis=1)
    elif y_np.ndim > 1:
        y_np = y_np.squeeze()


    if y_np.shape[0] == 0:
        print(f"Cannot show label distribution for '{dataset_name}': Input data y is empty.")
        return

    unique_labels, counts = np.unique(y_np, return_counts=True)

    print(f"\nLabel distribution for {dataset_name}:")
    if len(unique_labels) == 0:
        print("  No labels found.")
        return
        
    for label, count in zip(unique_labels, counts):
        print(f"  Label {label}: {count} samples")
    print(f"  Total samples: {np.sum(counts)}")
    print(f"  Number of unique classes: {len(unique_labels)}")

    if plot:
        plt.figure(figsize=(max(8, len(unique_labels) * 0.5), 6)) # Adjust width based on number of labels
        
        # Use seaborn for potentially better aesthetics if available, otherwise matplotlib
        try:
            sns.barplot(x=[str(l) for l in unique_labels], y=counts, palette="viridis")
        except ImportError:
            plt.bar([str(l) for l in unique_labels], counts, color='skyblue')
            
        plt.title(f"Label Distribution for {dataset_name}", fontsize=16)
        plt.xlabel("Class Label", fontsize=12)
        plt.ylabel("Number of Samples", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout() # Adjust layout to prevent labels from overlapping
        
        viz_dir = "visualizations"
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        safe_dataset_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '' for c in dataset_name).strip()
        plot_filename = f"label_distribution_{'_'.join(safe_dataset_name.lower().split())}.png"
        if not plot_filename.replace("label_distribution_", "").replace(".png",""): # Handle empty safe_dataset_name
            plot_filename = "label_distribution.png"

        save_path = os.path.join(viz_dir, plot_filename)
        plt.savefig(save_path)
        print(f"Label distribution plot saved to {save_path}")
        plt.close()
        
        
def load_parking_birmingham_dataset(test_size=0.3, random_state=42):
    """
    Loads and preprocesses the Birmingham Parking dataset.
    Features are scaled using MinMaxScaler.
    Target labels are encoded to 0 and 1.
    """
    # Fetch dataset
    birmingham_parking = fetch_ucirepo() 
    
    # Data (as pandas dataframes) 
    X_df = birmingham_parking.data.features 
    y_df = birmingham_parking.data.targets # y_df is a DataFrame, e.g., shape (N, 1)
     
    # Preprocess features
    scaler = MinMaxScaler()
    X_processed_np = scaler.fit_transform(X_df) 
    X_processed_np = X_processed_np.astype(np.float32)
    
    # Preprocess targets
    le = LabelEncoder()
    y_encoded_np = le.fit_transform(y_df.values.ravel()) # Encodes 'g'/'h' to 0/1
    
    # Split data into training and testing sets
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_processed_np, y_encoded_np, 
        test_size=test_size, 
        shuffle=True, 
        random_state=random_state,
        stratify=y_encoded_np # Stratification is good for classification
    )
    
    # Convert NumPy arrays to PyTorch tensors
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    X_test  = torch.tensor(X_test_np, dtype=torch.float32)
    y_test  = torch.tensor(y_test_np, dtype=torch.long)
    
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
   #X_train, y_train, X_test, y_test = load_K_chess_data_splitted()
   #X_train, y_train, X_test, y_test = load_Kp_chess_data_ord()
   #X_train, y_train, X_test, y_test = load_abalon_data()
   X_train, y_train, X_test, y_test = load_gamma_data()
   #X_train, y_train, X_test, y_test = load_shuttle_data()
   #X_train, y_train, X_test, y_test = load_poker_data()
   
   show_label_distribution(y_train, dataset_name="MAGIC Gamma Training Data")
   visualize_data_umap(X_train, y_train, title="UMAP of MAGIC Gamma Training Data")

    
    
        


    


    

    

    

       
       
