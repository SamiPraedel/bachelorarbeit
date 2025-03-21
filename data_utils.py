# data_utils.py

import torch
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from ucimlrepo import fetch_ucirepo
import pandas as pd
from scipy import stats
from data.openD import loadK, loadKP, loadPoker

def load_K_chess_data(test_size=0.2, random_state=42):
    df = loadK()
        # Annahme: Die letzte Spalte ist die Zielvariable
       # Features und Label trennen
    X = df.drop(columns=['Class'])
    y = df['Class']
    
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
    
    print("y_train shape:", y_train.shape)
    
    return X_train, y_train, X_test, y_test


def load_Kp_chess_data(test_size=0.2, random_state=42):
    df = loadKP()
        # Annahme: Die letzte Spalte ist die Zielvariable
       # Features und Label trennen
    X = df.drop(columns=['Class'])
    y = df['Class']
    
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
    
    print("y_train shape:", y_train.shape)
    
    return X_train, y_train, X_test, y_test

def load_heart_data(test_size=0.2, random_state=42):
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


if __name__ == "__main__":
    load_Kp_chess_data()

       
       
