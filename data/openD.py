import arff
import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

def loadKP():
    krKp_path = os.path.join(current_dir, 'kr-vs-kp.arff')

    with open(krKp_path, 'r') as f:
        datasetKp = arff.load(f)

    attribute_names_Kp = [attr[0] for attr in datasetKp['attributes']]

    df_Kp = pd.DataFrame(datasetKp['data'], columns=attribute_names_Kp)



    return df_Kp

def loadK():
    krK_path = os.path.join(current_dir, 'kr-vs-k.arff')

    with open(krK_path, 'r') as f:
        datasetK = arff.load(f)
    
    attribute_names_K = [attr[0] for attr in datasetK['attributes']]

    df_K = pd.DataFrame(datasetK['data'], columns=attribute_names_K)

    return df_K

def loadPoker():

    poker_path = os.path.join(current_dir, 'poker-9class.arff')

    with open(poker_path, 'r') as f:
        datasetPoker = arff.load(f)

    attribute_names_P = [attr[0] for attr in datasetPoker['attributes']]

    df_P = pd.DataFrame(datasetPoker['data'], columns=attribute_names_P)

    return df_P
