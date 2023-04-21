import numpy as np
import pandas as pd

def zoo_dataset():
    zoo = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data", header=None)
    zoo.columns = ['animal name','hair','feathers','eggs','milk','airborne','aquatic', \
                   'predator','toothed','backbone','breathes','venomous','fins','legs', \
                   'tail','domestic','catsiz','type']
    zoo = pd.get_dummies(zoo, columns=['legs'])
    Y_zoo = zoo.loc[:,'type'].to_numpy()
    del zoo['type']
    zoo_names = zoo.loc[:,'animal name'].to_list()
    del zoo['animal name']
    zoo_features = zoo.values.astype(np.int8)
    zoo_indices = np.arange(len(zoo_names))
    X_zoo = [(i, j, k) for i, j, k in zip(zoo_indices, zoo_features, zoo_names)]
    return X_zoo, Y_zoo, zoo.columns.to_list()

def nursery_dataset():
    nur = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data", header=None)
    nur.columns = ['parents','has_nurs','form','children','housing','finance','social','health','type']
    nur = pd.get_dummies(nur, columns=['parents','has_nurs','form','children','housing','finance','social','health'])
    Y = nur.loc[:,'type'].to_numpy()
    del nur['type']
    indices = np.arange(len(nur.values))
    features = nur.values.astype(np.int8)
    X = [(i, j, k) for i, j, k in zip(indices, features, indices)]
    return X, Y, nur.columns.to_list()

def chess_dataset():
     chess = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king/krkopt.data", header=None)
     chess.columns = ['White King file','White King rank','White Rook file','White Rook rank','Black King file','Black King rank','type']
     chess = pd.get_dummies(chess, columns=['White King file','White King rank','White Rook file','White Rook rank','Black King file','Black King rank'])
     Y = chess.loc[:,'type'].to_numpy()
     del chess['type']
     indices = np.arange(len(chess.values))
     features = chess.values.astype(np.int8)
     X = [(i, j, k) for i, j, k in zip(indices, features, indices)]
     return X, Y, chess.columns.to_list()

def mushroom_dataset():
     mush = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data", header=None)
     mush.columns = ['type','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22']
     mush = pd.get_dummies(mush, columns=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22'])
     Y = mush.loc[:,'type'].to_numpy()
     del mush['type']
     indices = np.arange(len(mush.values))
     features = mush.values.astype(np.int8)
     X = [(i, j, k) for i, j, k in zip(indices, features, indices)]
     return X, Y, mush.columns.to_list()

def con_dataset():
    con = pd.read_csv("./connect-4.data", header=None)
    con.columns = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','type']
    con = pd.get_dummies(con, columns=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42'])
    Y = con.loc[:,'type'].to_numpy()
    del con['type']
    indices = np.arange(len(con.values))
    features = con.values.astype(np.int8)
    X = [(i, j, k) for i, j, k in zip(indices, features, indices)]
    return X, Y, con.columns.to_list()
