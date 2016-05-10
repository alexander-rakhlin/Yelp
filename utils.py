import numpy as np
from os.path import exists
import os

from keras.models import Sequential
from keras.layers.core import Dense, Dropout

""" Convert probability matrix into string of labels """
def probs2str(y_b, mlb):
    # y_b - input binary matrix,
    # mlb - instance of MultiLabelBinarizer

    labels = mlb.inverse_transform(y_b)
    labels2str = lambda s: " ".join(map(str, s))
    return map(labels2str, labels)
    

""" Calculate F1 score for samples encoded as integers """    
def f1_int(h, y):
    P = bin(h).count("1")
    T = bin(y).count("1")
    A = bin(h & y).count("1")
    return 2.0*A/(P+T) if (P+T) !=0 else 1.0    # 0/0 = 1
    

""" Given probability matrix adjust for Maximum Expected F-Score, MEUF"""
def meuf(y_p):
    # y_p - input probability matrix
    
    num_lab = y_p.shape[1]
    F = np.array([[f1_int(i, j) for j in range(2**num_lab)] for i in range(2**num_lab)])
    formatter = '{{0:0{}b}}'.format(num_lab)
    D = np.zeros((y_p.shape[0], 2**num_lab))
    for b in range(2**num_lab):
        y_s = np.array([int(s) for s in formatter.format(b)])
        prob = np.vstack([np.log(h_j if y_s[i]==1 else 1-h_j) for i, h_j in enumerate(y_p.T)])
        D[:, b] = np.exp(prob.sum(axis=0))
    
    H = np.array([(d * F).sum(axis=1) for d in D])
    Harg = np.argmax(H, axis=1)
    Rh = np.array([np.array([int(s) for s in formatter.format(d)]) for d in Harg])
    return Rh    
    
    
def KerasClassifier(X_train, y_train, X_val, y_val, X_test, verbose=2):
    clf = Sequential()
    clf.add(Dense(100, activation='relu', input_dim=X_train.shape[1]))
    clf.add(Dropout(0.7))
    clf.add(Dense(y_train.shape[1], activation='sigmoid'))
    
    clf.compile(optimizer='adam', loss='binary_crossentropy')
    
    clf.fit(X_train, y_train, batch_size=128, nb_epoch=300,
            validation_data=(X_val, y_val), verbose=verbose)
    
    return clf.predict(X_val), clf.predict(X_test)  

    
def check_mkdir(dir):
    if not exists(dir): os.mkdir(dir)    