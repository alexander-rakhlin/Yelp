from __future__ import division

from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score

import cPickle as pickle
import time
from os.path import basename

import pandas as pd
import numpy as np
import paths as pt
from utils import probs2str


blend_feature_file = pt.blendLR_feature_fl
submission_file = pt.submission_LRblend_CV_fl
n_folds = pt.n_folds
C = pt.C
threshold = pt.threshold

data_sets = tuple(tuple(m[-1].format(t[0], c) for t in pt.data_sets) for m in pt.models for c in pt.comb_modes)
             
def load((train, test), mlb=None):
    with open(train, 'rb') as f:
        y_train, X_train, _ = pickle.load(f)
    with open(test, 'rb') as f:
        _, X_test, biz_ids = pickle.load(f)
    
    # Scale features
    scaler = MinMaxScaler().fit(np.vstack((X_train, X_test)))
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    if mlb is None: mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train)  #Convert list of labels to binary matrix
    
    return y_train, X_train, X_test, biz_ids, mlb


def process_data_set(X_train, y_train, X_val, X_test, c=1.0):
   
    cls = OneVsRestClassifier(LogisticRegression(C=c))
    
    # 4096 + 4096 + 384*4 + 256*4 # "fc6" "fc7" "flatten4" "flatten5"
    # [0, 4096, 8192, 9728, 10752]
    layers = np.array((4096, 4096, 384*4, 256*4))   
    layers = np.concatenate(([0], np.cumsum(layers)))

    r_ = range(layers[0], layers[4])
    x_tr = X_train[:, r_]
    x_vl = X_val[:, r_]
    x_ts = X_test[:, r_]
    
    cls.fit(x_tr, y_train)
    y_vl = cls.predict_proba(x_vl)
    y_ts = cls.predict_proba(x_ts)

    return y_vl, y_ts

y_train, X_train, X_test, biz_ids, mlb = load(data_sets[0])
N_train, num_labels = y_train.shape
N_test = X_test.shape[0]

kf = KFold(N_train, n_folds, shuffle=True, random_state=1)

print "Creating train and test sets for blending."

dataset_blend_train = np.zeros((N_train, len(data_sets), num_labels))
dataset_blend_test = np.zeros((N_test, len(data_sets), num_labels))

for j, ds in enumerate(data_sets):
    t_ds = time.time()
    print "\nData set", j+1
    print map(basename, ds)
    y, X, X_test, _, _ = load(ds, mlb)
    dataset_blend_test_j = np.zeros((N_test, len(kf), num_labels))

    for i, (train, validate) in enumerate(kf):
        t = time.time()
        print "...fold", i+1, "...",
        X_train = X[train]
        y_train = y[train]
        X_validate = X[validate]
        y_validate = y[validate]
        y_validate_predicted, y_test_predicted = process_data_set(X_train, y_train, X_validate, X_test, c=C)
        dataset_blend_train[validate, j, :] = y_validate_predicted
        dataset_blend_test_j[:, i, :] = y_test_predicted
        print "time passed: {0:.1f}sec".format(time.time()-t)
    dataset_blend_test[:, j, :] = dataset_blend_test_j.mean(1)
    
    y_p = 1*(dataset_blend_train[:, j, :]>threshold)
    f1 = f1_score(y, y_p, average='samples')
    print "time passed: {0:.1f}sec, F1 score: {1:.4f}".format(time.time() - t_ds, f1)

y_p = 1*(dataset_blend_train.mean(1)>threshold)
f1 = f1_score(y, y_p, average='samples')
print "Average F1 score across all training sets: {:.4f}".format(f1)

dataset_blend_train = dataset_blend_train.reshape(N_train, num_labels*len(data_sets))
dataset_blend_test = dataset_blend_test.reshape(N_test, num_labels*len(data_sets))

print "\nSaving to feature file."
with open(blend_feature_file, 'wb') as f:
    pickle.dump((y, dataset_blend_train, dataset_blend_test, biz_ids), f, -1)
    
print "\nBlending."
clf = OneVsRestClassifier(LogisticRegression())
clf.fit(dataset_blend_train, y)
P = clf.predict_proba(dataset_blend_test)

y_predict = 1*(P>threshold)

# Make submission
print "\nGenerating linear blend submission."
labels = probs2str(y_predict, mlb)
submission = pd.DataFrame(zip(biz_ids, labels), columns=('business_id', 'labels'))
submission.to_csv(submission_file, index=False)