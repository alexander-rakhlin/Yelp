import xgboost as xgb
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import MultiLabelBinarizer

import time
from os.path import basename
import cPickle as pickle
import numpy as np
import pandas as pd

from utils import meuf, probs2str, KerasClassifier
import paths as pt


random_state = np.random.RandomState(11)

def process_fold(X_train, X_val, y_train, y_val, X_test):
    #XGBoos
    clf = OneVsRestClassifier(xgb.XGBClassifier(learning_rate=0.005, n_estimators=500))
    clf.fit(X_train, y_train)
    y_p_x = clf.predict_proba(X_val)
    y_p_x_tst = clf.predict_proba(X_test)
    
    # Keras
    y_p_k, y_p_k_tst = KerasClassifier(X_train, y_train, X_val, y_val, X_test)
    
    return (y_p_x+y_p_k) / 2.0, (y_p_x_tst+y_p_k_tst) / 2.0


blend_feature_file = pt.blendKerasXGBoost_feature_fl
submission_nomeuf_file = pt.submission_nomeuf_fl
submission_meuf_file = pt.submission_meuf_fl

n_folds = pt.n_folds
threshold = pt.threshold

with open(pt.blendLR_feature_fl, 'rb') as f:
    y, X, X_test, biz_ids = pickle.load(f)
N_train, num_lab = y.shape

print "\nCreating train and test sets for blending."
kf = KFold(N_train, n_folds, shuffle=True, random_state=12)
blend_train = np.zeros((N_train, num_lab))
blend_test = np.zeros((X_test.shape[0], len(kf), num_lab))

for i, (train, validate) in enumerate(kf):
    t = time.time()
    print "Fold", i+1
    X_train = X[train]
    y_train = y[train]
    X_validate = X[validate]
    y_validate = y[validate]
    y_p_val, y_p_tst = process_fold(X_train, X_validate, y_train, y_validate, X_test)
    blend_train[validate, :] = y_p_val
    blend_test[:, i, :] = y_p_tst
    f1 = f1_score(y_validate, 1*(y_p_val>threshold), average='samples')
    print "...time passed: {0:.1f}sec, fold F1: {1:.4f}".format(time.time()-t, f1)

# F1
f1 = f1_score(y, 1*(blend_train>threshold), average='samples')
f1_meuf = f1_score(y, meuf(blend_train), average='samples')
print "Total F1: {:.4f}, F1-MEUF: {:.4f}".format(f1, f1_meuf) 


# Saving
blend_test = blend_test.mean(axis=1)

print "\nSaving to feature file."
with open(blend_feature_file, 'wb') as f:
    pickle.dump((y, blend_train, blend_test, biz_ids), f, -1)

print "\nGenerating submissions."
mlb = MultiLabelBinarizer()
mlb.fit_transform((range(num_lab),))

labels = probs2str(1*(blend_test>threshold), mlb)
submission = pd.DataFrame(zip(biz_ids, labels), columns=('business_id', 'labels'))
submission.to_csv(submission_nomeuf_file, index=False)

labels_meuf = probs2str(meuf(blend_test), mlb)
submission = pd.DataFrame(zip(biz_ids, labels_meuf), columns=('business_id', 'labels'))
submission.to_csv(submission_meuf_file, index=False)

print "{}, {} done".format(basename(submission_nomeuf_file),
                           basename(submission_meuf_file))