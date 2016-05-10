import numpy as np
import pandas as pd 
import h5py
from os.path import splitext, basename
import cPickle as pickle
import time
import paths as pt


def process_data_set(photo_to_biz_fl, labels_fl, image_feature_fl, business_feature_fl, comb_mode):
    photo_to_biz = pd.read_csv(photo_to_biz_fl)
    labels = pd.read_csv(labels_fl, index_col="business_id").dropna()
    labels['labels'] = labels['labels'].apply(lambda x: tuple(sorted(int(t) for t in x.split())))
    print "Number of businesses: ", len(labels)
    
    with h5py.File(image_feature_fl, 'r') as f:
        
        X = pd.DataFrame(np.zeros((len(labels), f['feature'].shape[1]),
                                  dtype=f['feature'].dtype),
                         index=labels.index)
        ph = pd.DataFrame(f['photo_id'][()], columns=['photo_id']).reset_index()
        comb = photo_to_biz.merge(ph, on="photo_id")
        buz_len = comb.groupby("business_id").aggregate(len)
        gr = comb.groupby("index")
        
        # read .h5 in chunks
        chunk_sz = 1000 
        for index in np.arange(0, len(f['feature']), chunk_sz):
            features = f['feature'][index:index+chunk_sz]
            for i, feature in enumerate(features): 
                image_index = index + i
                business_index = gr.get_group(image_index)["business_id"].tolist()
                ind = X.index.isin(business_index)
                if comb_mode == "mean":
                    X[ind] += feature
                elif comb_mode == "max":
                    X[ind] = np.maximum(X[ind], feature)
                else:
                    raise(ValueError("Create Buisness Features: unknown combine mode"))
                if image_index % 100==0: print "Images processed:", image_index
    
    if comb_mode == "mean":
        X = X.divide(buz_len["index"].loc[X.index], axis="index")
    
    y = labels['labels'].values
    biz_ids = labels.index.values
    with open(business_feature_fl, 'wb') as f:
        pickle.dump((y, X.as_matrix(), biz_ids), f, -1)
    

if __name__ == '__main__':
    for (model_weights, _, _, image_feature_fl, business_feature_fl) in pt.models:
        for comb_mode in pt.comb_modes:
            for (mode, _, photo_to_biz_fl, labels_fl) in pt.data_sets:
                print "Processing {}, {} comb mode, {} set".format(splitext(basename(model_weights))[0], comb_mode, mode)    
                t = time.time()
                process_data_set(photo_to_biz_fl, labels_fl,
                                 image_feature_fl.format(mode),
                                 business_feature_fl.format(mode, comb_mode),
                                 comb_mode)
                print "Time passed: ", "{0:.1f}".format(time.time()-t), "sec"
