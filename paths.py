from os.path import join
from utils import check_mkdir

# Most likely you will need to adjust caffe_root, data_root and *.caffemodel -
# - model_weights in models tuples

# GPU mode is recommended, CPU mode may take several 
# days to complete feature extraction
#
caffe_mode = "CPU"  # CPU|GPU
caffe_device = 0

caffe_root = "/home/torch/caffe-rc3"
data_root  = "/media/torch/WD/kaggle/Yelp"

check_mkdir(join(data_root,  "features"))
check_mkdir(join(data_root,  "submission"))

#
# models: tuple of (model_weights, model_prototxt, mean_image,
#                   image_feature_file, biz_feature_file)
#
models = (
    (join(caffe_root, "models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"),
     join(data_root,  "models/reference_caffenet.prototxt"),
     join(data_root,  "models/ilsvrc_2012_mean.npy"),
     join(data_root,  "features/bvlc_reference_{}_image_features.h5"),
     join(data_root,  "features/bvlc_reference_{}_biz_features_{}.pkl")),

    (join(caffe_root, "models/PlacesCNN/places205CNN_iter_300000.caffemodel"),
     join(data_root,  "models/places205CNN.prototxt"),
     join(data_root,  "models/places205CNN_mean.npy"),
     join(data_root,  "features/places205CNN_{}_image_features.h5"),
     join(data_root,  "features/places205CNN_{}_biz_features_{}.pkl")),

    (join(caffe_root, "models/PlacesHybridCNN/hybridCNN_iter_700000_upgraded.caffemodel"),
     join(data_root,  "models/places205CNN.prototxt"),
     join(data_root,  "models/hybridCNN_mean.npy"),
     join(data_root,  "features/hybridCNN_{}_image_features.h5"),
     join(data_root,  "features/hybridCNN_{}_biz_features_{}.pkl")),

    (join(caffe_root, "models/bvlc_alexnet/bvlc_alexnet.caffemodel"),
     join(data_root,  "models/alexnet.prototxt"),
     join(data_root,  "models/ilsvrc_2012_mean.npy"),
     join(data_root,  "features/alexnet_{}_image_features.h5"),
     join(data_root,  "features/alexnet_{}_biz_features_{}.pkl")),
     )
comb_modes = ("mean", "max")     
#
# * Notice:
#   bvlc_reference_caffenet and bvlc_alexnet use same mean image
#   PlacesHybridCNN uses places205CNN.prototxt
#
#   _mean/_max in biz_features file names determine how image features get
#   combined in Stage2_CreateBuisnessFeatures.py    

     
#     
# data_sets: tuple of (mode, image_folder, photo_to_biz, labels)
#
data_sets = (
    ("train",
     join(data_root, "train_photos"),
     join(data_root, "train_photo_to_biz_ids.csv"), 
     join(data_root, "train.csv")),

    ("test",
     join(data_root, "test_photos"),
     join(data_root, "test_photo_to_biz.csv"),
     join(data_root, "sample_submission.csv")),
    )
 

#
# blending parameters
#
n_folds = 10
C = 0.7
threshold = 0.5
#
# Results of blending
#   
blendLR_feature_fl = join(data_root, "features/all_models_blendLR_CV_features.pkl")
blendKerasXGBoost_feature_fl = join(data_root, "features/all_models_blendKerasXGBoost_CV_features.pkl")
#
# Submission files
#
submission_LRblend_CV_fl = join(data_root, "submission/all_models_blendLR_CV.csv")
submission_nomeuf_fl = join(data_root, "submission/keras_xgboost_blend_noMEUF.csv")
submission_meuf_fl = join(data_root, "submission/keras_xgboost_blend_MEUF.csv")
