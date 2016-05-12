# Yelp Restaurant Photo Classification

My solution that scored 0.82246 and finished [Yelp Restaurant Photo Classification](https://www.kaggle.com/c/yelp-restaurant-photo-classification) on 22 position out of 355 teams (top 10%).

## Implements the following pipeline:
* Extract image features using four Caffe models. Results in eight 9GB image feature files (two train/test sets for every model)
* Combine image features into business features in two different ways. Results in 16 feature sets, much smaller this time
* Train Logistic Regression classifiers on business features via 10-Fold CV on every business feature set. Stack classifier output into single set of meta features. Train Logistc Regression on meta features and generate an intermediate submission
* Train Neural Net and XGBoost on meta features, predict label probabilities, average predictions, generate submission without F1 Maximization
* Adjust probabilities via Maximum Expected Utility Framework for F-Measure Maximization, generate final submission.

## Requires:
* [Caffe](http://caffe.berkeleyvision.org/), deep learning framework
* Scientific Python Stack (including [NumPy](http://www.numpy.org/), [SciPy](http://www.scipy.org/), [Pandas](http://pandas.pydata.org/). All this can be obtaned with [Anaconda](https://www.continuum.io/downloads) distribution)
* [XGBoost](https://github.com/dmlc/xgboost)
* [Theano](http://deeplearning.net/software/theano/)
* [Keras](http://keras.io/)
* About 100 GB of free disk space is needed for train/test images, extracted image features, model dumps.

NVIDIA GPU is not required but recommended. Extracting image features on CPU may take several days.

## Download:
* The training and test datasets and other data can be downloaded from [here](https://www.kaggle.com/c/yelp-restaurant-photo-classification/data)
* Get pretrained Caffe models **BVLC Reference CaffeNet** and **BVLC AlexNet** as described [here](http://caffe.berkeleyvision.org/model_zoo.html). Download the other two models from Places CNN project: [Places205-AlexNet](http://places.csail.mit.edu/model/placesCNN_upgraded.tar.gz), [Hybrid-AlexNet](http://places.csail.mit.edu/model/hybridCNN_upgraded.tar.gz)  
Notice: customized prototxts and mean files already available in the **models** folder

## How to generate the solution(s):
1. After you downloaded and extracted datasets and models, adjust paths in paths.py and set caffe_mode (currently set to CPU)
2. Successively run (make sure you have enough disk space, see above):
python Stage1_ExtractImageFeatures.py 
python Stage2_CreateBuisnessFeatures.py 
python Stage3_BlendLRModelsCV.py 
python Stage4_KerasXGBoostMEUFsubmission.py 
3. You will get three submissions: 
all_models_blendLR_CV.csv 
keras_xgboost_blend_noMEUF.csv
keras_xgboost_blend_MEUF.csv

Enjoy!

Read my article ["What restaurant would your computer like to go to?"](https://www.linkedin.com/pulse/article/what-restaurant-would-your-computer-like-go-alexander-rakhlin) and like it if you like it.