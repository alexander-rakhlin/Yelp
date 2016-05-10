# Rank=20;t=0;Nteams=289; 100000*(np.power(Rank, -0.75))*(np.log10(1+np.log10(Nteams)))*(np.exp(-t/500.0))

import numpy as np
import sys
import os
from os.path import join, splitext, basename

import pandas as pd 
import h5py
import time
import paths as pt

caffe_root = pt.caffe_root
sys.path.insert(0, join(caffe_root, 'python'))
os.environ['GLOG_minloglevel'] = '2'
import caffe
if pt.caffe_mode == "CPU":  
    caffe.set_mode_cpu()
else:
    caffe.set_device(pt.caffe_device)
    caffe.set_mode_gpu()



def net_init(model_prototxt, model_weights, mean_data):
    net = caffe.Net(model_prototxt, model_weights, caffe.TEST)
    
    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', mean_data) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB]]
    return net, transformer


def extract_features(net, transformer, images, input_dim, layer = 'concat'):
    net.blobs['data'].reshape(len(images), *input_dim)
    net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data',caffe.io.load_image(x)), images)
    net.forward()
    return net.blobs[layer].data  


def process_data_set(net, transformer, feature_len, input_dim,
                     image_folder, photo_to_biz_fl, image_feature_fl):
    photo_to_biz = pd.read_csv(photo_to_biz_fl)
    image_list = photo_to_biz['photo_id'].unique()

    image_list = image_list[:100]
    
    num_images = len(image_list)
    print "Number of images: ", num_images
    
    batch_size = 64
    
    # Initialize .h5
    with h5py.File(image_feature_fl, 'w') as f:
        f.create_dataset('photo_id', (num_images,), dtype='int')
        f.create_dataset('feature', (num_images, feature_len))
    
    # extract image features and save it to .h5
    for i in range(0, num_images, batch_size): 
        
        # get full filename
        images = image_list[i: min(i+batch_size, num_images)]
        image_files = [join(image_folder, str(x)+'.jpg') for x in images]

        features = extract_features(net, transformer, image_files, input_dim)
        num_done = i+features.shape[0]
        with h5py.File(image_feature_fl, 'r+') as f:
            f['photo_id'][i: num_done] = np.array(images)
            f['feature'][i: num_done, :] = features
        if num_done % (batch_size*16)==0 or num_done==num_images:
            print "Train images processed: ", num_done


if __name__ == '__main__':
    
    for (model_weights, model_prototxt, mean_image, image_feature_fl, _) in pt.models:
    
        # BVLC models: "fc6" "fc7" "flatten4" "flatten5"
        feature_len = 4096 + 4096 + 384*4 + 256*4
        mean_data = np.load(mean_image).mean(1).mean(1)
        input_dim = (3, 227, 227)
        net, transformer = net_init(model_prototxt, model_weights, mean_data)
        
        for (mode, image_folder, photo_to_biz_fl, _) in pt.data_sets:
            print "Processing {}, {} set".format(splitext(basename(model_weights))[0], mode)
            t = time.time()
            process_data_set(net, transformer, feature_len, input_dim, image_folder,
                             photo_to_biz_fl, image_feature_fl.format(mode))
            print "Time passed: ", "{0:.1f}".format(time.time()-t), "sec"  
        
             