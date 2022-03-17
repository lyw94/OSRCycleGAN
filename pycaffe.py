import numpy as np
import cv2 as cv
import os
import sys
caffe_root = 'C:/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe

#caffe.set_device(0)
class CaffeNet():
    def __init__(self, model_def, model_weights):
        #caffe.set_mode_cpu()
        caffe.set_device(0)
        caffe.set_mode_gpu()
        self.net_1 = self.get_caffeNet(model_def[0], model_weights[0])
        self.net_2 = self.get_caffeNet(model_def[1], model_weights[1])

    def get_caffeNet(self, model_def, model_weights):
        net = caffe.Net(model_def,      # defines the structure of the model
                        model_weights,  # contains the trained weights
                        caffe.TEST)     # use test mode (e.g., don't perform dropout)
        return net

    def __call__(self, inputs, db_type):
        # self.net.blobs['data'] = inputs
        #
        # self.net.forward()
        # print(inputs.shape)
        net = None
        if db_type == 1:
            net = self.net_1
        if db_type == 2:
            net = self.net_2

        net.forward(data=np.asarray(inputs))

        return net.blobs['fc_out_distance'].data
