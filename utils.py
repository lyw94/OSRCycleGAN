import cv2 as cv
import numpy as np
import os
from pprint import pprint


def load_image(img_dir, normalize):
    high_res = cv.imread(img_dir)

    height, width, channel = high_res.shape
    low_res = cv.resize(high_res, (int(width/16), int(height/16)))
    low_res = cv.GaussianBlur(low_res, (3, 3), 3)
    low_res = cv.resize(low_res, (width, height))

    img_pair = np.array[_normalize_img(high_res, normalize), _normalize_img(low_res, normalize)]
    return img_pair

def _normalize_img(img, normalize):
    if normalize:
        return ((img/127.5)-1).astype(np.float32)
    else:
        return ((img+1)*127.5).astype(np.float32)


class ImageTool():
    def __init__(self, train_path, ground_path, dataset_labels, batch_size):

        self.train_image_dir = train_path
        self.ground_truth_dir = ground_path
        self.loading_labels = self.parse_image_list(dataset_labels)
        self.numImages = len(self.loading_labels)
        self.batch_size = batch_size
        self.batch_idx = 0
        self.batch_length = len(self.loading_labels) // self.batch_size

    def parse_image_list(self, path):
        out_list = []
        file = open(path, 'r')
        for line in file:
            line = line.replace('\n', '').split(' ')
            out_list.append(line)

        return out_list
        # return imglist

    def load_image(self, path):
        # print(self.train_image_dir, self.ground_truth_dir, path)
        imgA = cv.imread(self.train_image_dir+'/'+path)
        path_split = path.split('/')
        path = os.path.join(path_split[0], path_split[1], 'CROP_'+path_split[2]).replace('\\', '/')
        imgB = cv.imread(self.ground_truth_dir+'/'+path)

        imgA = (imgA/127.5 - 1.).astype(np.float32)
        imgB = (imgB/127.5 - 1.).astype(np.float32)

        imgAB = np.concatenate((imgA, imgB)).reshape(2, imgA.shape[0], imgA.shape[1], 3)
        # imgAB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
        return imgAB

    def calc_euclidean_distance(self, net, real, fake, batch_size, labels):
        '''
        real[0] - fake[1]
        real[1] - fake[0]
        '''
        auth_euclidean_loss = 0.0
        impo_euclidean_loss = 0.0
        for f, f_label in zip(fake, labels):
            f_num_label = int(f_label[1].decode('ascii'))
            f_caffe_mode = 2 if (f_num_label >= 142) else 1
            if f_caffe_mode == 2:
                f_num_label -= 142
            for r, r_label in zip(real, labels):
                r_num_label = int(r_label[1].decode('ascii'))
                r_caffe_mode = 2 if (r_num_label >= 142) else 1
                if r_caffe_mode == 2:
                    r_num_label -= 142
                real_reshape = cv.resize(self.restore_RGB(r), (224, 224), interpolation=cv.INTER_CUBIC).transpose(2, 0, 1).reshape(1, 3, 224, 224)
                fake_reshape = cv.resize(self.restore_RGB(f), (224, 224), interpolation=cv.INTER_CUBIC).transpose(2, 0, 1).reshape(1, 3, 224, 224)

                inception_real = net(real_reshape, r_caffe_mode).flatten()
                inception_fake = net(fake_reshape, f_caffe_mode).flatten()

                softmax_real = (np.exp(inception_real) / np.sum(np.exp(inception_real)))
                softmax_fake = (np.exp(inception_fake) / np.sum(np.exp(inception_fake)))

                temp = np.sqrt(np.sum(np.power(np.fabs(softmax_real[r_num_label] - softmax_fake[f_num_label]), 2)))
                if r_num_label == f_num_label:
                    auth_euclidean_loss += temp
                    # print(f_label[0].decode('ascii'), r_label[0].decode('ascii'), temp, 'auth')
                else:
                    impo_euclidean_loss += temp
                    # print(f_label[0].decode('ascii'), r_label[0].decode('ascii'), temp, 'impo')
        # print(auth_euclidean_loss+(1-impo_euclidean_loss))
        return auth_euclidean_loss, (1-impo_euclidean_loss)