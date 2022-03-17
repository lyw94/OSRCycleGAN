ngf = 64 # number of filters at first convolution layer in the generator
ndf = 64 # number of filters at first convolution layer in the discriminator

l1_lambda = 10.0
sgd_momentum = 0.9
beta_1 = 0.5
learning_rate = 2e-4
lr = learning_rate

batch_size = 2 # number of batch images
epochs = 200 # number of epoch

train_img_dir = '~/Desktop/exp/resources/train_img_list.txt'
gt_img_dir = '~/Desktop/exp/resources/gr_img_list.txt'
img_label = '~/Desktop/exp/resources/db_label.txt'

checkpoint_path = 'Checkpoints/gaussianblur_train_with_caffe'

use_sgd = False
training = True


########### caffe configuration for perceptual loss ###########
using_caffe = False

caffe_model_def = ['~/Desktop/exp/resources/db1_dir', '~/Desktop/exp/resources/db2_dir']
caffe_model_weights = ['~/Desktop/exp/resources/db1_weights', '~/Desktop/exp/resources/db2_weights']

# caffe_net = pycaffe.CaffeNet(caffe_model_def, caffe_model_weights)

