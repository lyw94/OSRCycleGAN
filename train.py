from model import OSRCycleGAN

import numpy as np
import tensorflow as tf
import keras
import pycaffe
import utils
import time
import os
# from pprint import pprint
import cv2 as cv
from tqdm import tqdm
from absl import logging

import preference as pref

tf.keras.backend.clear_session()  # For easy reset of notebook state.


if pref.use_sgd:
    discriminatorB_optimizer = tf.keras.optimizers.SGD(learning_rate=pref.lr, momentum=pref.sgd_momentum)
    discriminatorA_optimizer = tf.keras.optimizers.SGD(learning_rate=pref.lr, momentum=pref.sgd_momentum)
    generatorA2B_optimizer = tf.keras.optimizers.SGD(learning_rate=pref.lr, momentum=pref.sgd_momentum)
    generatorB2A_optimizer = tf.keras.optimizers.SGD(learning_rate=pref.lr, momentum=pref.sgd_momentum)
else:
    discriminatorB_optimizer = tf.keras.optimizers.Adam(learning_rate=pref.lr, beta_1= pref.beta_1)
    discriminatorA_optimizer = tf.keras.optimizers.Adam(learning_rate=pref.lr, beta_1= pref.beta_1)
    generatorA2B_optimizer = tf.keras.optimizers.Adam(learning_rate=pref.lr, beta_1= pref.beta_1)
    generatorB2A_optimizer = tf.keras.optimizers.Adam(learning_rate=pref.lr, beta_1= pref.beta_1)

gan = OSRCycleGAN(pref.ngf, pref.ndf, pref.L1_lambda, pref.training)

ckpt = tf.train.Checkpoint(generator_A_B=gan.generator_A_B,
                           generator_B_A=gan.generator_B_A,
                           discriminator_A=gan.discriminator_A,
                           discriminator_B=gan.discriminator_B,
                           generator_A2B_opt=generatorA2B_optimizer,
                           generator_B2A_opt=generatorB2A_optimizer,
                           discriminatorA_optimizer=discriminatorA_optimizer,
                           discriminatorB_optimizer=discriminatorB_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, pref.checkpoint_path, max_to_keep=None)

mse = keras.losses.MSE
mae = keras.losses.MeanAbsoluteError(reduction=keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

img_tool = utils.ImageTool(pref.train_img_dir, pref.gt_img_dir, pref.img_label, pref.batch_size)

dataset = tf.data.Dataset.from_tensor_slices(img_tool.loading_labels)
dataset = dataset.shuffle((len(dataset)/2)).batch(pref.batch_size)


def train(img_dirs):
    batch =  np.array([utils.load_image(each.numpy().decode('ascii')) for each in img_dirs]).transpose([1, 0, 2, 3, 4])
    realA, realB = batch

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fakeB, fakeA, reconB, reconA, discriminate_B, discriminate_A, discriminateGT_B, discriminateGT_A = gan(batch)

        if pref.using_caffe:
            auth, impo = utils.calc_euclidean_distance(caffe_net, realB, fakeB.numpy(), pref.batch_size, i.numpy())

        generator_loss = mse(discriminate_A, tf.ones_like(discriminate_A)) \
                        + mse(discriminate_B, tf.ones_like(discriminate_B)) \
                        + (pref.l1_lambda * mae(realA, reconA)) \
                        + (pref.l1_lambda * mae(realB, reconB))
        if pref.using_caffe:
            generator_loss = generator_loss + auth + impo
                        # + auth_euclidean_loss + impo_euclidean_loss

        discriminator_loss_realB = mse(discriminateGT_B, tf.ones_like(discriminateGT_B))
        discriminator_loss_fakeB = mse(discriminate_B, tf.zeros_like(discriminate_B))

        discriminator_loss_B = (discriminator_loss_realB + discriminator_loss_fakeB)/2

        discriminator_loss_realA = mse(discriminateGT_A, tf.ones_like(discriminateGT_A))
        discriminator_loss_fakeA = mse(discriminate_A, tf.zeros_like(discriminate_A))

        discriminator_loss_A = (discriminator_loss_realA + discriminator_loss_fakeA)/2

        discriminator_loss = discriminator_loss_A + discriminator_loss_B
        if pref.using_caffe:
            discriminator_loss = discriminator_loss + auth + impo

        
        generator_grad_A2B = gen_tape.gradient(generator_loss, gan.generator_A_B.trainable_variables)
        generator_grad_B2A = gen_tape.gradient(generator_loss, gan.generator_B_A.trainable_variables)
        discriminator_grad_B = disc_tape.gradient(discriminator_loss, gan.discriminator_B.trainable_variables)
        discriminator_grad_A = disc_tape.gradient(discriminator_loss, gan.discriminator_A.trainable_variables)

        generatorA2B_optimizer.apply_gradients(zip(generator_grad_A2B, gan.generator_A_B.trainable_variables))
        generatorB2A_optimizer.apply_gradients(zip(generator_grad_B2A, gan.generator_B_A.trainable_variables))
        discriminatorB_optimizer.apply_gradients(zip(discriminator_grad_B, gan.discriminator_B.trainable_variables))
        discriminatorA_optimizer.apply_gradients(zip(discriminator_grad_A, gan.discriminator_A.trainable_variables))

# ================================================ #
import pycaffe
caffe_net = pycaffe.CaffeNet(pref.caffe_model_def, pref.caffe_model_weights)

for epoch in tqdm(range(1, pref.epochs+1), desc='epochs', unit='epoch'):
    for iter, i in enumerate(tqdm(dataset, total=len(dataset), desc='iteration')):
        train(i)
    ckpt_save_path = ckpt_manager.save()