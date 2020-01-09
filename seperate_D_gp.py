#!/usr/bin/env/ python3

from __future__ import print_function, division

import keras
from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D, Reshape, Flatten, Dropout, multiply, MaxPooling2D
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Concatenate, Lambda, Add
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
import keras.backend as K
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam, RMSprop
from keras import metrics
from keras.layers.merge import _Merge
from functools import partial

import matplotlib.pyplot as plt
import os
import h5py
import pandas as pd

import numpy as np
import tensorflow as tf
from util import load_mnist, onehot, plot_table
from model import build_discriminator_realness, build_discriminator_digit, build_generator

GRADIENT_PENALTY_WEIGHT = 10

def exclude(arr):
    result = [ np.random.choice(list({0,1,2,3,4,5,6,7,8,9}-{digit}), 1)[0] for digit in arr ]
    return np.array(result)

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def gradient_penalty_loss(y_true, y_pred, averaged_samples,
                          gradient_penalty_weight):
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((128, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class GAN():
    def __init__(self, model_name=None, loss_weight=[1,1]):

        self.imgs, self.digits, self.test_imgs, self.test_digits = load_mnist()
        self.img_rows, self.img_cols, self.channels = self.imgs.shape[1:]
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer_D = Adam(0.0001, beta_1=0.5, beta_2=0.9)
        optimizer_G = Adam(0.0001, beta_1=0.5, beta_2=0.9)

        self.G, self.G_mask = build_generator()
        D_real = build_discriminator_realness()
        D_digit = load_model('outputs/D_digit.hdf5')

        ## generator
        # fix D
        for layer in D_real.layers:
            layer.trainable = False
        D_real.trainable = False
        for layer in D_digit.layers:
            layer.trainable = False
        D_digit.trainable = False

        img_input = Input(shape=self.img_shape)
        digit_input = Input(shape=(10,))
        output_realness = D_real(img_input)
        output_digit = D_digit(img_input)
        D_combined = Model(img_input, [output_realness, output_digit])

        G_img_input = Input(shape=self.img_shape)
        G_digit_input = Input(shape=(10,))
        img_added = self.G([G_img_input, G_digit_input])
        D_output = D_combined(img_added)
        self.G_combined = Model([G_img_input, G_digit_input], D_output)
        self.G_combined.compile(
            loss=[wasserstein_loss, 'categorical_crossentropy'],
            loss_weights=[10, 1],
            optimizer=optimizer_G,
            metrics=[wasserstein_loss, 'categorical_accuracy'])
        self.G_combined.summary()


        ## discriminator
        # fix G
        for layer in D_real.layers:
            layer.trainable = True
        D_real.trainable = True
        for layer in D_digit.layers:
            layer.trainable = True
        D_digit.trainable = True
        for layer in self.G.layers:
            layer.trainable = False
        self.G.trainable = False

        D_img_input = Input(shape=self.img_shape)
        D_digit_input = Input(shape=(10,))
        gen_input = self.G([D_img_input, D_digit_input])
        d_for_real = D_real(D_img_input)
        d_for_gen = D_real(gen_input)
        avg_input = RandomWeightedAverage()([D_img_input, gen_input])
        d_for_avg = D_real(avg_input)
        partial_gp_loss = partial(gradient_penalty_loss,
                                  averaged_samples=avg_input,
                                  gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
        partial_gp_loss.__name__ = 'gradient_penalty'
        self.D_combined = Model(inputs=[D_img_input, D_digit_input],
                                outputs=[d_for_real, d_for_gen, d_for_avg])

        self.D_combined.compile(
            loss=[wasserstein_loss,
                  wasserstein_loss,
                  partial_gp_loss],
            optimizer=optimizer_D,
            metrics=[wasserstein_loss])
        self.D_combined.summary()

        self.tb = keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=0,
            batch_size=64,
            write_graph=True,
            write_grads=True
        )
        self.tb.set_model(self.G_combined)

    def train(self, iterations, batch_size=128, sample_interval=100, save_model_interval=100,
                            train_D_iters=1, train_G_iters=1, img_dir='./', model_dir='./'):

        imgs, digits = self.imgs, self.digits

        valid = np.ones((batch_size, 1))
        fake = -valid
        dummy = np.zeros((batch_size, 1))

        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        for itr in range(1, iterations + 1):

            # ---------------------
            #  Train D_real
            # ---------------------
            for _ in range(train_D_iters):
                # Select a random half batch of images
                idx_real = np.random.randint(0, imgs.shape[0], batch_size)
                # idx_fake = np.random.randint(0, imgs.shape[0], batch_size)
                fake_target_digits = onehot( np.random.randint(0, 10, batch_size), 10 )
                # unmatch_digits = onehot( exclude(digits[idx_real]), 10 )
                real_imgs = imgs[idx_real]
                # real_digits = onehot( digits[idx_real], 10 )
                # fake_imgs = self.G.predict([imgs[idx_fake], fake_target_digits])

                d_loss_real = self.D_combined.train_on_batch([real_imgs, fake_target_digits],
                                                             [valid, fake, dummy])
                # # real image
                # d_loss_real = self.D_real.train_on_batch(real_imgs, valid)
                # # fake image
                # d_loss_fake = self.D_real.train_on_batch(fake_imgs, fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            for _ in range(train_G_iters):
                # Condition on labels
                idx = np.random.randint(0, imgs.shape[0], batch_size)
                fake_target_digits = onehot( np.random.randint(0, 10, batch_size), 10 )

                g_loss = self.G_combined.train_on_batch([imgs[idx], fake_target_digits], [valid, fake_target_digits])

            print('D')
            print(pd.DataFrame({
                'metrics': self.D_combined.metrics_names,
                'loss': d_loss_real,
            }))
            print('G')
            print(pd.DataFrame({
                'metrics': self.G_combined.metrics_names,
                'loss': g_loss,
            }))
            print()

            # If at save interval => save generated image samples
            if sample_interval > 0 and itr % sample_interval == 0:
                # self.sample_imgs(itr, img_dir)
                plot_table(self.G, self.D_combined, os.path.join(img_dir, f'{itr}.png'), save=True)

            if save_model_interval > 0 and itr % save_model_interval == 0:
                self.D_combined.save(os.path.join(model_dir, f'D{itr}.hdf5'))
                self.G.save(os.path.join(model_dir, f'G{itr}.hdf5'))
                self.G_mask.save(os.path.join(model_dir, f'G_mask{itr}.hdf5'))

        self.tb.on_train_end(None)


if __name__ == '__main__':

    ver_name = 'wgan_gp_testing'
    model = GAN(loss_weight=[1,5])
    model.train(
            iterations=20000,
            batch_size=128,
            sample_interval=500,
            save_model_interval=1000,
            train_D_iters=5,
            train_G_iters=1,
            img_dir=f'./outputs/{ver_name}/imgs',
            model_dir=f'./outputs/{ver_name}/models')

