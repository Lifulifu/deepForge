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

import matplotlib.pyplot as plt
import os
import h5py
import pandas as pd

import numpy as np
import tensorflow as tf
from util import load_mnist, onehot, plot_table
from model import build_discriminator_realness, build_discriminator_digit, build_generator

def exclude(arr):
    result = [ np.random.choice(list({0,1,2,3,4,5,6,7,8,9}-{digit}), 1)[0] for digit in arr ]
    return np.array(result)

class GAN():
    def __init__(self, model_name=None, loss_weight=[1,1]):

        self.imgs, self.digits, self.test_imgs, self.test_digits = load_mnist()
        self.img_rows, self.img_cols, self.channels = self.imgs.shape[1:]
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer_D = Adam(lr=0.0002)
        optimizer_G = Adam(lr=0.0002)

        self.D_real = build_discriminator_realness()
        self.D_real.compile(
            loss='binary_crossentropy',
            optimizer=optimizer_D,
            metrics=['binary_accuracy'])

        self.D_digit = load_model('outputs/D_digit.hdf5')

        input_img = Input(shape=self.img_shape)
        output_realness = self.D_real(input_img)
        output_digit = self.D_digit(input_img)
        self.D = Model(input_img, [output_realness, output_digit])
        self.D.compile(
            loss=['binary_crossentropy', 'categorical_crossentropy'],
            loss_weights=loss_weight,
            optimizer=optimizer_D,
            metrics=['binary_accuracy', 'categorical_accuracy'])
        self.D.summary()

        self.G, self.G_mask = build_generator()

        img_input = Input(shape=self.img_shape)
        digit_input = Input(shape=(10,))
        img_added = self.G([img_input, digit_input])

        self.D.trainable = False
        D_output = self.D(img_added)
        self.combined = Model([img_input, digit_input], D_output)
        self.combined.compile(
            loss=['binary_crossentropy', 'categorical_crossentropy'],
            loss_weights=[1, 1],
            optimizer=optimizer_G,
            metrics=['binary_accuracy', 'categorical_accuracy'])
        self.combined.summary()

        self.tb = keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=0,
            batch_size=64,
            write_graph=True,
            write_grads=True
        )
        self.tb.set_model(self.combined)

    def train(self, iterations, batch_size=128, sample_interval=100, save_model_interval=100,
                            train_D_iters=1, train_G_iters=1, img_dir='./', model_dir='./'):

        imgs, digits = self.imgs, self.digits

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        for itr in range(1, iterations + 1):

            # ---------------------
            #  Train D_real
            # ---------------------
            for _ in range(train_D_iters):
                # Select a random half batch of images
                idx_real = np.random.randint(0, imgs.shape[0], batch_size)
                idx_fake = np.random.randint(0, imgs.shape[0], batch_size)
                fake_target_digits = onehot( np.random.randint(0, 10, batch_size), 10 )
                unmatch_digits = onehot( exclude(digits[idx_real]), 10 )
                real_imgs = imgs[idx_real]
                real_digits = onehot( digits[idx_real], 10 )
                fake_imgs = self.G.predict([imgs[idx_fake], fake_target_digits])

                # real image
                d_loss_real = self.D_real.train_on_batch(real_imgs, valid)
                # fake image
                d_loss_fake = self.D_real.train_on_batch(fake_imgs, fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            for _ in range(train_G_iters):
                # Condition on labels
                idx = np.random.randint(0, imgs.shape[0], batch_size)
                fake_target_digits = onehot( np.random.randint(0, 10, batch_size), 10 )

                g_loss = self.combined.train_on_batch([imgs[idx], fake_target_digits], [valid, fake_target_digits])

            print(pd.DataFrame({
                'metrics': self.D_real.metrics_names,
                'real': d_loss_real,
                'fake': d_loss_fake
            }))
            print(pd.DataFrame({
                'metrics': self.combined.metrics_names,
                'loss': g_loss,
            }))
            print()

            # If at save interval => save generated image samples
            if sample_interval > 0 and itr % sample_interval == 0:
                # self.sample_imgs(itr, img_dir)
                plot_table(self.G, self.D, os.path.join(img_dir, f'{itr}.png'), save=True)

            if save_model_interval > 0 and itr % save_model_interval == 0:
                self.D.save(os.path.join(model_dir, f'D{itr}.hdf5'))
                self.G.save(os.path.join(model_dir, f'G{itr}.hdf5'))
                self.G_mask.save(os.path.join(model_dir, f'G_mask{itr}.hdf5'))

        self.tb.on_train_end(None)


if __name__ == '__main__':

    ver_name = 'sep_1:5'
    model = GAN(loss_weight=[1,5])
    model.train(
            iterations=20000,
            batch_size=128,
            sample_interval=1000,
            save_model_interval=1000,
            train_D_iters=1,
            train_G_iters=1,
            img_dir=f'./outputs/{ver_name}/imgs',
            model_dir=f'./outputs/{ver_name}/models')

