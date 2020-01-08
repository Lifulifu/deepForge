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
from model import build_discriminator_11_class, build_generator

def exclude(arr):
    result = [ np.random.choice(list({0,1,2,3,4,5,6,7,8,9}-{digit}), 1)[0] for digit in arr ]
    return np.array(result)

class GAN():
    def __init__(self, model_name=None):

        self.imgs, self.digits, self.test_imgs, self.test_digits = load_mnist()
        self.img_rows, self.img_cols, self.channels = self.imgs.shape[1:]
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer_D = Adam(lr=0.0002)
        optimizer_G = Adam(lr=0.0002)

        self.D = build_discriminator_11_class()
        self.D.name = 'D'
        self.D.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer_D,
            metrics=['accuracy'])
        self.D.summary()

        self.G, self.G_mask = build_generator()
        img_input = Input(shape=self.img_shape)
        digit_input = Input(shape=(10,))
        img_added = self.G([img_input, digit_input])

        self.D.trainable = False
        D_output = self.D(img_added)
        self.combined = Model([img_input, digit_input], D_output, name='combined')
        self.combined.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer_G,
            metrics=['accuracy'])
        self.combined.summary()

    def train(self, iterations, batch_size=128, sample_interval=100, save_model_interval=100,
                            train_D_iters=1, train_G_iters=1, img_dir='./', model_dir='./'):

        imgs, digits = self.imgs, self.digits

        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        for itr in range(1, iterations + 1):

            # ---------------------
            #  Train D_realness
            # ---------------------
            for _ in range(train_D_iters):
                # Select a random half batch of images
                idx_real = np.random.randint(0, imgs.shape[0], batch_size)
                idx_fake = np.random.randint(0, imgs.shape[0], batch_size)

                real_imgs = imgs[idx_real]
                real_ans = onehot( digits[idx_real], 11 ) # 11 class for D's output

                fake_target_digits = onehot( np.random.randint(0, 10, batch_size), 10 ) # for G's input
                fake_imgs = self.G.predict([imgs[idx_fake], fake_target_digits])
                fake_ans = np.zeros((batch_size, 11))
                fake_ans[:, 10] = 1 # only the 11th class is 1

                # real image 
                d_loss_real = self.D.train_on_batch(real_imgs, real_ans)
                # fake image 
                d_loss_fake = self.D.train_on_batch(fake_imgs, fake_ans)

            # ---------------------
            #  Train Generator
            # ---------------------
            for _ in range(train_G_iters):
                # Condition on labels
                idx = np.random.randint(0, imgs.shape[0], batch_size)
                target_nums = np.random.randint(0, 10, batch_size)
                fake_target_digits = onehot(target_nums, 10)
                fake_ans = onehot(target_nums, 11)

                g_loss = self.combined.train_on_batch([imgs[idx], fake_target_digits], fake_ans)

            print(f'--------\nEPOCH {itr}\n--------')

            print(pd.DataFrame({
                'D': self.D.metrics_names,
                'real': d_loss_real,
                'fake': d_loss_fake
            }).to_string(index=False))
            print()

            print(pd.DataFrame({
                'G': self.combined.metrics_names,
                'value': g_loss,
            }).to_string(index=False))
            print()

            # If at save interval => save generated image samples
            if sample_interval > 0 and itr % sample_interval == 0:
                # self.sample_imgs(itr, img_dir)
                plot_table(self.G, self.D, os.path.join(img_dir, f'{itr}.png'), save=True)

            if save_model_interval > 0 and itr % save_model_interval == 0:
                self.D.save(os.path.join(model_dir, f'D{itr}.hdf5'))
                self.G.save(os.path.join(model_dir, f'G{itr}.hdf5'))
                self.G_mask.save(os.path.join(model_dir, f'G_mask{itr}.hdf5'))


if __name__ == '__main__':

    ver_name = 'D_11class_G4D1'
    model = GAN()
    model.train(
            iterations=20000,
            batch_size=128,
            sample_interval=1000,
            save_model_interval=1000,
            train_G_iters=4,
            train_D_iters=1,
            img_dir=f'./outputs/{ver_name}/imgs',
            model_dir=f'./outputs/{ver_name}/models')

