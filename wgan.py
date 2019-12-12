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

import numpy as np
import tensorflow as tf
from util import load_mnist, onehot
from model import conv2d_bn, build_generator, build_generator_incep

def exclude(arr):
    result = [ np.random.choice(list({0,1,2,3,4,5,6,7,8,9}-{digit}), 1)[0] for digit in arr ]
    return np.array(result)

def build_discriminator():
    # -----
    # input: 32*32*1 image + target digit one hot
    # output: 0 ~ 1
    # -----

    img_input = Input(shape=(32, 32, 1))
    digit_input = Input(shape=(10,))

    x = Conv2D(16, (3,3), padding='same')(img_input)
    x = BatchNormalization(momentum=0.8)(x)
    x = MaxPooling2D((2,2))(x) # 16,16
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(32, (3,3), padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = MaxPooling2D((2,2))(x) # 8, 8
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(64, (3,3), padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = MaxPooling2D((2,2))(x) # 4, 4
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(128, (3,3), padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = MaxPooling2D((2,2))(x) # 2, 2
    x = LeakyReLU(alpha=0.1)(x)
    x = Flatten()(x)

    x = Concatenate()([x, digit_input])
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(64)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(32)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(16)(x)
    out = Dense(1)(x) # no activation for wgan

    model = Model([img_input, digit_input], out, name='D')

    return model

class WGAN():
    def __init__(self, model_name=None):

        self.imgs, self.digits, self.test_imgs, self.test_digits = load_mnist()
        self.img_rows, self.img_cols, self.channels = self.imgs.shape[1:]
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.loss_func = self.wasserstein_loss
        self.clip_value = 0.01

        optimizer_D = RMSprop(lr=0.00005)
        optimizer_G = RMSprop(lr=0.00005)

        self.D = build_discriminator()
        self.D.compile(loss=self.loss_func,
            optimizer=optimizer_D,
            metrics=[metrics.binary_accuracy])
        self.D.summary()
        self.G, self.G_mask = build_generator_incep()

        img_input = Input(shape=self.img_shape)
        digit_input = Input(shape=(10,))
        img_added = self.G([img_input, digit_input])

        self.D.trainable = False
        D_output = self.D([img_added, digit_input])
        self.combined = Model([img_input, digit_input], D_output)
        self.combined.compile(loss=self.loss_func,
            optimizer=optimizer_G,
            metrics=[metrics.binary_accuracy])
        self.combined.summary()

        self.tb = keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=0,
            batch_size=64,
            write_graph=True,
            write_grads=True
        )
        self.tb.set_model(self.combined)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def train(self, iterations, batch_size=128, sample_interval=100, save_model_interval=100,
                            train_D_iters=1, train_G_iters=1, img_dir='./imgs', model_dir='./models'):

        imgs, digits = self.imgs, self.digits

        valid = np.ones((batch_size, 1))
        fake = -np.ones((batch_size, 1))

        for itr in range(1, iterations + 1):

            # ---------------------
            #  Train Discriminator
            # ---------------------
            for _ in range(train_D_iters):
                # Select a random half batch of images
                idx_real = np.random.randint(0, imgs.shape[0], batch_size)
                idx_fake = np.random.randint(0, imgs.shape[0], batch_size)
                random_target_digits = onehot( np.random.randint(0, 10, batch_size), 10 )
                unmatch_digits = onehot( exclude(digits[idx_real]), 10 )
                real_imgs, real_digits = imgs[idx_real], onehot( digits[idx_real], 10 )
                fake_imgs = self.G.predict([imgs[idx_fake], random_target_digits])

                # real image and correct digit
                d_loss_real = self.D.train_on_batch([real_imgs, real_digits], valid)
                # fake image and random digit
                d_loss_fake = self.D.train_on_batch([fake_imgs, random_target_digits], fake)
                # real image but wrong digit
                d_loss_fake2 = self.D.train_on_batch([real_imgs, unmatch_digits], fake)

                # Clip critic weights
                for l in self.D.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

            # tensorboard
            logs = {
                'D_loss_real': d_loss_real[0],
                'D_loss_fake': d_loss_fake[0],
                'D_loss_fake2': d_loss_fake2[0]
            }
            self.tb.on_epoch_end(itr, logs)

            # ---------------------
            #  Train Generator
            # ---------------------
            for _ in range(train_G_iters):
                # Condition on labels
                idx = np.random.randint(0, imgs.shape[0], batch_size)
                random_target_digits = onehot( np.random.randint(0, 10, batch_size), 10 )

                g_loss = self.combined.train_on_batch([imgs[idx], random_target_digits], valid)

                # tensorboard
                logs = {
                    'G_loss': g_loss[0],
                }
                self.tb.on_epoch_end(itr, logs)


            # If at save interval => save generated image samples
            if sample_interval > 0 and itr % sample_interval == 0:
                self.sample_imgs(itr, img_dir)

            if save_model_interval > 0 and itr % save_model_interval == 0:
                if not os.path.isdir(model_dir):
                    os.makedirs(model_dir)
                self.D.save(os.path.join(model_dir, f'D{itr}.hdf5'))
                self.G.save(os.path.join(model_dir, f'G{itr}.hdf5'))
                self.G_mask.save(os.path.join(model_dir, f'G_mask{itr}.hdf5'))

            # Plot the progress
            print(f'{itr} [G loss: {g_loss[0]} | acc: {g_loss[1]}]')
            print(f'{itr} [D real: {d_loss_real[0]} | acc: {d_loss_real[1]}]')
            print(f'{itr} [D fake: {d_loss_fake[0]} | acc: {d_loss_fake[1]}]')
            print(f'{itr} [D fake2: {d_loss_fake2[0]} | acc: {d_loss_fake2[1]}]')
            print()

        self.tb.on_train_end(None)

    def sample_imgs(self, itr, img_dir):
        n = 5
        targets = onehot( np.full((n, 1), 4), 10 )
        test_imgs = self.test_imgs[:n]

        gen_imgs = self.G.predict([test_imgs, targets])
        masks = self.G_mask.predict([test_imgs, targets])
        D_pred_T = self.D.predict([test_imgs, targets])
        D_pred_F = self.D.predict([gen_imgs, targets])

        fig, axs = plt.subplots(n, 3, figsize=(8, 6))
        fig.tight_layout()
        for i in range(n):
            for no, img in enumerate([test_imgs, masks, gen_imgs]):
                axs[i, no].imshow(img[i, :, :, 0], cmap='gray')
                axs[i, no].axis('off')
                if 0 == no:
                    axs[i, no].text(-20, -2, f'D_pred_T: {D_pred_T[i]}')
                elif 2 == no:
                    axs[i, no].text(-20, -2, f'D_pred_F: {D_pred_F[i]}')
        if not os.path.isdir(img_dir):
            os.makedirs(img_dir)
        fig.savefig(os.path.join(img_dir, f'{itr}.png'))
        plt.close()


if __name__ == '__main__':


    model = WGAN()
    model.train(
            iterations=50000,
            batch_size=128,
            sample_interval=2000,
            save_model_interval=2000,
            train_D_iters=1,
            train_G_iters=1,
            img_dir=f'./output/wgan_G1D1/imgs/',
            model_dir=f'./output/wgan_G1D1/models/')

