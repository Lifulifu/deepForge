#!/usr/bin/env/ python3

from __future__ import print_function, division

import keras
import keras.backend as K
from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D, Reshape, Flatten, Dropout, multiply, MaxPooling2D
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Concatenate, Lambda, Add
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.backend import clip
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from keras import metrics

import matplotlib.pyplot as plt
import os

import numpy as np
import tensorflow as tf

def load_mnist():
    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
    # pad to 32*32 and normalize to 0~1
    xtrain = np.pad(xtrain, ((0,0),(2,2),(2,2)), 'constant') / 255
    xtest = np.pad(xtest, ((0,0),(2,2),(2,2)), 'constant') / 255
    # expand channel dim
    xtrain, xtest = xtrain[:, :, :, np.newaxis], xtest[:, :, :, np.newaxis]

    return xtrain, ytrain, xtest, ytest

def onehot(x, size):
    result = np.zeros((x.size, size))
    result[np.arange(x.size), x] = 1
    return result

class WGAN():
    def __init__(self):

        self.imgs, self.digits, self.test_imgs, self.test_digits = load_mnist()
        self.img_rows, self.img_cols, self.channels = self.imgs.shape[1:]
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.loss_func = self.wasserstein_loss
        self.clip = 0.01

        optimizer_D = RMSprop(lr=0.00005)
        optimizer_G = RMSprop(lr=0.00005)

        self.D = self.build_discriminator()
        self.D.compile(loss=self.loss_func,
            optimizer=optimizer_D,
            metrics=['accuracy'])
        self.D.summary()
        self.G, self.G_mask = self.build_generator()

        img_input = Input(shape=self.img_shape)
        digit_input = Input(shape=(10,))
        img_added = self.G([img_input, digit_input])

        self.D.trainable = False
        D_output = self.D([img_added, digit_input])
        self.combined = Model([img_input, digit_input], D_output)
        self.combined.compile(loss=self.loss_func, 
            optimizer=optimizer_G,
            metrics=['accuracy'])
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

    def build_generator(self):
        # -----
        # input: 32*32*1 image (0~1) + target digit one hot
        # output: 32*32*1 generated image (-1~1)
        # -----

        img_input = Input(shape=(32, 32, 1))
        digit_input = Input(shape=(10,))

        x = Conv2D(16, (3,3), padding='same')(img_input)
        x = BatchNormalization(momentum=0.8)(x)
        x = MaxPooling2D((2,2))(x) # 16,16
        x1 = LeakyReLU(alpha=0.1)(x)

        x = Conv2D(32, (3,3), padding='same')(x1)
        x = BatchNormalization(momentum=0.8)(x)
        x = MaxPooling2D((2,2))(x) # 8, 8
        x2 = LeakyReLU(alpha=0.1)(x)

        x = Conv2D(64, (3,3), padding='same')(x2)
        x = BatchNormalization(momentum=0.8)(x)
        x = MaxPooling2D((2,2))(x) # 4, 4
        x3 = LeakyReLU(alpha=0.1)(x)

        x = Conv2D(64, (3,3), padding='same')(x3)
        x = BatchNormalization(momentum=0.8)(x)
        x = MaxPooling2D((2,2))(x) # 2, 2
        x4 = LeakyReLU(alpha=0.1)(x)
        # x = Conv2D(64, (2, 2), padding='valid')
        # x = BatchNormalization(momentum=0.8)(x)
        x = Flatten()(x4)

        x = Concatenate()([x, digit_input])
        x = LeakyReLU(alpha=0.1)(x)
        x = Dense(128)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dense(64)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dense(2*2*64)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = Reshape((2, 2, 64))(x)
        x = Concatenate()([x, x4])
        x = UpSampling2D((2,2))(x) # 4, 4

        x = Conv2D(64, (3,3), padding='same')(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Concatenate()([x, x3])
        x = UpSampling2D((2,2))(x) # 8, 8

        x = Conv2D(32, (3,3), padding='same')(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Concatenate()([x, x2])
        x = UpSampling2D((2,2))(x) # 16, 16

        x = Conv2D(16, (3,3), padding='same')(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Concatenate()([x, x1])
        x = UpSampling2D((2,2))(x) # 32, 32

        mask = Conv2D(1, (3,3), padding='same', activation='tanh')(x)

        mask = Lambda(lambda x: (x + 1) * 0.5)(mask)
        img_added = Add()([img_input, mask])
        img_added = Lambda(lambda x: clip(x, 0, 1))(img_added)

        model = Model([img_input, digit_input], img_added, name='G')
        model_mask = Model([img_input, digit_input], mask, name='G')

        return model, model_mask


    def build_discriminator(self):
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
        out = Dense(1)(x)

        model = Model([img_input, digit_input], out, name='D')

        return model


    def train(self, iterations, batch_size=128, sample_interval=100,
                            train_D_iters=1, train_G_iters=1, img_dir='./imgs'):

        imgs, digits = self.imgs, self.digits
        imgs = (imgs.astype(np.float32) - .5) * 2

        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1)) # fake = 1, true = -1

        for itr in range(1, iterations + 1):

            # ---------------------
            #  Train Discriminator
            # ---------------------
            for _ in range(train_D_iters):
                # Select a random half batch of images
                idx_real = np.random.randint(0, imgs.shape[0], batch_size)
                idx_fake = np.random.randint(0, imgs.shape[0], batch_size)
                random_target_digits = onehot( np.random.randint(0, 10, batch_size), 10 )

                real_imgs, real_digits = imgs[idx_real], onehot( digits[idx_real], 10 )
                fake_imgs = self.G.predict([imgs[idx_fake], random_target_digits])

                # d_loss_real = [loss, acc]
                d_loss_real = self.D.train_on_batch([real_imgs, real_digits], valid)
                d_loss_fake = self.D.train_on_batch([fake_imgs, random_target_digits], fake)

                # Clip D weights
                for l in self.D.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip, self.clip) for w in weights]
                    l.set_weights(weights)

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # tensorboard
            logs = {
                'D_loss_real': d_loss_real[0],
                'D_loss_fake': d_loss_fake[0]
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

            # Plot the progress
            print(f'{itr} [G loss: {g_loss[0]} | acc: {g_loss[1]}]')
            print(f'{itr} [D real: {d_loss_real[0]} | acc: {d_loss_real[1]}]')
            print(f'{itr} [D fake: {d_loss_fake[0]} | acc: {d_loss_fake[1]}]')

        self.tb.on_train_end(None)


    def sample_imgs(self, itr, img_dir):
        n = 5
        targets = onehot(np.array([4] * n), 10)

        gen_imgs = self.G.predict([self.test_imgs[:n], targets])
        # gen_imgs = self.test_imgs[:n] + (gen_imgs + 1) * 0.5
        # # Rescale images 0 - 1
        # gen_imgs = np.clip(gen_imgs, 0, 1)

        fig, axs = plt.subplots(n, 2)
        for i in range(n):
            axs[i, 0].imshow(self.test_imgs[i,:,:,0], cmap='gray')
            axs[i, 0].axis('off')
            axs[i, 1].imshow(gen_imgs[i,:,:,0], cmap='gray')
            axs[i, 1].axis('off')
        if not os.path.isdir(img_dir):
            os.makedirs(img_dir)
        fig.savefig(os.path.join(img_dir, f'{itr}.png'))
        plt.close()


if __name__ == '__main__':

    model = WGAN()
    model.train(iterations=10000,
            batch_size=128,
            sample_interval=500,
            train_D_iters=5,
            train_G_iters=1,
            img_dir='./imgs/wgan_G1_D5')
