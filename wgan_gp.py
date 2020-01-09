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
from keras.layers.merge import _Merge
from functools import partial

import matplotlib.pyplot as plt
import os
import sys

import numpy as np
import tensorflow as tf

GRADIENT_PENALTY_WEIGHT = 10

def exclude(arr):
    result = [ np.random.choice(list({0,1,2,3,4,5,6,7,8,9}-{digit}), 1)[0] for digit in arr ]
    return np.array(result)

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

def tile_images(image_stack):
    """Given a stacked tensor of images, reshapes them into a horizontal tiling for
    display."""
    assert len(image_stack.shape) == 3
    image_list = [image_stack[i, :, :] for i in range(image_stack.shape[0])]
    tiled_images = np.concatenate(image_list, axis=1)
    return tiled_images

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((128, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class WGANGP():
    def __init__(self):

        self.imgs, self.digits, self.test_imgs, self.test_digits = load_mnist()
        self.img_rows, self.img_cols, self.channels = self.imgs.shape[1:]
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        loss_func = 'binary_crossentropy'

        self.n_critic = 5
        optimizer_D = RMSprop(lr=0.00005)
        optimizer_G = RMSprop(lr=0.00005)

        discriminator = self.build_discriminator()
        self.G, self.mask_G = self.build_generator()

        ## generator {{{
        for layer in discriminator.layers:
            layer.trainable = False
        discriminator.trainable = False
        G_img_input = Input(shape=self.img_shape)
        G_digit_input = Input(shape=(10,))
        img_added = self.G([G_img_input, G_digit_input])
        discriminator_for_generator = discriminator([img_added, G_digit_input])
        self.combined = Model(inputs=[G_img_input, G_digit_input],
                       outputs=[discriminator_for_generator])
        self.combined.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                       loss=wasserstein_loss)
        self.combined.summary()
        # }}}

        ## discriminator{{{
        for layer in discriminator.layers:
            layer.trainable = True
        for layer in self.G.layers:
            layer.trainable = False
        discriminator.trainable = True
        self.G.trainable = False

        real_input = Input(shape=self.img_shape)
        digit_input = Input(shape=(10,))
        fake_samples = self.G([real_input, digit_input])
        discriminator_output_from_real = discriminator([real_input, digit_input])
        discriminator_output_from_fake = discriminator([fake_samples, digit_input])

        averaged_samples = RandomWeightedAverage()([real_input, fake_samples])
        discriminator_output_from_average = discriminator([averaged_samples, digit_input])

        partial_gp_loss = partial(gradient_penalty_loss,
                                  averaged_samples=averaged_samples,
                                  gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
        partial_gp_loss.__name__ = 'gradient_penalty'

        self.D = Model(inputs=[real_input, digit_input],
                       outputs=[discriminator_output_from_real,
                                discriminator_output_from_fake,
                                discriminator_output_from_average])
        self.D.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                       loss=[wasserstein_loss,
                             wasserstein_loss,
                             partial_gp_loss])
        self.D.summary()
        # }}}

        self.tb = keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=0,
            batch_size=64,
            write_graph=True,
            write_grads=True
        )
        self.tb.set_model(self.combined)


    def build_generator(self):# {{{
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
# }}}

    def build_discriminator(self):# {{{
        # -----
        # input: 32*32*1 image + target digit one hot
        # output: 0 ~ 1
        # -----

        img_input = Input(shape=(32, 32, 1))
        digit_input = Input(shape=(10,))

        x = Conv2D(16, (3,3), kernel_initializer='he_normal', padding='same')(img_input)
        x = BatchNormalization(momentum=0.8)(x)
        x = MaxPooling2D((2,2))(x) # 16,16
        x = LeakyReLU(alpha=0.1)(x)

        x = Conv2D(32, (3,3), kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = MaxPooling2D((2,2))(x) # 8, 8
        x = LeakyReLU(alpha=0.1)(x)

        x = Conv2D(64, (3,3), kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = MaxPooling2D((2,2))(x) # 4, 4
        x = LeakyReLU(alpha=0.1)(x)

        x = Conv2D(128, (3,3), kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = MaxPooling2D((2,2))(x) # 2, 2
        x = LeakyReLU(alpha=0.1)(x)
        x = Flatten()(x)

        x = Concatenate()([x, digit_input])
        x = LeakyReLU(alpha=0.1)(x)
        x = Dense(128, kernel_initializer='he_normal')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dense(64, kernel_initializer='he_normal')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dense(16, kernel_initializer='he_normal')(x)
        out = Dense(1, kernel_initializer='he_normal')(x)

        model = Model([img_input, digit_input], out, name='D')

        return model
# }}}

    def train(self, iterations, batch_size=128, sample_interval=100,
                            train_D_iters=1, train_G_iters=1, img_dir='./imgs'):


        imgs, digits = self.imgs, self.digits
        imgs = (imgs.astype(np.float32) - .5) * 2

        valid = np.ones((batch_size, 1), dtype=np.float32)
        fake = -valid
        dummy = np.zeros((batch_size, 1), dtype=np.float32)

        for itr in range(1, iterations + 1):

            # ---------------------
            #  Train Discriminator
            # ---------------------
            for _ in range(train_D_iters):
                # Select a random half batch of images
                idxs1 = np.random.randint(0, imgs.shape[0], batch_size)
                idxs2 = np.random.randint(0, imgs.shape[0], batch_size)
                match_digits = onehot( digits[idxs1], 10 )
                unmatch_digits = onehot( exclude(digits[idxs2]), 10 )

                # matched
                real_imgs_m, digits_m = imgs[idxs1], match_digits

                # unmatched
                real_imgs_u, digits_u = imgs[idxs2], unmatch_digits

                # d_loss_real = [loss, acc]
                d_loss_match = self.D.train_on_batch([real_imgs_m, digits_m],
                                                     [valid, fake, dummy])
                d_loss_unmatch = self.D.train_on_batch([real_imgs_u, digits_u],
                                                       [fake, fake, dummy])

            d_loss = 0.5 * np.add(d_loss_match, d_loss_unmatch)

            # tensorboard
            logs = {
                'D_loss_real': d_loss_match[0],
                'D_loss_fake': d_loss_unmatch[0]
            }
            self.tb.on_epoch_end(itr, logs)

            # if itr % 100 == 0:
            #     real_res = self.D.predict([real_imgs, real_digits]).flatten()
            #     fake_res = self.D.predict([fake_imgs, random_target_digits]).flatten()
            #     print('\n------------------------------------')
            #     print('real', real_res[-20:])
            #     print('fake', fake_res[-20:])
            #     print('------------------------------------\n')


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
                    'G_loss': g_loss,
                }
                self.tb.on_epoch_end(itr, logs)


            # If at save interval => save generated image samples
            if sample_interval > 0 and itr % sample_interval == 0:
                self.sample_imgs(itr, img_dir)

            # Plot the progress
            print(f'{itr} [G loss: {g_loss}]')
            print(f'{itr} [D real: {d_loss_match[0]} | {d_loss_match[1]}]')
            print(f'{itr} [D fake: {d_loss_unmatch[0]} | {d_loss_unmatch[1]}]')

        self.tb.on_train_end(None)


    def sample_imgs(self, itr, img_dir):
        n = 5
        targets = onehot(np.array([4] * n), 10)

        print(self.test_imgs.shape)
        gen_imgs = self.G.predict([self.test_imgs[:n], targets])
        print(gen_imgs.shape)
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

    model = WGANGP()
    model.train(iterations=10000,
            batch_size=128,
            sample_interval=100,
            train_D_iters=5,
            train_G_iters=1,
            img_dir='./outputs/wgan_gp_testing')

