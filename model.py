from keras.layers import Input, Dense, Conv2D, Reshape, Flatten, Dropout, multiply, MaxPooling2D
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Concatenate, Lambda, Add
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
import keras.backend as K
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam, RMSprop
from keras import metrics

def build_discriminator():# {{{
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
    out = Dense(1, activation='sigmoid')(x)

    model = Model([img_input, digit_input], out, name='D')

    return model
# }}}

def build_generator():# {{{
    # -----
    # input: 32*32*1 image (0~1) + target digit one hot
    # output: 32*32*1 generated image (-1~1)
    # -----

    img_input = Input(shape=(32, 32, 1))
    digit_input = Input(shape=(10,))

    x1 = conv2d_bn(img_input, 16, (3, 3), max_pool=True) #16, 16

    x2 = conv2d_bn(x1, 32, (3, 3), max_pool=True) #8, 8

    x3 = conv2d_bn(x2, 64, (3, 3), max_pool=True) #4, 4

    x4 = conv2d_bn(x3, 64, (3, 3), max_pool=True) #2, 2
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
    img_added = Lambda(lambda x: K.clip(x, 0, 1))(img_added)

    model = Model([img_input, digit_input], img_added, name='G')
    model_mask = Model([img_input, digit_input], mask, name='G_mask')

    return model, model_mask
# }}}

def build_generator_incep():# {{{
    # -----
    # input: 32*32*1 image (0~1) + target digit one hot
    # output: 32*32*1 generated image (-1~1)
    # -----

    img_input = Input(shape=(32, 32, 1))
    digit_input = Input(shape=(10,))
    x = conv2d_bn(img_input, 16, (3, 3))
    x = conv2d_bn(x, 16, (3, 3))

    x = inception_block(x, 16, 'B')
    x1 = conv2d_bn(x, 16, (3, 3), max_pool=True) #16, 16

    x = inception_block(x1, 32, 'B')
    x2 = conv2d_bn(x, 32, (3, 3), max_pool=True) #8, 8

    x = inception_block(x2, 64, 'A')
    x3 = conv2d_bn(x, 64, (3, 3), max_pool=True) #4, 4

    x = inception_block(x3, 64, 'C')
    x4 = conv2d_bn(x, 64, (3, 3), max_pool=True) #2, 2
    x = conv2d_bn(x4, 64, (2, 2), padding='valid')
    x = Flatten()(x)

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
    img_added = Lambda(lambda x: K.clip(x, 0, 1))(img_added)

    model = Model([img_input, digit_input], img_added, name='G')
    model_mask = Model([img_input, digit_input], mask, name='G_mask')

    return model, model_mask
# }}}

def conv2d_bn(x, filters,
              kernel_size,
              max_pool=False,
              strides=1,
              padding='same'):
    x = Conv2D(filters, kernel_size,
               strides=strides,
               padding=padding,
               use_bias=False)(x)
    x = BatchNormalization(momentum=0.8)(x)
    if max_pool:
        x = MaxPooling2D((2,2))(x) # 16,16
    x = LeakyReLU(alpha=0.1)(x)
    return x

def inception_block(x, out, module='A'):

    # module A
    if 'A' == module:
        branch133 = conv2d_bn(x, out, (1, 1))
        branch133 = conv2d_bn(branch133, out, (3, 3))
        branch133 = conv2d_bn(branch133, out, (3, 3))

        branch13 = conv2d_bn(x, out, (1, 1))
        branch13 = conv2d_bn(branch13, out, (3, 3))

        branch1x1 = conv2d_bn(x, out, (1, 1))
        x = Concatenate()([x, branch133, branch13, branch1x1])

    # module B
    if 'B' == module:
        branch17777 = conv2d_bn(x, out, (1, 1))
        branch17777 = conv2d_bn(branch17777, out, (1, 7))
        branch17777 = conv2d_bn(branch17777, out, (7, 1))
        branch17777 = conv2d_bn(branch17777, out, (1, 7))
        branch17777 = conv2d_bn(branch17777, out, (7, 1))

        branch177 = conv2d_bn(x, out, (1, 1))
        branch177 = conv2d_bn(branch177, out, (1, 7))
        branch177 = conv2d_bn(branch177, out, (7, 1))

        branch1x1 = conv2d_bn(x, out, (1, 1))
        x = Concatenate()([x, branch17777, branch177, branch1x1])

    # module C
    if 'C' == module:
        branch133 = conv2d_bn(x, out, (1, 1))
        branch133 = conv2d_bn(branch133, out, (3, 3))
        branch133_1 = conv2d_bn(branch133, out, (3, 1))
        branch133_2 = conv2d_bn(branch133, out, (1, 3))

        branch13 = conv2d_bn(x, out, (1, 1))
        branch13 = conv2d_bn(branch13, out, (1, 3))
        branch13_ = conv2d_bn(branch13, out, (3, 1))

        branch1x1 = conv2d_bn(x, out, (1, 1))
        x = Concatenate()([x, branch133, branch13, branch1x1])

    return conv2d_bn(x, out, (1, 1))
