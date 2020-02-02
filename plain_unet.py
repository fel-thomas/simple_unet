#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# 2020 Thomas FEL.
"""Plain Unet network to copy paste (keras & tensorflow 2.X)."""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation,\
                                    BatchNormalization, Dropout, MaxPooling2D,\
                                    Conv2D, concatenate, Conv2DTranspose
# Unet inspired model
# for the original see @https://arxiv.org/abs/1505.04597
#
# the detailed aspect of the layers is done on purpose, in order to capture
# the architecture of Unet
# 4 levels of encoder/decoder and batchnorm after each conv layer (except last)
BASE_FILTERS = 32
INPUT_SIZE   = (128, 128, 3)
DROPOUT      = 0.4
ACTIVATION   = 'relu'
INITIALIZER  = 'he_normal'

input_layer = Input(INPUT_SIZE)

c1 = Conv2D(BASE_FILTERS, (3, 3), activation=ACTIVATION, kernel_initializer=INITIALIZER, padding='same') (input_layer)
c1 = BatchNormalization()(c1)
c1 = Dropout(DROPOUT)(c1)
c1 = Conv2D(BASE_FILTERS, (3, 3), activation=ACTIVATION, kernel_initializer=INITIALIZER, padding='same') (c1)
c1 = BatchNormalization()(c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(BASE_FILTERS * 2, (3, 3), activation=ACTIVATION, kernel_initializer=INITIALIZER, padding='same') (p1)
c2 = BatchNormalization()(c2)
c2 = Dropout(DROPOUT)(c2)
c2 = Conv2D(BASE_FILTERS * 2, (3, 3), activation=ACTIVATION, kernel_initializer=INITIALIZER, padding='same') (c2)
c2 = BatchNormalization()(c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(BASE_FILTERS * 4, (3, 3), activation=ACTIVATION, kernel_initializer=INITIALIZER, padding='same') (p2)
c3 = BatchNormalization()(c3)
c3 = Dropout(DROPOUT)(c3)
c3 = Conv2D(BASE_FILTERS * 4, (3, 3), activation=ACTIVATION, kernel_initializer=INITIALIZER, padding='same') (c3)
c3 = BatchNormalization()(c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(BASE_FILTERS * 8, (3, 3), activation=ACTIVATION, kernel_initializer=INITIALIZER, padding='same') (p3)
c4 = BatchNormalization()(c4)
c4 = Dropout(DROPOUT)(c4)
c4 = Conv2D(BASE_FILTERS * 8, (3, 3), activation=ACTIVATION, kernel_initializer=INITIALIZER, padding='same') (c4)
c4 = BatchNormalization()(c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(BASE_FILTERS * 16, (3, 3), activation=ACTIVATION, kernel_initializer=INITIALIZER, padding='same') (p4)
c5 = BatchNormalization()(c5)
c5 = Dropout(DROPOUT)(c5)
c5 = Conv2D(BASE_FILTERS * 16, (3, 3), activation=ACTIVATION, kernel_initializer=INITIALIZER, padding='same') (c5)
c5 = BatchNormalization()(c5)

u6 = Conv2DTranspose(BASE_FILTERS * 8, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(BASE_FILTERS * 8, (3, 3), activation=ACTIVATION, kernel_initializer=INITIALIZER, padding='same') (u6)
c6 = BatchNormalization()(c6)
c6 = Dropout(DROPOUT)(c6)
c6 = Conv2D(BASE_FILTERS * 8, (3, 3), activation=ACTIVATION, kernel_initializer=INITIALIZER, padding='same') (c6)
c6 = BatchNormalization()(c6)

u7 = Conv2DTranspose(BASE_FILTERS * 4, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(BASE_FILTERS * 4, (3, 3), activation=ACTIVATION, kernel_initializer=INITIALIZER, padding='same') (u7)
c7 = BatchNormalization()(c7)
c7 = Dropout(DROPOUT)(c7)
c7 = Conv2D(BASE_FILTERS * 4, (3, 3), activation=ACTIVATION, kernel_initializer=INITIALIZER, padding='same') (c7)
c7 = BatchNormalization()(c7)

u8 = Conv2DTranspose(BASE_FILTERS * 2, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(BASE_FILTERS * 2, (3, 3), activation=ACTIVATION, kernel_initializer=INITIALIZER, padding='same') (u8)
c8 = BatchNormalization()(c8)
c8 = Dropout(DROPOUT)(c8)
c8 = Conv2D(BASE_FILTERS * 2, (3, 3), activation=ACTIVATION, kernel_initializer=INITIALIZER, padding='same') (c8)
c8 = BatchNormalization()(c8)

u9 = Conv2DTranspose(BASE_FILTERS, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(BASE_FILTERS, (3, 3), activation=ACTIVATION, kernel_initializer=INITIALIZER, padding='same') (u9)
c9 = BatchNormalization()(c9)
c9 = Dropout(DROPOUT)(c9)
c9 = Conv2D(BASE_FILTERS, (3, 3), activation=ACTIVATION, kernel_initializer=INITIALIZER, padding='same') (c9)

# in the case where predictions concerns several classes, you will have to
# adjust the ouput with the number of classes to be predicted and change the
# activation function
#
# eg. ouput_layer = Conv2D(NUMBER_OF_CLASSES, (1, 1), activation='softmax')(c9)
ouput_layer = Conv2D(1, (1, 1), activation='sigmoid')(c9)

model = Model(inputs=[input_layer], outputs=[ouput_layer])
model.summary()
