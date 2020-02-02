#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# 2020 Thomas FEL.
"""function to build Unet network to copy paste (keras & tensorflow 2.X)."""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation,\
    BatchNormalization, Dropout, MaxPooling2D,\
    Conv2D, concatenate, Conv2DTranspose


def build_encoder_block(previous_layer, filters, activation, use_batchnorm, dropout):
    c = Conv2D(filters, (3, 3), activation=activation,
               kernel_initializer='he_normal', padding='same')(previous_layer)
    if use_batchnorm:
        c = BatchNormalization()(c)
    if dropout:
        c = Dropout(dropout)(c)
    c = Conv2D(filters, (3, 3), activation=activation,
               kernel_initializer='he_normal', padding='same')(c)
    if use_batchnorm:
        c = BatchNormalization()(c)
    p = MaxPooling2D((2, 2))(c)

    return c, p


def build_decoder_block(previous_layer, skip_layer, is_last, filters, activation, use_batchnorm, dropout):
    u = Conv2DTranspose(filters, (2, 2), strides=(2, 2),
                        padding='same')(previous_layer)
    u = concatenate([u, skip_layer])
    c = Conv2D(filters, (3, 3), activation=activation,
               kernel_initializer='he_normal', padding='same')(u)
    if use_batchnorm:
        c = BatchNormalization()(c)
    if dropout:
        c = Dropout(dropout)(c)
    c = Conv2D(filters, (3, 3), activation=activation,
               kernel_initializer='he_normal', padding='same')(c)
    if use_batchnorm and not is_last:
        c = BatchNormalization()(c)

    return c


def build_bottleneck(previous_layer, filters, activation, use_batchnorm, dropout):
    c = Conv2D(filters, (3, 3), activation=activation,
               kernel_initializer='he_normal', padding='same')(previous_layer)
    if use_batchnorm:
        c = BatchNormalization()(c)
    if dropout:
        c = Dropout(dropout)(c)
    c = Conv2D(filters, (3, 3), activation=activation,
               kernel_initializer='he_normal', padding='same')(c)
    if use_batchnorm:
        c = BatchNormalization()(c)

    return c


def build_Unet(
    input_shape,
    output_channels,
    base_filters  = 32,
    depth         = 4,
    dropout       = 0.4,
    activation    = 'relu',
    use_batchnorm = True,
):
    """
    From the given parameters, builds a modified Unet type architecture.

    Parameters:
        input_shape (tuple):
            Shape of the input images (channel last), e.g (128, 128, 3)
        output_channels (int):
            Number of ouputs (1 if binary segmentation)
        base_filters (int):
            Number of filter for the layer 0, the number of filters increase
            exponentially such nb_filters = base_filter * 2 ** (layer_depth)
        depth (int):
            Number of layer in the encoding / decoding part
        dropout (float, bool):
            If not False, dropout rate to be applied
        activation (string):
            Activation function to use beetwen each layers
        use_batchnorm (bool):
            If True, apply batch normalization after each conv layer

    Returns:
        model (keras.model)
    """
    encoder_blocks = []
    decoder_blocks = []

    input_layer = Input(input_shape)

    # encoder
    for encoding_index in range(depth):
        previous_layer = input_layer if encoding_index == 0\
            else encoder_blocks[-1][1]

        encoder_block = build_encoder_block(
            previous_layer,
            base_filters * 2 ** encoding_index,
            activation,
            use_batchnorm,
            dropout
        )

        encoder_blocks.append(encoder_block)

    # bottleneck
    bottleneck = build_bottleneck(
        encoder_blocks[-1][1],
        base_filters * 2 ** depth,
        activation,
        use_batchnorm,
        dropout
    )

    # decoder
    for decoding_index in range(depth):
        previous_layer = bottleneck if decoding_index == 0\
            else decoder_blocks[-1]
        skip_layer = encoder_blocks[- decoding_index - 1][0]

        decoder_block = build_decoder_block(
            previous_layer,
            skip_layer,
            decoding_index == depth - 1,
            base_filters * 2 ** (depth - 1 - decoding_index),
            activation,
            use_batchnorm,
            dropout
        )

        decoder_blocks.append(decoder_block)

    output_function = 'softmax' if output_channels > 1 else 'sigmoid'
    ouput_layer     = Conv2D(output_channels, (1, 1),
                             activation=output_function)(decoder_blocks[-1])

    model = Model(inputs=[input_layer], outputs=[ouput_layer])

    return model