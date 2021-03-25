from tensorflow.keras.layers import Conv2DTranspose, Reshape
from tensorflow.keras.layers import BatchNormalization, concatenate
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Model


def build_generator(inputs, labels, image_size):
    """
    crate the generator for our GAN
    :Arguments:
    inputs:Input layer of the generator.
    labels: Input layer of one-hot vector to condition inputs.
    image_size:Target size of a side (if shape is square).

    :return:
    generator model.
    """

    image_resize = image_size // 4
    kernel_size = 5
    layer_filters = [128, 64, 32, 1]

    x = concatenate([inputs, labels], axis=1)
    x = Dense(image_resize * image_resize * layer_filters[0])(x)
    x = Reshape((image_resize, image_resize, layer_filters[0]))(x)
    for filter in layer_filters:
        if layer_filters[-2] < filter:
            stride = 2
        else:
            stride = 1
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(
            filter,
            kernel_size,
            stride,
            padding='same',
        )(x)
    x = Activation('sigmoid')(x)
    outputs = x
    generator = Model([inputs, labels], outputs, name='generator')
    return generator
