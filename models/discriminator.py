from tensorflow.keras.layers import Conv2D, Flatten, Reshape
from tensorflow.keras.layers import LeakyReLU, concatenate
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Model


def build_discriminator(inputs, labels, image_size):
    """
    crate the discriminator for our GAN
    :Arguments:
    inputs:Input layer of the discriminator
    labels: Input layer of one-hot vector to condition inputs
    image_size:Target size of a side (if shape is square)

    :return:
    discriminator model
    """

    kernel_size = 5
    layer_filters = [32, 64, 128, 256]

    x = inputs
    y = Dense(image_size * image_size)(labels)
    y = Reshape((image_size, image_size, 1))(y)
    x = concatenate([x, y])
    for filter in layer_filters:
        if layer_filters[-1] == filter:
            stride = 1
        else:
            stride = 2

        x = LeakyReLU(0.2)(x)
        x = Conv2D(
            filter,
            kernel_size,
            stride,
            padding='same'
        )(x)
    x = Flatten()(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    discriminator = Model(inputs=[inputs, labels], outputs=x, name='discriminator')
    return discriminator
