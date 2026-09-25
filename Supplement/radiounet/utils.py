import tensorflow as tf
from tensorflow import keras


import tensorflow as tf
from tensorflow import keras


@tf.keras.utils.register_keras_serializable(package="RadioUNetBaseline")
class Downsample(keras.layers.Layer):
    def __init__(self, filters, size, apply_batchnorm=True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.size = size
        self.apply_batchnorm = apply_batchnorm

        initializer = tf.random_normal_initializer(0.0, 0.02)
        self.conv = keras.layers.Conv2D(
            filters,
            kernel_size=size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=not apply_batchnorm,
        )
        if apply_batchnorm:
            self.bn = keras.layers.BatchNormalization()

    def call(self, x, training=None):
        x = self.conv(x)
        if self.apply_batchnorm:
            x = self.bn(x, training=training)
        x = tf.nn.leaky_relu(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "size": self.size,
            "apply_batchnorm": self.apply_batchnorm,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="RadioUNetBaseline")
class Upsample(keras.layers.Layer):
    def __init__(self, filters, size, apply_dropout=False, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.size = size
        self.apply_dropout = apply_dropout

        initializer = tf.random_normal_initializer(0.0, 0.02)
        self.up = keras.layers.Conv2DTranspose(
            filters,
            kernel_size=size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
        self.bn = keras.layers.BatchNormalization()
        if apply_dropout:
            self.drop = keras.layers.Dropout(0.5)

    def call(self, x, skip, training=None):
        x = self.up(x)
        x = self.bn(x, training=training)
        if self.apply_dropout:
            x = self.drop(x, training=training)
        x = tf.nn.relu(x)
        x = tf.concat([x, skip], axis=-1)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "size": self.size,
            "apply_dropout": self.apply_dropout,
        })
        return config


def build_radiounet(input_shape=(256, 256, 2), base_filters=32, output_activation="sigmoid"):
    """
    Fair-comparison RadioUNet-style baseline:
    - U-Net-like encoder-decoder
    - 2-channel input aligned to the user's final experiment: [building, r_map]
    - 1-channel regression output in [0,1] if output_activation='sigmoid'
    """
    inputs = keras.Input(shape=input_shape, name="radiomap_input")

    d1 = Downsample(base_filters, 4, apply_batchnorm=False)(inputs)          # 128
    d2 = Downsample(base_filters * 2, 4, apply_batchnorm=True)(d1)           # 64
    d3 = Downsample(base_filters * 4, 4, apply_batchnorm=True)(d2)           # 32
    d4 = Downsample(base_filters * 8, 4, apply_batchnorm=True)(d3)           # 16
    d5 = Downsample(base_filters * 16, 4, apply_batchnorm=True)(d4)          # 8
    bottleneck = Downsample(base_filters * 16, 4, apply_batchnorm=True)(d5)  # 4

    u1 = Upsample(base_filters * 16, 4, apply_dropout=True)(bottleneck, d5)  # 8
    u2 = Upsample(base_filters * 8, 4, apply_dropout=True)(u1, d4)           # 16
    u3 = Upsample(base_filters * 4, 4, apply_dropout=False)(u2, d3)          # 32
    u4 = Upsample(base_filters * 2, 4, apply_dropout=False)(u3, d2)          # 64
    u5 = Upsample(base_filters, 4, apply_dropout=False)(u4, d1)              # 128

    initializer = tf.random_normal_initializer(0.0, 0.02)
    outputs = keras.layers.Conv2DTranspose(
        1,
        kernel_size=4,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        activation=output_activation,
        name="radiomap_output",
    )(u5)

    model = keras.Model(inputs=inputs, outputs=outputs, name="RadioUNetBaseline")
    return model
