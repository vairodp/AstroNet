import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.layers import BatchNormalization, Conv2D, Input, ZeroPadding2D, LeakyReLU, Layer, Concatenate, Reshape

from configs.train_config import NUM_CLASSES

class Mish(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, x):
        return x * backend.tanh(backend.softplus(x))

def cnn_block(inputs, 
            num_filters, 
            kernel_size, 
            strides, 
            padding='same', 
            zero_padding=False, 
            activation='leaky'):

    if zero_padding:
        inputs = ZeroPadding2D(((1,0), (1,0)))(inputs)
    
    inputs = Conv2D(filters=num_filters, 
                    kernel_size=kernel_size, 
                    strides=strides, 
                    padding=padding, 
                    use_bias=False)(inputs)
    
    inputs = BatchNormalization()(inputs)
    
    if activation == 'leaky_relu':
        inputs = LeakyReLU(alpha=0.1)(inputs)
    elif activation == 'mish':
        inputs = Mish()(inputs)
    
    return inputs

def residual_block(inputs, num_blocks):
    
    f_, _, _, filters = inputs.shape
    x = inputs
    for _ in range(num_blocks):
        block_inputs = x
        x = cnn_block(x, filters, kernel_size=1, strides=1, activation="mish")
        x = cnn_block(x, filters, kernel_size=3, strides=1, activation="mish")

        x = x + block_inputs

    return x

def csp_block(inputs, filters, num_blocks):

    half_filters = filters // 2

    x = cnn_block(
        inputs,
        num_filters=filters,
        kernel_size=3,
        strides=2,
        zero_padding=True,
        padding="valid",
        activation="mish",
    )
    route = cnn_block(
        x, num_filters=half_filters, kernel_size=1, strides=1, activation="mish"
    )
    x = cnn_block(x, num_filters=half_filters, kernel_size=1, strides=1, activation="mish")

    x = residual_block(x, num_blocks=num_blocks)
    x = cnn_block(x, num_filters=half_filters, kernel_size=1, strides=1, activation="mish")
    x = Concatenate()([x, route])

    x = cnn_block(x, num_filters=filters, kernel_size=1, strides=1, activation="mish")

    return x

def scale_prediction(inputs, num_classes, num_anchors_stage=3):
    
    x = Conv2D(
        filters=num_anchors_stage * (num_classes + 5),
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=True,
    )(inputs)
    x = Reshape(
        (x.shape[1], x.shape[2], num_anchors_stage, num_classes + 5))(x)
    return x