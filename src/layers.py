import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.layers.advanced_activations import ReLU
from tensorflow.keras.layers import BatchNormalization, Conv2D, Input, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU, Layer, Concatenate, Reshape, Conv2DTranspose

from configs.train_config import WEIGHT_DECAY

class Mish(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, x):
        return x * backend.tanh(backend.softplus(x))

class DropBlock(tf.keras.layers.Layer):
    def __init__(self, drop_rate=0.2, block_size=3, **kwargs):
        super(DropBlock, self).__init__(**kwargs)
        self.rate = drop_rate
        self.block_size = block_size

    def call(self, inputs, training=None):
        if training:
            #batch size
            b = tf.shape(inputs)[0]
            
            random_tensor = tf.random.uniform(shape=[b, self.m_h, self.m_w, self.c]) + self.bernoulli_rate
            binary_tensor = tf.floor(random_tensor)
            binary_tensor = tf.pad(binary_tensor, [[0,0],
                                                   [self.block_size // 2, self.block_size // 2],
                                                   [self.block_size // 2, self.block_size // 2],
                                                   [0, 0]])
            binary_tensor = tf.nn.max_pool(binary_tensor,
                                           [1, self.block_size, self.block_size, 1],
                                           [1, 1, 1, 1],
                                           'SAME')
            binary_tensor = 1 - binary_tensor
            inputs = tf.math.divide(inputs, (1 - self.rate)) * binary_tensor
        return inputs
    
    def get_config(self):
        config = super(DropBlock, self).get_config()
        return config

    def build(self, input_shape):
        #feature map size (height, weight, channel)
        self.b, self.h, self.w, self.c = input_shape.as_list()
        #mask h, w
        self.m_h = self.h - (self.block_size // 2) * 2
        self.m_w = self.w - (self.block_size // 2) * 2
        self.bernoulli_rate = (self.rate * self.h * self.w) / (self.m_h * self.m_w * self.block_size**2)

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
                    kernel_regularizer=l2(WEIGHT_DECAY),
                    strides=strides, 
                    padding=padding, 
                    use_bias=False)(inputs)
    
    inputs = BatchNormalization()(inputs)
    
    if activation == 'leaky_relu':
        inputs = LeakyReLU(alpha=0.1)(inputs)
    elif activation == 'mish':
        inputs = Mish()(inputs)
    elif activation =='relu':
        inputs = ReLU()(inputs)
    
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

def scale_prediction(inputs, num_classes, num_anchors_stage=3, num=0):
    
    x = Conv2D(
        filters=num_anchors_stage * (num_classes + 5),
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=True,
        kernel_regularizer=l2(WEIGHT_DECAY),
        name='conv2d_pred_{:02}_{}'.format(num_classes, num)
    )(inputs)
    x = Reshape(
        (x.shape[1], x.shape[2], num_anchors_stage, num_classes + 5), name=f'reshape_{num_classes}_{num}')(x)
    return x

def decoder_block(inputs, skip_connection, num_filters):
    x = Conv2DTranspose(num_filters, kernel_size=2, padding='same', strides=2)(inputs)
    x = Concatenate()([x, skip_connection])
    x = cnn_block(x, num_filters=num_filters, kernel_size=3, strides=1, activation='relu')
    x = cnn_block(x, num_filters=num_filters, kernel_size=3, strides=1, activation='relu')

    return x