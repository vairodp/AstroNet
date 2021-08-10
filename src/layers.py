import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.layers import BatchNormalization, Conv2D, Input, ZeroPadding2D, LeakyReLU, UpSampling2D, Layer, Add, MaxPool2D, Concatenate, Activation, Reshape

from configs.train_config import NUM_CLASSES

class ActivationFunction():
    def __init__(self, name):
        self.name = name
    
    def get_funct(self):
        if self.name == 'mish':
            return Mish()
        elif self.name == 'leaky':
            return LeakyReLU(0.1)
        elif self.name == 'sigmoid':
            return Activation(tf.nn.sigmoid)
        else:
            return Activation('linear', dtype=tf.float32)
            
class CNNBlock(tf.keras.Model):
    def __init__(self, num_filters, kernel_size, bn_act=True, padding='same', activation='leaky', **kwargs):
        super().__init__()
        self.padding = padding
        if self.padding != 'same':
            self.conv = Conv2D(filters=num_filters, kernel_size=kernel_size, use_bias=not bn_act, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
            kernel_initializer=tf.initializers.GlorotNormal(), **kwargs)
        else:
            self.conv = Conv2D(filters=num_filters, kernel_size=kernel_size, use_bias=not bn_act, padding=self.padding, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
            kernel_initializer=tf.initializers.GlorotNormal(),  **kwargs)
        self.bn = BatchNormalization()
        self.activation = ActivationFunction(activation).get_funct()
        self.use_bn_act = bn_act

        self.zero_padding = ZeroPadding2D(padding=(1,1))
        self._build_graph(kwargs['input_shape'])
    
    def _build_graph(self, input_shape): 
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        out = self.call(inputs)
        self.out_shape = out.shape

    def call(self, x, training=False):
        if self.padding != 'same':
            x = self.zero_padding(x, training=training)
        x = self.conv(x, training=training)

        if self.use_bn_act:
            x = self.bn(x, training=training)
            x = self.activation(x, training=training)
        return x

class Mish(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, x):
        return x * backend.tanh(backend.softplus(x))

class ResidualBlock(tf.keras.Model):
    def __init__(self, num_filters, use_residual=True, repeats=1, **kwargs):
        super().__init__()
        self.layer_list = []
        for _ in range(repeats):
            seq = tf.keras.Sequential()
            if len(self.layer_list) == 0:
                input_shape = kwargs['input_shape']
            else:
                input_shape = self.layer_list[-1].layers[-1].out_shape
            conv_1 = CNNBlock(num_filters=num_filters, kernel_size=1, activation='mish', input_shape=input_shape)
            seq.add(conv_1)
            input_shape = conv_1.out_shape
            seq.add(CNNBlock(num_filters=num_filters, kernel_size=3, activation='mish', padding = 'same', input_shape=input_shape))
            self.layer_list.append(seq)
        self.use_residual = use_residual
        self.repeats = repeats
    
        self._build_graph(kwargs['input_shape'])
    
    def _build_graph(self, input_shape): 
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        out = self.call(inputs)
        self.out_shape = out.shape

    def call(self, x, training=False):
        for layer in self.layer_list:
            if self.use_residual:
                x = Add()([x,layer(x, training=training)])
            else:
                x = layer(x, training=training)
        return x

class CSPBlock(tf.keras.Model):
    def __init__(self, num_filters, num_residual_blocks=1, **kwargs):
        super().__init__()
        self.downsampling = CNNBlock(num_filters=num_filters, padding='valid',
                             activation='mish', kernel_size=3, strides=2, input_shape=kwargs['input_shape'])
        input_shape = self.downsampling.out_shape
        self.part_1 = CNNBlock(num_filters=num_filters//2, padding='same',
                             activation='mish', kernel_size=1, strides=1, input_shape=input_shape)

        self.part_2 = tf.keras.Sequential()
        conv_1 = CNNBlock(num_filters=num_filters//2, padding='same',
                             activation='mish', kernel_size=1, strides=1, input_shape=input_shape)
        self.part_2.add(conv_1)
        input_shape = conv_1.out_shape
        res_block = ResidualBlock(num_filters=num_filters//2, repeats=num_residual_blocks, input_shape=input_shape)
        input_shape = res_block.out_shape
        self.part_2.add(res_block)
        self.part_2.add(CNNBlock(num_filters=num_filters//2, padding='same',
                             activation='mish', kernel_size=1, strides=1, input_shape=input_shape))
        self._build_graph(kwargs['input_shape'])
    
    def _build_graph(self, input_shape): 
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        out = self.call(inputs)
        self.out_shape = out.shape
    
    def call(self, x, training=False):
        x = self.downsampling(x, training=training)
        part_1 = self.part_1(x, training=training)
        part_2 = self.part_2(x, training=training)

        x = Concatenate()([part_1, part_2])

        return x
    
class SPPBlock(tf.keras.Model):
    def __init__(self, input_shape):
        super().__init__()
        self.conv= CNNBlock(num_filters=512, kernel_size=1,strides=1,
                        padding='same', activation = 'leaky', input_shape=input_shape)
        self.maxpool_1 = MaxPool2D((5, 5), strides=1, padding="same", input_shape=self.conv.out_shape)
        self.maxpool_2 = MaxPool2D((9, 9), strides=1, padding="same", input_shape=self.conv.out_shape)
        self.maxpool_3 = MaxPool2D((13, 13), strides=1, padding="same", input_shape=self.conv.out_shape)
    
        self._build_graph(input_shape)
    
    def _build_graph(self, input_shape): 
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        out = self.call(inputs)
        self.out_shape = out.shape
    
    def call(self, x, training=False):
        conv = self.conv(x, training=training)
        max_poolings = [
            self.maxpool_1(conv, training=training),
            self.maxpool_2(conv, training=training),
            self.maxpool_3(conv, training=training)
        ]
        x = tf.concat([conv, *max_poolings], -1)
        return x

class UpSampling(tf.keras.Model):
    def __init__(self, num_filters, input_shape, size = 2, **kwargs):
        super().__init__(**kwargs)
        conv_input_shape = (None, input_shape[1] * size, input_shape[2]*size, input_shape[3])
        self.up_sampling = tf.keras.Sequential([
            UpSampling2D(size=size, input_shape=input_shape[1:]),
            CNNBlock(num_filters = num_filters, kernel_size=1, input_shape=conv_input_shape)
        ])
    
        self._build_graph(input_shape)
    
    def _build_graph(self, input_shape): 
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        out = self.call(inputs)
        self.out_shape = out.shape
    
    def call(self, x, training=False):
        return self.up_sampling(x, training=training)

class SpatialAttention(tf.keras.Model):
    def __init__(self, input_shape, **kwargs):
        super().__init__(**kwargs)
        self.spatial_conv = CNNBlock(num_filters=1, kernel_size=7, activation='sigmoid', input_shape=input_shape)
    
        self._build_graph(input_shape)
    
    def _build_graph(self, input_shape): 
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        out = self.call(inputs)
        self.out_shape = out.shape
    
    def call(self, x, training=False):
        y = self.spatial_conv(x, training=training)
        return tf.multiply(x, y)

class ScalePrediction(tf.keras.Model):
    def __init__(self, num_filters, input_shape, **kwargs):
        super().__init__(**kwargs)
        conv_1 = CNNBlock(num_filters = num_filters, kernel_size=3, input_shape=input_shape)
        conv_2 = CNNBlock(num_filters=(NUM_CLASSES + 5) * 3, 
                        bn_act=False, 
                        kernel_size=1,
                        activation='linear',
                        input_shape=conv_1.out_shape)
        self.pred = tf.keras.Sequential([conv_1, conv_2])
        self.reshape = Reshape((conv_2.out_shape[1], conv_2.out_shape[2], 3, NUM_CLASSES + 5), input_shape = conv_2.out_shape[1:])
        self._build_graph(input_shape)
    
    def _build_graph(self, input_shape): 
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        out = self.call(inputs)
        self.out_shape = out.shape

    def call(self, x, training=False):
        x = self.pred(x, training=training)
        x = self.reshape(x, training=training)
        return x