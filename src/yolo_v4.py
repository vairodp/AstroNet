import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.layers import BatchNormalization, Conv2D, Input, ZeroPadding2D, LeakyReLU, UpSampling2D, Layer, Add, MaxPool2D

from configs.yolo_v4 import cspdarknet53, neck, head

#TODO understand why alpha = 0.1
class CNNBlock(Layer):
    def __init__(self, num_filters, kernel_size, bn_act=True, padding='same', activation='leaky', **kwargs):
        super().__init__()
        self.padding = padding
        self.conv = Conv2D(filters=num_filters, kernel_size=kernel_size, use_bias=not bn_act, **kwargs)
        self.bn = BatchNormalization()
        self.leaky = LeakyReLU(0.1)
        self.use_bn_act = bn_act

        self.zero_padding = ZeroPadding2D(padding=(1,1))
        self.mish = Mish()
        self.activation = activation

    def call(self, input_tensor):
        if self.activation == 'leaky':
            if self.padding == 'same':
                if self.use_bn_act:
                    return self.leaky(self.bn(self.conv(input_tensor)))
                else:
                    return self.conv(input_tensor)
            else:
                if self.use_bn_act:
                    z = self.zero_padding(input_tensor)
                    return self.leaky(self.bn(self.conv(z)))
                else:
                    z = self.zero_padding(input_tensor)
                    return self.conv(z) 
        elif self.activation == 'mish':
            if self.padding == 'same':
                if self.use_bn_act:
                    return self.mish(self.bn(self.conv(input_tensor)))
                else:
                    return self.conv(input_tensor)
            else:
                if self.use_bn_act:
                    z = self.zero_padding(input_tensor)
                    return self.mish(self.bn(self.conv(z)))
                else:
                    z = self.zero_padding(input_tensor)
                    return self.conv(z) 

class Mish(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, x):
        return x * backend.tanh(backend.softplus(x))

class ResidualBlock(Layer):
    def __init__(self, num_filters, use_residual=True, repeats=1):
        super().__init__()
        self.layers = []
        for _ in range(repeats):
            self.layers.append(
                tf.keras.Sequential(
                    CNNBlock(num_filters=num_filters, kernel_size=1, activation='mish'),
                    CNNBlock(num_filters=num_filters, kernel_size=3, activation='mish', padding = 'same'),
                    ))
        self.use_residual = use_residual
        self.repeats = repeats

    def call(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = Add()([x,layer(x)])
            else:
                x = layer(x)
        return x

class CSPBlock(Layer):
    def __init__(self, num_filters, num_residual_blocks=1):
        super().__init__()
        self.part_1 = CNNBlock(out_channels=num_filters, padding='valid',
                             activation='mish', kernel_size=3, strides=2)
        self.part_2 = tf.keras.Sequential(
            CNNBlock(num_filters=num_filters//2, padding='same',
                             activation='mish', kernel_size=1, strides=1),
            ResidualBlock(num_filters=num_filters//2, num_repeats=num_residual_blocks),
            CNNBlock(num_filters=num_filters//2, padding='same',
                             activation='mish', kernel_size=1, strides=1)
            
        )
    
    def call(self, x):
        part_1 = self.part_1(x)
        part_2 = self.part_2(x)

        x = tf.concat([part_1, part_2], -1)

        return x
    
class SPPBlock(Layer):
    def __init__(self):
        super().__init__()
        self.conv= CNNBlock(num_filters=512, kernel_size=1,strides=1,
                        padding='same', activation = 'leaky')
        self.maxpool_1 = MaxPool2D((5, 5), strides=1, padding="same")
        self.maxpool_2 = MaxPool2D((9, 9), strides=1, padding="same")
        self.maxpool_3 = MaxPool2D((13, 13), strides=1, padding="same")
    
    def call(self, x):
        conv = self.conv(x)
        max_poolings = [
            self.maxpool_1(conv),
            self.maxpool_2(conv),
            self.maxpool_3(conv)
        ]
        x = tf.concat([conv, *max_poolings], -1)
        return x


class ScalePrediction():
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = tf.keras.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(
                2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1
            ),
        )
        self.num_classes = num_classes

    def call(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )

class YoloV4(tf.keras.Model):
    def __init__(self, num_classes, shape=(128, 128, 3), backbone=cspdarknet53, neck=neck, head=head):
        super().__init__()
        self.num_classes = num_classes
        self.img_shape = shape
        self.backbone = self._get_backbone(backbone)
    
    def _get_backbone(self, config):
        in_filters = self.img_shape[2]
        layers = []
        for module in config:
            if isinstance(module, list):
                repeats = module[-1]
                layers.append(CSPBlock(num_filters=in_filters*2, num_residual_blocks=repeats))
            elif isinstance(module, tuple):
                out_filters, kernel_size, strides, padding, activation = module
                layers.append(
                    CNNBlock(num_filters=out_filters, kernel_size=kernel_size,
                    padding=padding, activation=activation, strides=strides)
                )
                in_filters = out_filters
        return layers
    
    def call(self, x):
        outputs_backbone = []
        for i, layer in enumerate(self.backbone):
            if i in [6, 8, 10]:
                outputs_backbone.append(layer(x))
            x = layer(x)
        return x
