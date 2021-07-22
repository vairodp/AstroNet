import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.layers import BatchNormalization, Conv2D, Input, ZeroPadding2D, LeakyReLU, UpSampling2D, Layer, Add, MaxPool2D, Concatenate, Activation

from configs.yolo_v4 import cspdarknet53, panet, head, NUM_CLASSES

#TODO understand why alpha = 0.1
#TODO understand if the activation function can be apllied before bn or not
#TODO implement DropBlock layer, maybe?
#TODO understand why "multiply" in spatial attention


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
            
class CNNBlock(Layer):
    def __init__(self, num_filters, kernel_size, bn_act=True, padding='same', activation='leaky', **kwargs):
        super().__init__()
        self.padding = padding
        if self.padding != 'same':
            self.conv = Conv2D(filters=num_filters, kernel_size=kernel_size, use_bias=not bn_act, **kwargs)
        else:
            self.conv = Conv2D(filters=num_filters, kernel_size=kernel_size, use_bias=not bn_act, padding=self.padding, **kwargs)
        self.bn = BatchNormalization()
        self.activation = ActivationFunction(activation).get_funct()
        self.use_bn_act = bn_act

        self.zero_padding = ZeroPadding2D(padding=(1,1))

    def call(self, x):
        if self.padding != 'same':
            x = self.zero_padding(x)
        x = self.conv(x)

        if self.use_bn_act:
            x = self.bn(x)
            x = self.activation(x)
        return x

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
                tf.keras.Sequential([
                    CNNBlock(num_filters=num_filters, kernel_size=1, activation='mish'),
                    CNNBlock(num_filters=num_filters, kernel_size=3, activation='mish', padding = 'same')]
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
        self.downsampling = CNNBlock(num_filters=num_filters, padding='valid',
                             activation='mish', kernel_size=3, strides=2)
        self.part_1 = CNNBlock(num_filters=num_filters//2, padding='same',
                             activation='mish', kernel_size=1, strides=1)

        self.part_2 = tf.keras.Sequential([
            CNNBlock(num_filters=num_filters//2, padding='same',
                             activation='mish', kernel_size=1, strides=1),
            ResidualBlock(num_filters=num_filters//2, repeats=num_residual_blocks),
            CNNBlock(num_filters=num_filters//2, padding='same',
                             activation='mish', kernel_size=1, strides=1)]            
        )
    
    def call(self, x):
        x = self.downsampling(x)
        part_1 = self.part_1(x)
        part_2 = self.part_2(x)

        x = Concatenate()([part_1, part_2])

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

class UpSampling(Layer):
    def __init__(self, num_filters, size = 2, **kwargs):
        super().__init__(**kwargs)
        self.up_sampling = tf.keras.Sequential([
            UpSampling2D(size=size),
            CNNBlock(num_filters = num_filters, kernel_size=1)
        ])
    
    def call (self, x):
        return self.up_sampling(x)

class SpatialAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.spatial_conv = CNNBlock(num_filters=1, kernel_size=7, activation='sigmoid')
    
    def call(self, x):
        y = self.spatial_conv(x)
        return tf.multiply(x, y)




class ScalePrediction(Layer):
    def __init__(self, num_filters, **kwargs):
        super().__init__(**kwargs)
        self.pred = tf.keras.Sequential([
            CNNBlock(num_filters = num_filters, kernel_size=3),
            CNNBlock(num_filters=(NUM_CLASSES + 5) * 3, 
                    bn_act=False, 
                    kernel_size=1,
                    activation='linear'
                )]
        )

    def call(self, x):
        x = self.pred(x)
        x = tf.reshape(x, 
            [-1, tf.shape(x)[1], tf.shape(x)[2], 3, NUM_CLASSES + 5])
        return x
        

class Neck(Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.concat = Concatenate()
        self.attentions = []
        self.layers = {
            'S': [],
            'M': [],
            'L': []
        }
        self.upsamplings = {
            'S': None,
            'M': None
        }
        self.concats = {
            'M': None,
            'L': None
        }
        self._get_neck(config)
    
    def _get_neck(self, config):
        size = 'S'
        for module in config:
            if isinstance(module, str):
                size = module
            elif isinstance(module, tuple):
                num_filters, kernel_size, strides, padding, activation = module
                self.layers[size].append(
                    CNNBlock(num_filters=num_filters, kernel_size=kernel_size,
                    padding=padding, activation=activation, strides=strides)
                )
            elif isinstance(module, list):
                if module[0] == 'SPP':
                    self.layers[size].append(SPPBlock())
                elif module[0] == 'U':
                    self.upsamplings[size] = UpSampling(num_filters=module[1])
                elif module[0] == 'A':
                    for _ in range(module[1]):
                        self.attentions.append(SpatialAttention())
                else:
                    _, num_filters, kernel_size, strides, padding, activation = module
                    self.concats[size] = CNNBlock(num_filters=num_filters, 
                                                kernel_size=kernel_size,
                                                padding=padding, 
                                                activation=activation, 
                                                strides=strides)

    def call(self, x):
        out_large, out_medium, out_small = x
        
        for layer in self.layers['S']:
            out_small = layer(out_small)
        small_upsampled = self.upsamplings['S'](out_small)
        out_medium = self.concats['M'](out_medium)
        out_medium = self.concat([out_medium, small_upsampled])
        
        for layer in self.layers['M']:
            out_medium = layer(out_medium)
        medium_upsampled = self.upsamplings['M'](out_medium)
        out_large = self.concats['L'](out_large)
        out_large = self.concat([out_large, medium_upsampled])
        
        for layer in self.layers['L']:
            out_large = layer(out_large)
        
        out_small = self.attentions[0](out_small)
        out_medium = self.attentions[1](out_medium)
        out_large = self.attentions[2](out_large)

        return out_small, out_medium, out_large


class Head(Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.concat = Concatenate()
        self.layers = {
            'S': [],
            'M': [],
            'L': []
        }
        self.outputs = []
        self._get_head(config)
    
    def _get_head(self, config):
        size = 'L'
        for module in config:
            if isinstance(module, str):
                size = module
            elif isinstance(module, tuple):
                num_filters, kernel_size, strides, padding, activation = module
                self.layers[size].append(
                    CNNBlock(num_filters=num_filters, kernel_size=kernel_size,
                    padding=padding, activation=activation, strides=strides)
                )
            elif isinstance(module, list):
                if module[0] == 'S':
                    self.layers[size].append(ScalePrediction(num_filters=module[1]))
    
    def call(self, x):
        output_small, output_medium, output_large = x

        shortcut_large = output_large

        output_large = self.layers['L'][0](output_large)

        large_downsampled = self.layers['L'][1](shortcut_large)
        output_medium = self.concat([large_downsampled, output_medium])

        for layer in self.layers['M'][:-2]:
            output_medium = layer(output_medium)
        
        shortcut_medium = output_medium
        output_medium = self.layers['M'][-2](output_medium)

        medium_downsampled = self.layers['M'][-1](shortcut_medium)
        output_small = self.concat([medium_downsampled, output_small])

        for layer in self.layers['S']:
            output_small = layer(output_small)
        
        return output_small, output_medium, output_large

class YoloV4(tf.keras.Model):
    def __init__(self, num_classes, shape=(128, 128, 3), backbone=cspdarknet53, neck=panet, head=head):
        super().__init__()
        self.num_classes = num_classes
        self.img_shape = shape
        self.backbone = self._get_backbone(backbone)
        self.neck = Neck(neck)
        self.head = Head(head)
    
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data['image'], list(data['label'])

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            #loss = tf.reduce_sum(loss)
            #regularization_loss = tf.reduce_sum(self.losses)
            #total_loss = loss + regularization_loss

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        # Unpack the data
        x, y = data['image'], list(data['label'])

        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
    
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
        x = self.neck(outputs_backbone)
        x = self.head(x)
        return x
    