import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.layers import BatchNormalization, Conv2D, Input, ZeroPadding2D, LeakyReLU, UpSampling2D, Layer, Add, MaxPool2D, Concatenate, Activation, Reshape
from tensorflow.python.ops.gen_array_ops import shape

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
            
class CNNBlock(tf.keras.Model):
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
        self._build_graph(kwargs['input_shape'])
    
    def _build_graph(self, input_shape): 
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        out = self.call(inputs)
        self.out_shape = out.shape

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

    def call(self, x):
        for layer in self.layer_list:
            if self.use_residual:
                x = Add()([x,layer(x)])
            else:
                x = layer(x)
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
    
    def call(self, x):
        x = self.downsampling(x)
        part_1 = self.part_1(x)
        part_2 = self.part_2(x)

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
    
    def call(self, x):
        conv = self.conv(x)
        max_poolings = [
            self.maxpool_1(conv),
            self.maxpool_2(conv),
            self.maxpool_3(conv)
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
    
    def call (self, x):
        return self.up_sampling(x)

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
    
    def call(self, x):
        y = self.spatial_conv(x)
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

    def call(self, x):
        x = self.pred(x)
        x = self.reshape(x)
        return x
        
class Neck(tf.keras.Model):
    def __init__(self, config, input_shapes, **kwargs):
        super().__init__(**kwargs)
        self.concat = Concatenate()
        self.attentions = []
        self.in_shapes = {
            'S': input_shapes[2],
            'M': input_shapes[1],
            'L': input_shapes[0]
        }
        self.layer_list = {
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
        self._build_graph(input_shapes)
    
    def _build_graph(self, input_shape): 
        input_shape_nobatch = [in_shape[1:] for in_shape in input_shape]
        self.build(input_shape)
        inputs = [tf.keras.Input(shape=input_shape_nobatch[i]) for i in range(3)]
        out = self.call(inputs)
        self.out_shape = [out[i].shape for i in range(3)]
    
    def _get_neck(self, config):
        size = 'S'
        for module in config:
            if isinstance(module, str):
                size = module
            elif isinstance(module, tuple):
                num_filters, kernel_size, strides, padding, activation = module
                conv = CNNBlock(num_filters=num_filters, kernel_size=kernel_size,
                    padding=padding, activation=activation, strides=strides, input_shape=self.in_shapes[size])
                self.layer_list[size].append(conv)
                self.in_shapes[size] = conv.out_shape
            elif isinstance(module, list):
                if module[0] == 'SPP':
                    spp_block = SPPBlock(input_shape = self.in_shapes[size])
                    self.layer_list[size].append(spp_block)
                    self.in_shapes[size] = spp_block.out_shape
                elif module[0] == 'U':
                    upsampling = UpSampling(num_filters=module[1], input_shape=self.in_shapes[size])
                    self.upsamplings[size] = upsampling
                elif module[0] == 'A':
                    for i in range(module[1]):
                        self.attentions.append(SpatialAttention(input_shape=list(self.in_shapes.values())[i]))
                else:
                    _, num_filters, kernel_size, strides, padding, activation = module
                    concat = CNNBlock(num_filters=num_filters, 
                                                kernel_size=kernel_size,
                                                padding=padding, 
                                                activation=activation, 
                                                strides=strides, input_shape=self.in_shapes[size])
                    self.concats[size] = concat
                

    def call(self, x):
        out_large, out_medium, out_small = x
        
        for layer in self.layer_list['S']:
            out_small = layer(out_small)
        small_upsampled = self.upsamplings['S'](out_small)
        out_medium = self.concats['M'](out_medium)
        out_medium = self.concat([out_medium, small_upsampled])
        
        for layer in self.layer_list['M']:
            out_medium = layer(out_medium)
        medium_upsampled = self.upsamplings['M'](out_medium)
        out_large = self.concats['L'](out_large)
        out_large = self.concat([out_large, medium_upsampled])
        
        for layer in self.layer_list['L']:
            out_large = layer(out_large)
        
        out_small = self.attentions[0](out_small)
        out_medium = self.attentions[1](out_medium)
        out_large = self.attentions[2](out_large)

        return out_small, out_medium, out_large


class Head(tf.keras.Model):
    def __init__(self, config, input_shapes, **kwargs):
        super().__init__(**kwargs)
        self.in_shapes = {
            'S': (*input_shapes[0][:-1], input_shapes[0][-1]*2),
            'M': (*input_shapes[1][:-1], input_shapes[1][-1]*2),
            'L': input_shapes[2]
        }
        self.concat = Concatenate()
        self.layer_list = {
            'S': [],
            'M': [],
            'L': []
        }
        self._get_head(config)
        self._build_graph(input_shapes)
    
    def _build_graph(self, input_shape):
        input_shape_nobatch = [in_shape[1:] for in_shape in input_shape]
        self.build(input_shape)
        inputs = [tf.keras.Input(shape=input_shape_nobatch[i]) for i in range(3)]
        out = self.call(inputs)
        self.out_shape = [out[i].shape for i in range(3)]
    
    def _get_head(self, config):
        size = 'L'

        for module in config:
            if isinstance(module, str):
                size = module
            elif isinstance(module, tuple):
                num_filters, kernel_size, strides, padding, activation = module
                conv = CNNBlock(num_filters=num_filters, kernel_size=kernel_size,
                    padding=padding, activation=activation, strides=strides, input_shape=self.in_shapes[size])
                self.layer_list[size].append(conv)
                self.in_shapes[size] = conv.out_shape
            elif isinstance(module, list):
                if module[0] == 'S':
                    self.layer_list[size].append(ScalePrediction(num_filters=module[1], input_shape=self.in_shapes[size]))

    def call(self, x):
        output_small, output_medium, output_large = x

        shortcut_large = output_large

        output_large = self.layer_list['L'][0](output_large)

        large_downsampled = self.layer_list['L'][1](shortcut_large)
        output_medium = self.concat([large_downsampled, output_medium])

        for layer in self.layer_list['M'][:-2]:
            output_medium = layer(output_medium)
            

        shortcut_medium = output_medium
        output_medium = self.layer_list['M'][-2](output_medium)

        medium_downsampled = self.layer_list['M'][-1](shortcut_medium)
        output_small = self.concat([medium_downsampled, output_small])

        for layer in self.layer_list['S']:
            output_small = layer(output_small)
        
        return output_small, output_medium, output_large

class YoloV4(tf.keras.Model):
    def __init__(self, num_classes, shape=(128, 128, 3), backbone=cspdarknet53, neck=panet, head=head):
        super().__init__()
        self.num_classes = num_classes
        self.img_shape = shape
        self.backbone = self._get_backbone(backbone)
        input_shapes = [self.backbone[i].out_shape for i in [6,8,10]]
        self.neck = Neck(neck, input_shapes=input_shapes)
        self.head = Head(head, input_shapes=self.neck.out_shape)
        self._build_graph(input_shape=(None, *self.img_shape))
    
    def _build_graph(self, input_shape): 
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        out = self.call(inputs)
        self.out_shape = [out[i].shape for i in range(3)]
    
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
            if len(layers) == 0:
                input_shape = (None, *self.img_shape)
            else:
                input_shape = layers[-1].out_shape
            if isinstance(module, list):
                repeats = module[-1]
                layers.append(CSPBlock(num_filters=in_filters*2, num_residual_blocks=repeats, input_shape=input_shape))
            elif isinstance(module, tuple):
                out_filters, kernel_size, strides, padding, activation = module
                layers.append(
                    CNNBlock(num_filters=out_filters, kernel_size=kernel_size,
                    padding=padding, activation=activation, strides=strides, input_shape=input_shape)
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
    