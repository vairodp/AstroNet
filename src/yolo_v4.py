import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.layers import Concatenate
from layers import CNNBlock, CSPBlock, SpatialAttention, ScalePrediction, SPPBlock, UpSampling

from configs.yolo_v4 import cspdarknet53, panet, head
from configs.train_config import NUM_CLASSES
from metrics.mean_ap import mAP

#TODO understand why alpha = 0.1
#TODO understand if the activation function can be apllied before bn or not
#TODO implement DropBlock layer, maybe?
#TODO understand why "multiply" in spatial attention
        
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
                

    def call(self, x, training=False):
        out_large, out_medium, out_small = x
        
        for layer in self.layer_list['S']:
            out_small = layer(out_small, training=training)
        small_upsampled = self.upsamplings['S'](out_small, training=training)
        out_medium = self.concats['M'](out_medium, training=training)
        out_medium = self.concat([out_medium, small_upsampled])
        
        for layer in self.layer_list['M']:
            out_medium = layer(out_medium, training=training)
        medium_upsampled = self.upsamplings['M'](out_medium, training=training)
        out_large = self.concats['L'](out_large, training=training)
        out_large = self.concat([out_large, medium_upsampled])
        
        for layer in self.layer_list['L']:
            out_large = layer(out_large, training=training)
        
        out_small = self.attentions[0](out_small, training=training)
        out_medium = self.attentions[1](out_medium, training=training)
        out_large = self.attentions[2](out_large, training=training)

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

    def call(self, x, training=False):
        output_small, output_medium, output_large = x

        shortcut_large = output_large

        output_large = self.layer_list['L'][0](output_large, training=training)

        large_downsampled = self.layer_list['L'][1](shortcut_large, training=training)
        output_medium = self.concat([large_downsampled, output_medium])

        for layer in self.layer_list['M'][:-2]:
            output_medium = layer(output_medium, training=training)
            

        shortcut_medium = output_medium
        output_medium = self.layer_list['M'][-2](output_medium, training=training)

        medium_downsampled = self.layer_list['M'][-1](shortcut_medium, training=training)
        output_small = self.concat([medium_downsampled, output_small])

        for layer in self.layer_list['S']:
            output_small = layer(output_small, training=training)
        
        return output_small, output_medium, output_large

mAP_tracker = mAP(overlap_threshold=0.5, name='mAP_0.5')
#mAP_tracker = DetectionMAP(n_class=NUM_CLASSES)

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
    
    @property
    def metrics(self):
        return self.compiled_loss.metrics + self.compiled_metrics.metrics + [mAP_tracker]
    
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

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        #mAP_tracker.update_state(y_pred, data['bbox'], data['num_of_bbox'])
        mAP_tracker.update_state(y_pred, data['bbox'], data['num_of_bbox'])
        # Return a dict mapping metric names to current value 
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        # Unpack the data
        x, y = data['image'], list(data['label'])
        # Compute predictions
        y_pred = self(x, training=True)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        #mAP_tracker.update_state(y_pred, data['bbox'], data['num_of_bbox'])
        mAP_tracker.update_state(y_pred, data['bbox'], data['num_of_bbox'])
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
        #return {m.name: m.result() for m in self.metrics}
    
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
    
    def call(self, x, training=False):
        outputs_backbone = []
        for i, layer in enumerate(self.backbone):
            if i in [6, 8, 10]:
                outputs_backbone.append(layer(x, training=training))
            x = layer(x)
        x = self.neck(outputs_backbone, training=training)
        x = self.head(x, training=training)
        return x
    
    def predict_step(self, data):
        data = data['image']
        out = super().predict_step(data)
        # TODO: pass out to the non_maximum suppression util and return both those outputs and the standard outputs
        # new_outs = non_max_suppression(out)
        return out # , new_outs
    