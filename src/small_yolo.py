import tensorflow as tf
from layers import CNNBlock, CSPBlock
from yolo_v4 import Head

from configs.small_yolo import backbone, head
from metrics.mean_ap import mAP

mAP_tracker = mAP(overlap_threshold=0.5, model='small_yolo', name='mAP_0.5')

class SmallYolo(tf.keras.Model):
    def __init__(self, num_classes, shape=(128, 128, 3), backbone=backbone, head=head):
        super().__init__()
        self.num_classes = num_classes
        self.img_shape = shape
        self.backbone = self._get_backbone(backbone)
        input_shapes = [self.backbone[i].out_shape for i in [2,4,6]]
        self.head = Head(head, input_shapes=input_shapes[::-1])
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
        mAP_tracker.update_state(y_pred, data['bbox'], data['num_of_bbox'])
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
        mAP_tracker.update_state(y_pred, data['bbox'], data['num_of_bbox'])
        
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
    
    def call(self, x, training=False):
        outputs_backbone = []
        for i, layer in enumerate(self.backbone):
            if i in [2, 4, 6]:
                outputs_backbone.append(layer(x, training=training))
            x = layer(x, training=training)
        x = self.head(outputs_backbone[::-1])
        return x
    
    def predict_step(self, data):
        data = data['image']
        out = super().predict_step(data)
        # TODO: pass out to the non_maximum suppression util and return both those outputs and the standard outputs
        # new_outs = non_max_suppression(out)
        return out # , new_outs
    