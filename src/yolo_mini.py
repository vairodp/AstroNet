import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import MaxPool2D, Input, Lambda, Concatenate
from configs.train_config import NUM_CLASSES, MAX_NUM_BBOXES, loss_params, INITIAL_LR, NUM_EPOCHS, ITER_PER_EPOCH
from datasets.ska_dataset import SKADataset
from loss import yolo3_loss
from anchors import CUSTOM_ANCHORS1, CUSTOM_ANCHORS_TINY, compute_normalized_anchors, YOLOLITE_ANCHORS

from layers import cnn_block, residual_block, route_group, scale_prediction, upsample

def backbone_tiny(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    input_data = cnn_block(inputs, num_filters=16, kernel_size=3, strides=2, zero_padding=True, padding='valid')
    input_data = cnn_block(input_data, num_filters=16, kernel_size=3, strides=1)

    input_data = residual_block(input_data, num_blocks=1)

    input_data = cnn_block(input_data, num_filters=32, kernel_size=3, strides=2, zero_padding=True, padding='valid')
    input_data = cnn_block(input_data, num_filters=32, kernel_size=3, strides=1)

    input_data = residual_block(input_data, num_blocks=2)

    input_data = cnn_block(input_data, num_filters=64, kernel_size=3, strides=2, zero_padding=True, padding='valid')
    input_data = cnn_block(input_data, num_filters=64, kernel_size=3, strides=1)

    input_data = residual_block(input_data, num_blocks=1)

    return tf.keras.Model(inputs, input_data, name="Backbone-tiny")

class YOLOv4Lite(tf.keras.Model):
    def __init__(self, input_shape, num_classes,
        anchors,
        training=False,
        yolo_max_boxes=MAX_NUM_BBOXES,
        yolo_iou_threshold=loss_params['iou_threshold']):
        super().__init__(name='YOLOv4')
        self.num_classes = num_classes
        #self.input_shape = input_shape
        self.training = training
        self.anchors = anchors
        self.yolo_max_boxes = yolo_max_boxes
        self.yolo_iou_threshold = yolo_iou_threshold


        if (input_shape[0] % 32 != 0) | (input_shape[1] % 32 != 0):
            raise ValueError(
            f"Provided height and width in input_shape {input_shape} is not a multiple of 32"
        )

        backbone = backbone_tiny(input_shape)
        print(backbone.summary())
        print(backbone.outputs)

        self.normalized_anchors = compute_normalized_anchors(anchors, input_shape)

        inputs = tf.keras.Input(shape=input_shape)
        x = backbone(inputs)
        maxpool_1 = MaxPool2D((5, 5), strides=1, padding="same")(x)
        maxpool_2 = MaxPool2D((9, 9), strides=1, padding="same")(x)
        maxpool_3 = MaxPool2D((13, 13), strides=1, padding="same")(x)

        spp = Concatenate()([maxpool_3, maxpool_2, maxpool_1, x])
        x = cnn_block(spp, num_filters=64, kernel_size=3, strides=1)
        out = scale_prediction(x, num_anchors_stage=len(anchors[0]), num_classes=NUM_CLASSES)

        anchors = np.array([anchor for subl in self.normalized_anchors for anchor in subl])

        y_true = [Input(shape=(None, None, len(self.normalized_anchors[0]), NUM_CLASSES+5), name='y_true_0')]

        self.model_body = tf.keras.Model(inputs=inputs, outputs=out, name="YOLOv4Tiny")
        model_loss, location_loss, confidence_loss, class_loss = Lambda(yolo3_loss, name='yolo_loss',
            arguments={'anchors': anchors, 'num_classes': NUM_CLASSES, 'ignore_thresh': 0.5, 'label_smoothing': loss_params['smooth_factor'], 'elim_grid_sense': True})(
        [out, *y_true])

        self.model = tf.keras.Model([self.model_body.input, *y_true], model_loss)

        loss_dict = {'location_loss':location_loss, 'confidence_loss':confidence_loss, 'class_loss':class_loss}
        self.add_metrics(loss_dict)

        #weights_path = get_weights_by_keyword_or_path(weights, model=self.model)
        #if weights_path is not None:
        #    self.model.load_weights(str(weights_path), by_name=True, skip_mismatch=True)
    
    def add_metrics(self, metric_dict):
        '''
        add metric scalar tensor into model, which could be tracked in training
        log and tensorboard callback
        '''
        for (name, metric) in metric_dict.items():
            # seems add_metric() is newly added in tf.keras. So if you
            # want to customize metrics on raw keras model, just use
            # "metrics_names" and "metrics_tensors" as follow:
            #
            #model.metrics_names.append(name)
            #model.metrics_tensors.append(loss)
            self.model.add_metric(metric, name=name, aggregation='mean')
    
    def call(self, x, training=False):
        return self.model(x, training)
    
    def train_step(self, data):
        x, y = data['image'], data['label']
        print(y.shape)

        with tf.GradientTape() as tape:
            y_pred = self([x, y], training=True)  # Forward pass
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        #mAP_tracker.update_state(y_pred, data['bbox'], data['num_of_bbox'])
        metrics = {m.name: m.result() for m in self.metrics}
        metrics['reg_loss'] = (loss - y_pred)[0]
        return metrics
    
    def test_step(self, data):
        x, y = data['image'], data['label']
        
        y_pred = self([x, y], training=True)
        loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)

        #mAP_tracker.update_state(y_pred, data['bbox'], data['num_of_bbox'])
        metrics = {m.name: m.result() for m in self.metrics}
        metrics['reg_loss'] = (loss - y_pred)[0]

        return metrics

yolo = YOLOv4Lite(input_shape=(128,128,1), num_classes=NUM_CLASSES, anchors=CUSTOM_ANCHORS_TINY, training=True)
yolo.model.summary(line_length=200)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipvalue=1.0)

yolo.compile(optimizer=optimizer, loss={'output_1': lambda y_true, y_pred: y_pred})
anchor_masks = [np.array([0,1,2,3,4,5,6])]
dataset_train = SKADataset(mode='train', anchors=CUSTOM_ANCHORS_TINY, anchor_masks=anchor_masks, grid_size=8).get_dataset()
val_data = SKADataset(mode='validation', anchors=CUSTOM_ANCHORS_TINY, anchor_masks=anchor_masks, grid_size=8).get_dataset()
yolo.fit(dataset_train, epochs=NUM_EPOCHS,
        validation_data=val_data, steps_per_epoch=ITER_PER_EPOCH)
