import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import MaxPool2D, Input, Lambda
from configs.train_config import NUM_CLASSES, MAX_NUM_BBOXES, loss_params
from loss import yolo3_loss
from anchors import compute_normalized_anchors, YOLOLITE_ANCHORS

from layers import cnn_block, route_group, scale_prediction, upsample

def cspdarknet53_tiny(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    input_data = cnn_block(inputs, num_filters=32, kernel_size=3, strides=2, zero_padding=True, padding='valid')
    input_data = cnn_block(input_data, num_filters=64, kernel_size=3, strides=2, zero_padding=True, padding='valid')
    input_data = cnn_block(input_data, num_filters=64, kernel_size=3, strides=1)

    route = input_data
    input_data = route_group(input_data, 2, 1)
    input_data = cnn_block(input_data, num_filters=32, kernel_size=3, strides=1)
    route_1 = input_data
    input_data = cnn_block(input_data, num_filters=32, kernel_size=3, strides=1)
    input_data = tf.concat([input_data, route_1], axis=-1)
    input_data = cnn_block(input_data, num_filters=64, kernel_size=1, strides=1)
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = MaxPool2D(2, 2, 'same')(input_data)

    input_data = cnn_block(input_data, num_filters=128, kernel_size=3, strides=1)
    route = input_data
    input_data = route_group(input_data, 2, 1)
    input_data = cnn_block(input_data, num_filters=64, kernel_size=3, strides=1)
    route_1 = input_data
    input_data = cnn_block(input_data, num_filters=64, kernel_size=3, strides=1)
    input_data = tf.concat([input_data, route_1], axis=-1)
    input_data = cnn_block(input_data, num_filters=128, kernel_size=1, strides=1)
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = MaxPool2D(2, 2, 'same')(input_data)

    input_data = cnn_block(input_data, num_filters=256, kernel_size=3, strides=1)
    route = input_data
    input_data = route_group(input_data, 2, 1)
    input_data = cnn_block(input_data, num_filters=128, kernel_size=3, strides=1)
    route_1 = input_data
    input_data = cnn_block(input_data, num_filters=128, kernel_size=3, strides=1)
    input_data = tf.concat([input_data, route_1], axis=-1)
    input_data = cnn_block(input_data, num_filters=256, kernel_size=1, strides=1)
    output_1 = input_data
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = MaxPool2D(2, 2, 'same')(input_data)

    output_2 = cnn_block(input_data, num_filters=512, kernel_size=3, strides=1)

    return tf.keras.Model(inputs, [output_1, output_2], name="CSPDarknet53-tiny")

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

        backbone = cspdarknet53_tiny(input_shape)
        print(backbone.summary())
        print(backbone.outputs)

        self.normalized_anchors = compute_normalized_anchors(anchors, input_shape)

        inputs = tf.keras.Input(shape=input_shape)
        output_1, output_2 = backbone(inputs)
        output_2 = cnn_block(output_2, num_filters=256, kernel_size=1, strides=1)
        route = cnn_block(output_2, num_filters=512, kernel_size=3, strides=1)
        out_lbbox = scale_prediction(route, num_classes=NUM_CLASSES)

        output_2 = cnn_block(output_2, num_filters=128, kernel_size=1, strides=1)
        output_2 = upsample(output_2)
        output_2 = tf.concat([output_2, output_1], axis=-1)

        route = cnn_block(output_2, num_filters=256, kernel_size=3, strides=1)
        out_mbbox = scale_prediction(route, num_classes=NUM_CLASSES)

        anchors = np.array([anchor for subl in self.normalized_anchors for anchor in subl])

        y_true = [Input(shape=(None, None, 3, NUM_CLASSES+5), name='y_true_{}'.format(l)) for l in range(2)]

        self.model_body = tf.keras.Model(inputs=inputs, outputs=[out_mbbox, out_lbbox], name="YOLOv4")
        model_loss, location_loss, confidence_loss, class_loss = Lambda(yolo3_loss, name='yolo_loss',
            arguments={'anchors': anchors, 'num_classes': NUM_CLASSES, 'ignore_thresh': 0.5, 'label_smoothing': loss_params['smooth_factor'], 'elim_grid_sense': False})(
        [out_mbbox, out_lbbox, *y_true])

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
        x, y = data['image'], list(data['label'])

        with tf.GradientTape() as tape:
            y_pred = self([x, *y], training=True)  # Forward pass
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
        x, y = data['image'], list(data['label'])
        
        y_pred = self([x, *y], training=True)
        loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)

        #mAP_tracker.update_state(y_pred, data['bbox'], data['num_of_bbox'])
        metrics = {m.name: m.result() for m in self.metrics}
        metrics['reg_loss'] = (loss - y_pred)[0]

        return metrics