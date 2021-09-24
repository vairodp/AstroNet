import numpy as np
import tensorflow as tf

from loss import yolo3_loss
from anchors import compute_normalized_anchors
from layers import cnn_block, csp_block, scale_prediction
from tensorflow.keras.layers import Concatenate, MaxPool2D, UpSampling2D, Input, Lambda

from configs.train_config import NUM_CLASSES, MAX_NUM_BBOXES, SCORE_THRESHOLD, USE_CUSTOM_ANCHORS, loss_params

def csp_darknet53(input_shape):

    inputs = tf.keras.Input(shape=input_shape)

    x = cnn_block(inputs, num_filters=32, kernel_size=3, strides=1, activation="mish")
    
    x = cnn_block(
        x,
        num_filters=64,
        kernel_size=3,
        strides=2,
        zero_padding=True,
        padding="valid",
        activation="mish",
    ) 
    route = cnn_block(x, num_filters=64, kernel_size=1, strides=1, activation="mish")

    shortcut = cnn_block(x, num_filters=64, kernel_size=1, strides=1, activation="mish")
    x = cnn_block(shortcut, num_filters=32, kernel_size=1, strides=1, activation="mish")
    x = cnn_block(x, num_filters=64, kernel_size=3, strides=1, activation="mish")

    x = x + shortcut
    x = cnn_block(x, num_filters=64, kernel_size=1, strides=1, activation="mish")
    x = Concatenate()([x, route])
    x = cnn_block(x, num_filters=64, kernel_size=1, strides=1, activation="mish")

    x = csp_block(x, filters=128, num_blocks=2)

    output_1 = csp_block(x, filters=256, num_blocks=8)

    output_2 = csp_block(output_1, filters=512, num_blocks=8)

    output_3 = csp_block(output_2, filters=1024, num_blocks=4)

    return tf.keras.Model(inputs, [output_1, output_2, output_3], name="CSPDarknet53")
 
def yolov4_neck(input_shapes):    
    
    input_1 = tf.keras.Input(shape=filter(None, input_shapes[0]))
    input_2 = tf.keras.Input(shape=filter(None, input_shapes[1]))
    input_3 = tf.keras.Input(shape=filter(None, input_shapes[2]))

    x = cnn_block(input_3, num_filters=512, kernel_size=1, strides=1, activation="leaky_relu")
    x = cnn_block(x, num_filters=1024, kernel_size=3, strides=1, activation="leaky_relu")
    x = cnn_block(x, num_filters=512, kernel_size=1, strides=1, activation="leaky_relu")

    maxpool_1 = MaxPool2D((5, 5), strides=1, padding="same")(x)
    maxpool_2 = MaxPool2D((9, 9), strides=1, padding="same")(x)
    maxpool_3 = MaxPool2D((13, 13), strides=1, padding="same")(x)

    spp = Concatenate()([maxpool_3, maxpool_2, maxpool_1, x])

    x = cnn_block(spp, num_filters=512, kernel_size=1, strides=1, activation="leaky_relu")
    x = cnn_block(x, num_filters=1024, kernel_size=3, strides=1, activation="leaky_relu")
    output_3 = cnn_block(
        x, num_filters=512, kernel_size=1, strides=1, activation="leaky_relu"
    )
    x = cnn_block(
        output_3, num_filters=256, kernel_size=1, strides=1, activation="leaky_relu"
    )

    upsampled = UpSampling2D()(x)

    x = cnn_block(input_2, num_filters=256, kernel_size=1, strides=1, activation="leaky_relu")
    x = Concatenate()([x, upsampled])

    x = cnn_block(x, num_filters=256, kernel_size=1, strides=1, activation="leaky_relu")
    x = cnn_block(x, num_filters=512, kernel_size=3, strides=1, activation="leaky_relu")
    x = cnn_block(x, num_filters=256, kernel_size=1, strides=1, activation="leaky_relu")
    x = cnn_block(x, num_filters=512, kernel_size=3, strides=1, activation="leaky_relu")
    output_2 = cnn_block(
        x, num_filters=256, kernel_size=1, strides=1, activation="leaky_relu"
    )
    x = cnn_block(
        output_2, num_filters=128, kernel_size=1, strides=1, activation="leaky_relu"
    )

    upsampled = UpSampling2D()(x)

    x = cnn_block(input_1, num_filters=128, kernel_size=1, strides=1, activation="leaky_relu")
    x = tf.keras.layers.Concatenate()([x, upsampled])

    x = cnn_block(x, num_filters=128, kernel_size=1, strides=1, activation="leaky_relu")
    x = cnn_block(x, num_filters=256, kernel_size=3, strides=1, activation="leaky_relu")
    x = cnn_block(x, num_filters=128, kernel_size=1, strides=1, activation="leaky_relu")
    x = cnn_block(x, num_filters=256, kernel_size=3, strides=1, activation="leaky_relu")
    output_1 = cnn_block(
        x, num_filters=128, kernel_size=1, strides=1, activation="leaky_relu"
    )

    return tf.keras.Model(
        [input_1, input_2, input_3], [output_1, output_2, output_3], name="YOLOv4_neck"
    )

def yolov3_head(
    input_shapes,
    anchors,
    num_classes):
 
    input_1 = tf.keras.Input(shape=filter(None, input_shapes[0]))
    input_2 = tf.keras.Input(shape=filter(None, input_shapes[1]))
    input_3 = tf.keras.Input(shape=filter(None, input_shapes[2]))

    x = cnn_block(input_1, num_filters=256, kernel_size=3, strides=1, activation="leaky_relu")
    output_1 = scale_prediction(
        x, num_anchors_stage=len(anchors[0]), num_classes=num_classes, num=93
    )

    x = cnn_block(
        input_1,
        num_filters=256,
        kernel_size=3,
        strides=2,
        zero_padding=True,
        padding="valid",
        activation="leaky_relu",
    )
    x = Concatenate()([x, input_2])
    x = cnn_block(x, num_filters=256, kernel_size=1, strides=1, activation="leaky_relu")
    x = cnn_block(x, num_filters=512, kernel_size=3, strides=1, activation="leaky_relu")
    x = cnn_block(x, num_filters=256, kernel_size=1, strides=1, activation="leaky_relu")
    x = cnn_block(x, num_filters=512, kernel_size=3, strides=1, activation="leaky_relu")
    connection = cnn_block(
        x, num_filters=256, kernel_size=1, strides=1, activation="leaky_relu"
    )
    x = cnn_block(
        connection, num_filters=512, kernel_size=3, strides=1, activation="leaky_relu"
    )
    output_2 = scale_prediction(
        x, num_anchors_stage=len(anchors[1]), num_classes=num_classes, num=101
    )

    x = cnn_block(
        connection,
        num_filters=512,
        kernel_size=3,
        strides=2,
        zero_padding=True,
        padding="valid",
        activation="leaky_relu",
    )
    x = Concatenate()([x, input_3])
    x = cnn_block(x, num_filters=512, kernel_size=1, strides=1, activation="leaky_relu")
    x = cnn_block(x, num_filters=1024, kernel_size=3, strides=1, activation="leaky_relu")
    x = cnn_block(x, num_filters=512, kernel_size=1, strides=1, activation="leaky_relu")
    x = cnn_block(x, num_filters=1024, kernel_size=3, strides=1, activation="leaky_relu")
    x = cnn_block(x, num_filters=512, kernel_size=1, strides=1, activation="leaky_relu")
    x = cnn_block(x, num_filters=1024, kernel_size=3, strides=1, activation="leaky_relu")
    output_3 = scale_prediction(
        x, num_anchors_stage=len(anchors[2]), num_classes=num_classes, num=109
    )

    return tf.keras.Model([input_1, input_2, input_3], [output_1, output_2, output_3], name="YOLOv3_head")

class YOLOv4(tf.keras.Model):
    def __init__(self, input_shape, 
        num_classes,
        anchors,
        yolo_max_boxes=MAX_NUM_BBOXES,
        yolo_iou_threshold=loss_params['iou_threshold'],
        yolo_score_threshold=SCORE_THRESHOLD):
        super().__init__(name='YOLOv4')
        self.num_classes = num_classes
        self.anchors = anchors
        self.yolo_max_boxes = yolo_max_boxes
        self.yolo_iou_threshold = yolo_iou_threshold
        self.yolo_score_threshold = yolo_score_threshold

        if (input_shape[0] % 32 != 0) | (input_shape[1] % 32 != 0):
            raise ValueError(
            f"Provided height and width in input_shape {input_shape} is not a multiple of 32"
        )

        backbone = csp_darknet53(input_shape)

        neck = yolov4_neck(input_shapes=backbone.output_shape)

        self.normalized_anchors = compute_normalized_anchors(anchors, input_shape)
        head = yolov3_head(
            input_shapes=neck.output_shape,
            anchors=self.normalized_anchors,
            num_classes=num_classes)

        inputs = tf.keras.Input(shape=input_shape)
        lower_features = backbone(inputs)
        medium_features = neck(lower_features)
        upper_features = head(medium_features)

        anchors = np.array([anchor for subl in self.normalized_anchors for anchor in subl])

        y_true = [Input(shape=(None, None, len(self.anchors[l]), NUM_CLASSES+5), name='y_true_{}'.format(l)) for l in range(3)]

        self.model_body = tf.keras.Model(inputs=inputs, outputs=upper_features, name="YOLOv4")
        model_loss, location_loss, confidence_loss, class_loss = Lambda(
            yolo3_loss, name='yolo_loss',
            arguments={'anchors': anchors, 
                    'anchor_masks': 'custom' if USE_CUSTOM_ANCHORS else 'yolo',
                    'num_layers': 3,
                    'num_classes': NUM_CLASSES, 
                    'ignore_thresh': loss_params['iou_threshold'], 
                    'label_smoothing': loss_params['smooth_factor'], 
                    'elim_grid_sense': loss_params['elim_grid_sense'],
                    'use_vf_loss': loss_params['use_vf_loss'], 
                    'use_focal_loss': loss_params['use_focal_loss'],
                    'use_focal_obj_loss': loss_params['use_focal_obj_loss'],
                    'use_diou_loss': loss_params['use_diou'],
                    'use_giou_loss': loss_params['use_giou'],
                    'use_ciou_loss': loss_params['use_ciou'],
                    'focal_gamma': loss_params['focal_gamma']})([*upper_features, *y_true])

        self.model = tf.keras.Model([self.model_body.input, *y_true], model_loss)

        loss_dict = {'location_loss':location_loss, 'confidence_loss':confidence_loss, 'class_loss':class_loss}
        self.add_metrics(loss_dict)
    
    def add_metrics(self, metric_dict):
        '''
        add metric scalar tensor into model, which could be tracked in training
        log and tensorboard callback
        '''
        for (name, metric) in metric_dict.items():
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
        metrics = {m.name: m.result() for m in self.metrics}
        metrics['reg_loss'] = (loss - y_pred)[0]
        return metrics
    
    def test_step(self, data):
        x, y = data['image'], list(data['label'])
        
        y_pred = self([x, *y], training=True)
        loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)

        metrics = {m.name: m.result() for m in self.metrics}
        metrics['reg_loss'] = (loss - y_pred)[0]

        return metrics

    def predict_step(self, data):
        if isinstance(data, dict):
            x = data['image']
        else:
            x = data
        return self.model_body(x)
