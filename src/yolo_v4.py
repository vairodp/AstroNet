import numpy as np
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.layers import Concatenate, MaxPool2D, UpSampling2D
from layers import cnn_block, csp_block, scale_prediction

from configs.train_config import NUM_CLASSES, MAX_NUM_BBOXES, SCORE_THRESHOLD, loss_params
from metrics.mean_ap import mAP
from utils import decode_predictions, non_max_suppression

from anchors import compute_normalized_anchors

#TODO: remove repetition activation leaky relu        
#TODO: implement spp as a layer

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

    return tf.keras.Model(inputs, [output_1, output_2, output_3], name="CSPDarknet")
 
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
        x, num_anchors_stage=len(anchors[0]), num_classes=num_classes
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
        x, num_anchors_stage=len(anchors[1]), num_classes=num_classes
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
        x, num_anchors_stage=len(anchors[2]), num_classes=num_classes
    )

    return tf.keras.Model([input_1, input_2, input_3], [output_1, output_2, output_3], name="YOLOv3_head")

mAP_tracker = mAP(overlap_threshold=0.5, model='small_yolo', name='mAP_0.5')

class YOLOv4(tf.keras.Model):
    def __init__(self, input_shape, num_classes,
        anchors,
        training=False,
        yolo_max_boxes=MAX_NUM_BBOXES,
        yolo_iou_threshold=loss_params['iou_threshold'],
        yolo_score_threshold=SCORE_THRESHOLD,
        weights="darknet"):
        super().__init__(name='YOLOv4')
        self.num_classes = num_classes
        #self.input_shape = input_shape
        self.training = training
        self.anchors = anchors
        self.yolo_max_boxes = yolo_max_boxes
        self.yolo_iou_threshold = yolo_iou_threshold
        self.yolo_score_threshold = yolo_score_threshold
        #self.weights = weights

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

        self.model = tf.keras.Model(inputs=inputs, outputs=upper_features, name="YOLOv4")

        #weights_path = get_weights_by_keyword_or_path(weights, model=self.model)
        #if weights_path is not None:
        #    self.model.load_weights(str(weights_path), by_name=True, skip_mismatch=True)
    
    #@property
    #def metrics(self):
    #    return self.compiled_loss.metrics + self.compiled_metrics.metrics + [mAP_tracker]
    
    def call(self, x, training=False):
        return self.model(x, training)
    
    def train_step(self, data):
        x, y = data['image'], list(data['label'])

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        #mAP_tracker.update_state(y_pred, data['bbox'], data['num_of_bbox'])
        
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        x, y = data['image'], list(data['label'])
        
        y_pred = self(x, training=True)
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)

        #mAP_tracker.update_state(y_pred, data['bbox'], data['num_of_bbox'])

        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        if not isinstance(data, np.ndarray):
            data = data['image']
        output_1, output_2, output_3 = self(data)
        prediction_1 = decode_predictions(output_1, self.normalized_anchors[0])
        prediction_2 = decode_predictions(output_2, self.normalized_anchors[1])
        prediction_3 = decode_predictions(output_3, self.normalized_anchors[2])
        output = non_max_suppression([prediction_1, prediction_2, prediction_3],
        self.yolo_max_boxes, self.yolo_iou_threshold, self.yolo_score_threshold)
        return output 
