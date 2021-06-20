import tensorflow as tf
from tensorflow.keras.losses import Loss, binary_crossentropy

from configs.yolo_v4 import NUM_CLASSES, loss_params

class YoloLoss(Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.anchors = '' #TODO

    def interpret_boxes(self, pred):
        grid_size = tf.shape(pred)[1]
        box_xy, box_wh, objectness, probs = tf.split(pred, (2, 2, 1, -1), axis=-1)

        box_xy = loss_params["sensitivity_factor"] * tf.sigmoid(box_xy)
        objectness = tf.sigmoid(objectness)
        probs = tf.sigmoid(probs)
        box = tf.concat([box_xy, box_wh], axis=-1)

        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)

        box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
        box_wh = tf.exp(box_wh) * self.anchors

        box_x1y1 = box_xy - box_wh / 2
        box_x2y2 = box_xy + box_wh / 2
        bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

        return bbox, objectness, probs, box

