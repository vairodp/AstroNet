import numpy as np

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.losses import Loss, binary_crossentropy, categorical_crossentropy
from utils import decode_predictions

from configs.train_config import NUM_CLASSES, loss_params


class YoloLoss(Loss):
    def __init__(self, 
                anchors,
                iou_threshold,
                use_focal_obj_loss=True,
                use_focal_loss=True,
                class_weights=None,
                smooth_factor=0.0, 
                use_giou=False, 
                use_ciou=False, 
                use_diou=False, 
                **kwargs):
        super().__init__(**kwargs)
        self.smooth_factor = smooth_factor
        self.iou_threshold = iou_threshold
        self.use_focal_obj_loss = use_focal_obj_loss
        self.use_focal_loss = use_focal_loss
        self.use_giou = use_giou
        self.use_ciou = use_ciou
        self.use_diou = use_diou
        self.valid_anchors = anchors
        self.class_weights = class_weights
    
    def weighted_cetegorical_crossentropy(self, class_true, class_pred):
        if self.class_weights is not None:
            weights = backend.variable(self.class_weights)
            # scale predictions so that the class probas of each sample sum to 1
            class_pred = tf.math.divide_no_nan(class_pred, backend.sum(class_pred, axis=-1, keepdims=True))
            # clip to prevent NaN's and Inf's
            class_pred = backend.clip(class_pred, backend.epsilon(), 1 - backend.epsilon())
            # calc
            loss = class_true * backend.log(class_pred) * weights
            loss = -backend.sum(loss, -1)
        else:
            loss = binary_crossentropy(class_true, class_pred)
        return loss

    def label_smoothing(self, true_labels):
        return true_labels * (1.0 - self.smooth_factor) + self.smooth_factor / NUM_CLASSES
    
    def get_true_scores(self, y_true):
        true_box, true_obj, true_class = tf.split(
            y_true, (4, 1, NUM_CLASSES), axis=-1)
        true_xy = true_box[..., 0:2]
        true_wh = true_box[..., 2:4]
        
        return true_xy, true_wh, true_obj, true_class

    def interpret_true_boxes(self, y_true):
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)

        true_xy, true_wh, _, _ = self.get_true_scores(y_true)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - \
                      tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / self.valid_anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh),
                               tf.zeros_like(true_wh), true_wh)

        return true_xy, true_wh

    def iou(self, box_true, box_pred):
        width = tf.maximum(tf.minimum(box_true[..., 2], box_pred[..., 2]) -
                           tf.maximum(box_true[..., 0], box_pred[..., 0]), 0)
        height = tf.maximum(tf.minimum(box_true[..., 3], box_pred[..., 3]) -
                           tf.maximum(box_true[..., 1], box_pred[..., 1]), 0)
                                                   
        area = width * height
        box_true_area = (box_true[..., 2] - box_true[..., 0]) * \
                     (box_true[..., 3] - box_true[..., 1])
        box_pred_area = (box_pred[..., 2] - box_pred[..., 0]) * \
                     (box_pred[..., 3] - box_pred[..., 1])


        area_union = box_true_area + box_pred_area - area
        iou = tf.math.divide_no_nan(area, area_union)
        
        return area_union, iou
    
    def broadcast_iou(self, box_true, box_pred):
        # broadcast boxes
        #box_true = tf.reshape(box_true, (-1, 4))
        #print('NEW BOX TRUE: ' , box_true.shape)

        #box_true = box_true[:,-1]
        
        box_pred = tf.expand_dims(box_pred, -2)
        box_true = tf.expand_dims(box_true, 0)
        
        # new_shape: (..., N, (x1, y1, x2, y2))
        new_shape = tf.broadcast_dynamic_shape(tf.shape(box_pred), tf.shape(box_true))
        box_true = tf.broadcast_to(box_true, new_shape)
        box_pred = tf.broadcast_to(box_pred, new_shape)

        _, iou = self.iou(box_true, box_pred)

        return iou

    def giou(self, box_true, box_pred):
        area_union, iou = self.iou(box_true, box_pred)

        enclose_left_up = tf.minimum(box_true[..., :2], box_pred[..., :2])
        enclose_right_down = tf.maximum(box_true[..., 2:], box_pred[..., 2:])
        enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
        enclose_area = enclose[..., 0] * enclose[..., 1]
        giou = iou - 1.0 * tf.math.divide_no_nan((enclose_area - area_union), enclose_area)

        return giou

    def diou(self, box_true, box_pred):

        _, iou = self.iou(box_true,box_pred)

        # find enclosed area
        enclose_left_up = tf.minimum(box_true[..., :2], box_pred[..., :2])
        enclose_right_down = tf.maximum(box_true[..., 2:], box_pred[..., 2:])

        enclose_wh = enclose_right_down - enclose_left_up
        c2 = tf.square(enclose_wh[..., 0]) + tf.square(enclose_wh[..., 1])

        box_true_center_x = (box_true[..., 0] + box_true[..., 2]) / 2
        box_true_center_y = (box_true[..., 1] + box_true[..., 3]) / 2
        box_pred_center_x = (box_pred[..., 0] + box_pred[..., 2]) / 2
        box_pred_center_y = (box_pred[..., 1] + box_pred[..., 3]) / 2

        d = tf.square(box_true_center_x - box_pred_center_x) + tf.square(box_true_center_y - box_pred_center_y)

        diou = iou - tf.math.divide_no_nan(d, c2)

        return diou

    def ciou(self, box_true, box_pred):
        diou = self.diou(box_true, box_pred)
        _, iou = self.iou(box_true, box_pred)

        box_true_w, box_true_h = box_true[..., 2] - box_true[..., 0], box_true[..., 3] - box_true[..., 1]
        box_pred_w, box_pred_h = box_pred[..., 2] - box_pred[..., 0], box_pred[..., 3] - box_pred[..., 1]

        atan = tf.atan(tf.math.divide_no_nan(box_true_w, box_true_h)) - tf.atan(tf.math.divide_no_nan(box_pred_w, box_pred_h))
        v = (atan * 2 / np.pi) ** 2
        alpha = tf.stop_gradient(tf.math.divide_no_nan(v, 1 - iou + v))
        
        ciou = diou - alpha * v

        return ciou

    def focal_loss(self, y_true, y_pred, gamma=2.0, alpha=0.25, label_smoothing=0):
        sigmoid_loss = binary_crossentropy(y_true, y_pred, label_smoothing=label_smoothing)
        sigmoid_loss = tf.expand_dims(sigmoid_loss, axis=-1)

        p_t = ((y_true * y_pred) + ((1 - y_true) * (1 - y_pred)))
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        alpha_weight_factor = (y_true * alpha + (1 - y_true) * (1 - alpha))

        sigmoid_focal_loss = modulating_factor * alpha_weight_factor * sigmoid_loss

        sigmoid_focal_loss = tf.reduce_sum(sigmoid_focal_loss, axis=-1)

        return sigmoid_focal_loss

    def compute_loss(self, y_true, y_pred):
        box_pred, obj_pred, class_pred, raw_box_pred = decode_predictions(y_pred, self.valid_anchors)

        xy_true, wh_true, obj_true, class_true = self.get_true_scores(y_true)
        box_true = tf.concat([xy_true - wh_true/2.0, xy_true + wh_true/2.0], axis=-1)
        
        class_true = self.label_smoothing(class_true)
        weights = 2 - wh_true[..., 0] * wh_true[..., 1]

        # in order to element-wise multiply the result from tf.reduce_sum
        # we need to squeeze one dimension for objectness here
        obj_mask = tf.squeeze(obj_true, axis=-1)
        
        best_iou, _, _ = tf.map_fn(
            lambda x: (tf.reduce_max(self.broadcast_iou(tf.boolean_mask(
                x[1], tf.cast(x[2], tf.bool)), x[0]), axis=-1), 0, 0),
            (box_pred, box_true, obj_mask))
        ignore_mask = tf.cast(best_iou < self.iou_threshold, tf.float32)

        
        if self.use_focal_obj_loss:
            confidence_loss = self.focal_loss(obj_true, obj_pred)
        else:
            confidence_loss = binary_crossentropy(obj_true, obj_pred)
            confidence_loss = obj_mask * confidence_loss + (1 - obj_mask) * ignore_mask * confidence_loss

        if self.use_focal_loss:
            class_loss = self.focal_loss(class_true, class_pred)
        else:
            class_loss = obj_mask * self.weighted_cetegorical_crossentropy(class_true, class_pred)

        # box loss
        if self.use_giou:
            giou = self.giou(box_true, box_pred)
            box_loss = obj_mask * weights * (1 - giou)
            box_loss = tf.reduce_sum(box_loss, axis=(1, 2, 3))
        elif self.use_ciou:
            ciou = self.ciou(box_true, box_pred)
            box_loss = obj_mask * weights * (1 - ciou)
            box_loss = tf.reduce_sum(box_loss, axis=(1, 2, 3))
        elif self.use_diou:
            diou = self.diou(box_true, box_pred)
            box_loss = obj_mask * weights * (1 - diou)
            box_loss = tf.reduce_sum(box_loss, axis=(1, 2, 3))
        else:
            # traditional loss for xy and wh
            pred_xy = raw_box_pred[..., 0:2]
            pred_wh = raw_box_pred[..., 2:4]

            # invert box equation
            true_xy, true_wh = self.interpret_true_boxes(y_true)

            # sum squared box loss
            xy_loss = obj_mask * weights * \
                      tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
            wh_loss = obj_mask * weights * \
                      tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)

            xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
            wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
            box_loss = xy_loss + wh_loss

        # sum of all loss
        confidence_loss = tf.reduce_sum(confidence_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return box_loss + confidence_loss + class_loss

    def call(self, y_true, y_pred):

        loss = self.compute_loss(y_true, y_pred)

        return loss