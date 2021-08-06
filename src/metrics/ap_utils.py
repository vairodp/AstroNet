import tensorflow as tf
import numpy as np

class APAccumulator:
    def __init__(self):
        self.TP, self.FP, self.FN = 0, 0, 0

    def inc_good_prediction(self, value=1):
        self.TP += value

    def inc_bad_prediction(self, value=1):
        self.FP += value

    def inc_not_predicted(self, value=1):
        self.FN += value
    
    @property
    def precision(self):
        total_predicted = self.TP + self.FP
        if total_predicted == 0:
            total_gt = self.TP + self.FN
            if total_gt == 0:
                return 1.
            else:
                return 0.
        return float(self.TP) / total_predicted
    
    @property
    def recall(self):
        total_gt = self.TP + self.FN
        if total_gt == 0:
            return 1.
        return float(self.TP) / total_gt
    
    def __str__(self):
        to_print = f"""
        TP: {self.TP}
        FP: {self.FP}
        FN: {self.FN}
        Precision: {self.precision}
        Recall: {self.recall}
        """
        return to_print

def _intersection_area(box_true, box_pred):
    # box_true.shape = [A, 4]
    # box_pred.shape = [B, 4]
    # intersection.shape = [A, B]
    extended_true_bb = box_true[:, tf.newaxis, :]
    extended_pred_bb = box_pred[tf.newaxis, :, :]
    max_xy = tf.minimum(extended_true_bb[:, :, 2:], extended_pred_bb[:, :, 2:])
    min_xy = tf.maximum(extended_true_bb[:, :, :2], extended_pred_bb[:, :, :2])

    diff_xy = (max_xy - min_xy)
    intersection = np.clip(diff_xy, a_min=0, a_max=np.max(diff_xy))
    return intersection[..., 0] * intersection[..., 1]

def iou(box_true, box_pred):
    intersection = _intersection_area(box_true, box_pred)
    box_true_area = (box_true[..., 2] - box_true[..., 0]) * \
                    (box_true[..., 3] - box_true[..., 1])
    box_true_area = box_true_area[:, tf.newaxis]

    box_pred_area = (box_pred[..., 2] - box_pred[..., 0]) * \
                    (box_pred[..., 3] - box_pred[..., 1])
    box_pred_area = box_pred_area[tf.newaxis, :]

    area_union = box_true_area + box_pred_area - intersection
    iou = tf.transpose(intersection / area_union)
    
    return iou
