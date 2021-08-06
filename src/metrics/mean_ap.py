import numpy as np
import tensorflow as tf

from configs.train_config import SCORE_THRESHOLD, NUM_CLASSES, get_anchors
from utils import non_max_suppression

from metrics.ap_utils import APAccumulator, iou


class mAP(tf.keras.metrics.Metric):
    def __init__(self,
        num_classes=NUM_CLASSES, 
        overlap_threshold = SCORE_THRESHOLD,
        model='yolo',
        pr_samples=11, **kwargs):
        super().__init__(**kwargs)
        self.mAP = self.add_weight(name='mAP', initializer='zeros')
        self.n_class = num_classes
        self.overlap_threshold = overlap_threshold
        self.pr_scale = np.linspace(0, 1, pr_samples)
        self.total_accumulators = []
        self.reset_state()
        self.anchor_dict = get_anchors(model)
    
    def result(self):
        return self.mAP
    
    def reset_state(self):
        self.total_accumulators = []
        for _ in self.pr_scale:
            class_accumulators = [APAccumulator() for _ in range(self.n_class)]
            self.total_accumulators.append(class_accumulators)
        
        self.mAP.assign(0.0)

    def update_state(self, y_pred, true_bboxes, num_true_boxes, *args, **kwargs):
        pred_bboxes, pred_scores, pred_class_ids, valid_detections = non_max_suppression(y_pred, **self.anchor_dict)

        for frame in zip(pred_bboxes.numpy(), pred_class_ids.numpy(), 
                         pred_scores.numpy(), valid_detections.numpy(),
                         true_bboxes.numpy(), num_true_boxes.numpy()):
            pred_bbox, pred_class, pred_score, valid_detection, true_bbox, num_true_box = frame

            # get all predicion and label
            pred_bbox = pred_bbox[:valid_detection]
            pred_class = pred_class[:valid_detection]
            pred_score = pred_score[:valid_detection]
            true_box = true_bbox[:num_true_box]
            true_bbox = true_box[..., :4]
            true_class = true_box[..., 4]

            #
            frame = pred_bbox, pred_class, pred_score, true_bbox, true_class
            self._update_accumulators(*frame)
        
        self._update_mAP()

    def _update_accumulators(self, pred_bbox, pred_class, pred_score, true_bbox, true_class):
        if pred_bbox.ndim == 1:
            pred_bbox = np.repeat(pred_bbox[:, np.newaxis], 4, axis=1)
        IoUmask = None
        if len(pred_bbox) > 0:
            IoUmask = self.compute_IoU_mask(pred_bbox, true_bbox)
        for accumulators, r in zip(self.total_accumulators, self.pr_scale):
            self._evaluate_at_recall(IoUmask, accumulators, pred_class, pred_score, true_class, r)

    def _update_mAP(self):
        average_precisions = []
        for cls in range(self.n_class):
            precisions, recalls = self.compute_precision_recall_(cls)
            average_precisions.append(self.compute_ap(precisions, recalls))
        
        self.mAP.assign(np.mean(average_precisions))

    @staticmethod
    def _evaluate_at_recall(IoUmask, accumulators, pred_class, pred_score, true_class, confidence_threshold):
        pred_classes = pred_class.astype(np.int)
        gt_classes = true_class.astype(np.int)

        for i, acc in enumerate(accumulators):
            gt_number = np.sum(gt_classes == i)
            pred_mask = np.logical_and(pred_classes == i, pred_score >= confidence_threshold)
            pred_number = np.sum(pred_mask)
            if pred_number == 0:
                acc.inc_not_predicted(gt_number)
                continue

            IoU1 = IoUmask[pred_mask, :]
            mask = IoU1[:, gt_classes == i]

            tp = mAP.compute_true_positive(mask)
            fp = pred_number - tp
            fn = gt_number - tp
            acc.inc_good_prediction(tp)
            acc.inc_not_predicted(fn)
            acc.inc_bad_prediction(fp)
    
    def compute_IoU_mask(self, pred_bbox, true_bbox):
        IoU = iou(true_bbox, pred_bbox).numpy()
        # for each prediction select gt with the largest IoU and ignore the others
        for i in range(len(pred_bbox)):
            maxj = IoU[i, :].argmax()
            IoU[i, :maxj] = 0
            IoU[i, (maxj + 1):] = 0
        # make a mask of all "matched" predictions vs gt
        return IoU >= self.overlap_threshold

    @staticmethod
    def compute_true_positive(mask):
        # sum all gt with prediction of its class
        return np.sum(mask.any(axis=0))
    
    def compute_ap(self, precisions, recalls):
        previous_recall = 0
        average_precision = 0
        for precision, recall in zip(precisions[::-1], recalls[::-1]):
            average_precision += precision * (recall - previous_recall)
            previous_recall = recall
        return average_precision

    def compute_precision_recall_(self, class_index):
        precisions = []
        recalls = []
        for acc in self.total_accumulators:
            precisions.append(acc[class_index].precision)
            recalls.append(acc[class_index].recall)

        interpolated_precision = []
        for precision in precisions:
            last_max = 0
            if interpolated_precision:
                last_max = max(interpolated_precision)
            interpolated_precision.append(max(precision, last_max))
        precisions = interpolated_precision
        return precisions, recalls
    

# To be used when calling it from train and test step
#gt_boxes = data["bbox"]
#num_of_gt_boxes = data["num_of_bbox"]
