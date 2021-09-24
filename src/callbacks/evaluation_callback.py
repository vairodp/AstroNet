import os
import tensorflow as tf
from configs.train_config import IMG_SIZE, MAX_NUM_BBOXES, SCORE_THRESHOLD, loss_params
from anchors import compute_normalized_anchors
from utils import decode_predictions, non_max_suppression
from metrics.utils import true_label_to_file, predictions_to_file

TRUE_LABEL_PATH = 'metrics/input/ground-truth/'
PRED_PATH = 'metrics/input/detection-results/'

class EvaluationCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data,
                anchors,
                epoch_interval=20, 
                yolo_max_boxes=MAX_NUM_BBOXES,
                yolo_iou_threshold=loss_params['iou_threshold'], 
                overlap_threshold=SCORE_THRESHOLD):
        super().__init__()
        self.epoch_interval = epoch_interval
        self.val_data = val_data
        self.yolo_max_boxes = yolo_max_boxes
        self.yolo_iou_threshold = yolo_iou_threshold
        self.overlap_threshold = overlap_threshold
        self.anchors = compute_normalized_anchors(anchors, (IMG_SIZE,IMG_SIZE,3))

    def on_epoch_end(self, epoch, logs):
        if epoch % self.epoch_interval == 0 and epoch != 0:
            num = 0
            for feature in self.val_data:
                pred = self.model.model_body.predict(feature['image'])
                output_1, output_2, output_3 = pred
                input_shape = tf.constant([IMG_SIZE, IMG_SIZE])
                prediction_1 = decode_predictions(tf.convert_to_tensor(output_1), self.anchors[0], input_shape)
                prediction_2 = decode_predictions(tf.convert_to_tensor(output_2), self.anchors[1], input_shape)
                prediction_3 = decode_predictions(tf.convert_to_tensor(output_3), self.anchors[2], input_shape)
                pred_bboxes, pred_scores, pred_class_ids, valid_detections = non_max_suppression([prediction_1, prediction_2, prediction_3], 
                                                                                self.yolo_max_boxes,
                                                                                self.yolo_iou_threshold, 
                                                                                self.overlap_threshold)
                true_labels = feature['bbox']
                num_true_box = feature['num_of_bbox']
                for labels, num_box, pred_bbox, pred_class_id, pred_score, valid_detection in zip(true_labels, 
                                                                                num_true_box, 
                                                                                pred_bboxes, 
                                                                                pred_class_ids, 
                                                                                pred_scores,
                                                                                valid_detections):
                    pred_bbox = pred_bbox[:valid_detection]
                    pred_score = pred_score[:valid_detection] 
                    pred_class_id = tf.cast(pred_class_id[:valid_detection], tf.int8)
                    label = labels[:num_box]
                    true_bbox = labels[..., :4]
                    true_class = tf.cast(label[..., 4], tf.int8)
                    true_label_to_file(file_path=TRUE_LABEL_PATH+f'image_{num}.txt', 
                                        true_bbox=true_bbox, 
                                        true_class=true_class)
                    predictions_to_file(file_path=PRED_PATH + f'image_{num}.txt', 
                                        pred_boxes=pred_bbox, 
                                        pred_classes=pred_class_id, 
                                        pred_confs=pred_score)
                    num += 1
            os.system("python metrics/compute_ap.py")
                
