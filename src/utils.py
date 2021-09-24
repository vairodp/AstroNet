import tensorflow as tf

from configs.train_config import NUM_CLASSES
from loss import yolo3_decode

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask

def bbox_to_x1y1x2y2(bbox):
    # bbox = [x, y, w, h] --> bbox = [x1, y1, x2, y2]

    bbox_xy = bbox[..., 0:2]
    bbox_wh = bbox[..., 2:4]
    bbox_x1y1 = bbox_xy - bbox_wh / 2
    bbox_x2y2 = bbox_xy + bbox_wh / 2

    return tf.concat([bbox_x1y1, bbox_x2y2], axis=-1)

def bbox_to_xywh(bbox):
    # bbox = [x1, y1, x2, y2] --> bbox = [x, y, w, h]

    bbox_x1y1 = bbox[..., 0:2]
    bbox_x2y2 = bbox[..., 2:4]

    bbox_wh = bbox_x2y2 - bbox_x1y1
    bbox_xy = bbox_x1y1 + bbox_wh / 2

    return tf.concat([bbox_xy, bbox_wh], axis=-1)

def non_max_suppression(yolo_feats, yolo_max_boxes, yolo_iou_threshold, yolo_score_threshold):
    """
    Applies the non max suppression to YOLO features and returns predicted boxes.
    """
    bbox_per_stage, objectness_per_stage, class_probs_per_stage = [], [], []

    for stage_feats in yolo_feats:
        num_boxes = (
            stage_feats[0].shape[1] * stage_feats[0].shape[2] * stage_feats[0].shape[3]
        )  # num_anchors * grid_x * grid_y
        bbox_per_stage.append(
            tf.reshape(
                stage_feats[0],
                (tf.shape(stage_feats[0])[0], num_boxes, stage_feats[0].shape[-1]),
            )
        )  # [None,num_boxes,4]
        objectness_per_stage.append(
            tf.reshape(
                stage_feats[1],
                (tf.shape(stage_feats[1])[0], num_boxes, stage_feats[1].shape[-1]),
            )
        )  # [None,num_boxes,1]
        class_probs_per_stage.append(
            tf.reshape(
                stage_feats[2],
                (tf.shape(stage_feats[2])[0], num_boxes, stage_feats[2].shape[-1]),
            )
        )  # [None,num_boxes,num_classes]

    bbox = tf.concat(bbox_per_stage, axis=1)
    objectness = tf.concat(objectness_per_stage, axis=1)
    class_probs = tf.concat(class_probs_per_stage, axis=1)

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.expand_dims(bbox, axis=2),
        scores=objectness * class_probs,
        max_output_size_per_class=yolo_max_boxes,
        max_total_size=yolo_max_boxes,
        iou_threshold=yolo_iou_threshold,
        score_threshold=yolo_score_threshold,
    )

    return [boxes, scores, classes, valid_detections]

def decode_predictions(pred, anchors, input_shape):
    prediction = yolo3_decode(pred, anchors,
        NUM_CLASSES, input_shape, scale_x_y=None, calc_loss=False)
    box_xy, box_wh, box_confidence, box_class_probs = prediction
    bbox = tf.concat([box_xy, box_wh], axis=-1)
    print(tf.shape(bbox))
    bbox = bbox_to_x1y1x2y2(bbox)
    prediction = [bbox, box_confidence, box_class_probs]

    return prediction