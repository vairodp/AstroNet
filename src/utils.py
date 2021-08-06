import numpy as np
import tensorflow as tf
import cv2

from configs.train_config import loss_params, SCORE_THRESHOLD, MAX_NUM_BBOXES

def non_sim(inputs):
    
    model_size = (128,128,3)
    max_output_size = MAX_NUM_BBOXES
    max_output_size_per_class = MAX_NUM_BBOXES
    iou_threshold = loss_params['iou_threshold']
    confidence_threshold = SCORE_THRESHOLD
                   
                            
    bbox, confs, class_probs = tf.split(inputs, [4, 1, -1], axis=-1)
    bbox=bbox/model_size[0]
    scores = confs * class_probs
    boxes, scores, classes, valid_detections = \
        tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(scores, (tf.shape(scores)[0], -1,
                                   tf.shape(scores)[-1])),
        max_output_size_per_class=max_output_size_per_class,
        max_total_size=max_output_size,
        iou_threshold=iou_threshold,
        score_threshold=confidence_threshold
    )
    return boxes, scores, classes, valid_detections

def non_max_suppression(inputs, anchors, anchor_masks):
    iou_threshold = loss_params['iou_threshold']
    score_threshold = SCORE_THRESHOLD
    max_bbox_size = MAX_NUM_BBOXES

    output_small, output_medium, output_large = inputs
    output_small = decode(output_small, anchors[anchor_masks[0]])
    output_medium = decode(output_medium, anchors[anchor_masks[1]])
    output_large = decode(output_large, anchors[anchor_masks[2]])

    # flatten output to shape [batch_size, toto_grid_size, *]
    bbox_small, objectness_small, class_probs_small = flatten_output(output_small)
    bbox_medium, objectness_medium, class_probs_medium = flatten_output(output_medium)
    bbox_large, objectness_large, class_probs_large = flatten_output(output_large)

    # concat all output
    bbox = tf.concat([bbox_small, bbox_medium, bbox_large], axis=1)
    confidence = tf.concat([objectness_small, objectness_medium, objectness_large], axis=1)
    class_probs = tf.concat([class_probs_small, class_probs_medium, class_probs_large], axis=1)

    scores = confidence * class_probs

    bboxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=max_bbox_size,
        max_total_size=max_bbox_size,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold
    )

    return bboxes, scores, classes, valid_detections


def decode(pred, anchors):
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    grid_size = tf.shape(pred)[1]
    box_xy, box_wh, objectness, class_probs = tf.split(pred, (2, 2, 1, -1), axis=-1)

    box_xy = 1.1 * tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)

    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs


def flatten_output(outputs):
    bbox, objectness, class_probs = outputs
    bbox = tf.reshape(bbox, (tf.shape(bbox)[0], -1, tf.shape(bbox)[-1]))
    objectness = tf.reshape(objectness, (tf.shape(objectness)[0], -1, tf.shape(objectness)[-1]))
    class_probs = tf.reshape(class_probs, (tf.shape(class_probs)[0], -1, tf.shape(class_probs)[-1]))

    return bbox, objectness, class_probs

def draw_outputs(img, boxes, objectness, classes, nums):
    
    print(boxes)
    
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    boxes=np.array(boxes)
    for i in range(nums):
        x1y1 = tuple((boxes[i,0:2] * 128).astype(np.int32))
        x2y2 = tuple((boxes[i,2:4] * 128).astype(np.int32))
        
        print(x1y1)
        print(x2y2)
        
        img = cv2.rectangle(img, (x1y1), (x2y2), (255,0,0), 2)
        
        img = cv2.putText(img, '{} {:.4f}'.format(
            "test", objectness[i]),
                          (x1y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    
    return img