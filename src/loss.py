import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

def varifocal_loss(y_true, y_pred, gamma=1.5, alpha=0.25):
    loss = K.binary_crossentropy(y_true, y_pred, from_logits=True)
    pred_prob = K.sigmoid(y_pred)  # prob from logits

    focal_weight = y_true * tf.cast((y_true > 0.0), tf.float32) + alpha * tf.pow(tf.abs(pred_prob - y_true), gamma) * tf.cast((y_true <= 0.0), tf.float32)
    loss *= focal_weight

    return loss

def yolo3_decode(feats, anchors, num_classes, input_shape, scale_x_y=None, calc_loss=False):
    """Decode final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3] # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    if scale_x_y:
        # Eliminate grid sensitivity trick involved in YOLOv4
        box_xy_tmp = K.sigmoid(feats[..., :2]) * scale_x_y - (scale_x_y - 1) / 2
        box_xy = (box_xy_tmp + grid) / K.cast(grid_shape[..., ::-1], K.dtype(feats))
    else:
        box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[..., ::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[..., ::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs

def sigmoid_focal_loss(y_true, y_pred, gamma=4.3, alpha=0.25):
    """
    Compute sigmoid focal loss.
    """
    sigmoid_loss = K.binary_crossentropy(y_true, y_pred, from_logits=True)

    pred_prob = tf.sigmoid(y_pred)
    p_t = ((y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob)))
    modulating_factor = tf.pow(1.0 - p_t, gamma)
    alpha_weight_factor = (y_true * alpha + (1 - y_true) * (1 - alpha))

    sigmoid_focal_loss = modulating_factor * alpha_weight_factor * sigmoid_loss

    return sigmoid_focal_loss


def box_iou(b1, b2):
    """
    Return iou tensor
    """
    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def box_giou(b_true, b_pred):
    """
    Calculate GIoU loss on anchor boxes
    """
    b_true_xy = b_true[..., :2]
    b_true_wh = b_true[..., 2:4]
    b_true_wh_half = b_true_wh/2.
    b_true_mins = b_true_xy - b_true_wh_half
    b_true_maxes = b_true_xy + b_true_wh_half

    b_pred_xy = b_pred[..., :2]
    b_pred_wh = b_pred[..., 2:4]
    b_pred_wh_half = b_pred_wh/2.
    b_pred_mins = b_pred_xy - b_pred_wh_half
    b_pred_maxes = b_pred_xy + b_pred_wh_half

    intersect_mins = K.maximum(b_true_mins, b_pred_mins)
    intersect_maxes = K.minimum(b_true_maxes, b_pred_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b_true_area = b_true_wh[..., 0] * b_true_wh[..., 1]
    b_pred_area = b_pred_wh[..., 0] * b_pred_wh[..., 1]
    union_area = b_true_area + b_pred_area - intersect_area
    # calculate IoU, add epsilon in denominator to avoid dividing by 0
    iou = intersect_area / (union_area + K.epsilon())

    # get enclosed area
    enclose_mins = K.minimum(b_true_mins, b_pred_mins)
    enclose_maxes = K.maximum(b_true_maxes, b_pred_maxes)
    enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    # calculate GIoU, add epsilon in denominator to avoid dividing by 0
    giou = iou - 1.0 * (enclose_area - union_area) / (enclose_area + K.epsilon())
    giou = K.expand_dims(giou, -1)

    return giou


def box_diou(b_true, b_pred, use_ciou=False):
    """
    Calculate DIoU/CIoU loss on anchor boxes
    """
    b_true_xy = b_true[..., :2]
    b_true_wh = b_true[..., 2:4]
    b_true_wh_half = b_true_wh/2.
    b_true_mins = b_true_xy - b_true_wh_half
    b_true_maxes = b_true_xy + b_true_wh_half

    b_pred_xy = b_pred[..., :2]
    b_pred_wh = b_pred[..., 2:4]
    b_pred_wh_half = b_pred_wh/2.
    b_pred_mins = b_pred_xy - b_pred_wh_half
    b_pred_maxes = b_pred_xy + b_pred_wh_half

    intersect_mins = K.maximum(b_true_mins, b_pred_mins)
    intersect_maxes = K.minimum(b_true_maxes, b_pred_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b_true_area = b_true_wh[..., 0] * b_true_wh[..., 1]
    b_pred_area = b_pred_wh[..., 0] * b_pred_wh[..., 1]
    union_area = b_true_area + b_pred_area - intersect_area
    # calculate IoU, add epsilon in denominator to avoid dividing by 0
    iou = intersect_area / (union_area + K.epsilon())

    # box center distance
    center_distance = K.sum(K.square(b_true_xy - b_pred_xy), axis=-1)
    # get enclosed area
    enclose_mins = K.minimum(b_true_mins, b_pred_mins)
    enclose_maxes = K.maximum(b_true_maxes, b_pred_maxes)
    enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.0)
    # get enclosed diagonal distance
    enclose_diagonal = K.sum(K.square(enclose_wh), axis=-1)
    # calculate DIoU, add epsilon in denominator to avoid dividing by 0
    diou = iou - 1.0 * (center_distance) / (enclose_diagonal + K.epsilon())

    if use_ciou:
        box_true_w, box_true_h = b_true[..., 2] - b_true[..., 0], b_true[..., 3] - b_true[..., 1]
        box_pred_w, box_pred_h = b_pred[..., 2] - b_pred[..., 0], b_pred[..., 3] - b_pred[..., 1]

        atan = tf.atan(tf.math.divide_no_nan(box_true_w, box_true_h)) - tf.atan(tf.math.divide_no_nan(box_pred_w, box_pred_h))
        v = (atan * 2 / np.pi) ** 2
        alpha = tf.stop_gradient(tf.math.divide_no_nan(v, 1.0 - iou + v))
        
        diou = diou - alpha * v

    diou = K.expand_dims(diou, -1)
    return diou


def _smooth_labels(y_true, label_smoothing):
    label_smoothing = K.constant(label_smoothing, dtype=K.floatx())
    return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing


def yolo3_loss(args, anchors,
            num_classes, 
            anchor_masks='custom',
            num_layers=3,
            ignore_thresh=.5, 
            label_smoothing=0, 
            elim_grid_sense=False, 
            use_vf_loss=False, 
            use_focal_loss=True, 
            use_focal_obj_loss=True,
            use_giou_loss=False, 
            use_diou_loss=True,
            use_ciou_loss=False,
            focal_gamma=4.3):
    '''
    YOLOv3 loss function.
    '''
    yolo_outputs = args[:num_layers][::-1]
    y_true = args[num_layers:][::-1]

    if anchor_masks == 'custom':
        anchor_mask = [[9], [8,7], [0,1,2,3,4,5,6]]
    else:
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
    grid_size = 32
    scale_x_y = [1.05, 1.1, 1.2] if elim_grid_sense else [None, None, None]

    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * grid_size, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[i])[1:3], K.dtype(y_true[0])) for i in range(num_layers)]
    loss = 0
    total_location_loss = 0
    total_confidence_loss = 0
    total_class_loss = 0
    batch_size = K.shape(yolo_outputs[0])[0] # batch size, tensor
    batch_size_f = K.cast(batch_size, K.dtype(yolo_outputs[0]))

    for i in range(num_layers):
        object_mask = y_true[i][..., 4:5]
        true_class_probs = y_true[i][..., 5:]
        if label_smoothing:
            true_class_probs = _smooth_labels(true_class_probs, label_smoothing)
            true_objectness_probs = _smooth_labels(object_mask, label_smoothing)
        else:
            true_objectness_probs = object_mask

        grid, raw_pred, pred_xy, pred_wh = yolo3_decode(yolo_outputs[i],
             anchors[anchor_mask[i]], num_classes, input_shape, scale_x_y=scale_x_y[i], calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[i][..., :2]*grid_shapes[i][::-1] - grid
        raw_true_wh = K.log(y_true[i][..., 2:4] / anchors[anchor_mask[i]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[i][...,2:3]*y_true[i][...,3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[i][b,...,0:4], object_mask_bool[b,...,0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask
        _, ignore_mask = tf.while_loop(lambda b,*args: b<batch_size, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        if use_focal_obj_loss:
            # Focal loss for objectness confidence
            confidence_loss = sigmoid_focal_loss(true_objectness_probs, raw_pred[...,4:5], gamma=focal_gamma)
        else:
            confidence_loss = object_mask * K.binary_crossentropy(true_objectness_probs, raw_pred[...,4:5], from_logits=True)+ \
                (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask

        if use_focal_loss:
            # Focal loss for classification score
            class_loss = sigmoid_focal_loss(true_class_probs, raw_pred[...,5:], gamma=focal_gamma)
        elif use_vf_loss:
            class_loss = varifocal_loss(true_class_probs, raw_pred[...,5:], gamma=focal_gamma)
        else:
            # use sigmoid style classification output
            class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)


        if use_giou_loss:
            # Calculate GIoU loss as location loss
            raw_true_box = y_true[i][...,0:4]
            giou = box_giou(raw_true_box, pred_box)
            giou_loss = object_mask * box_loss_scale * (1 - giou)
            giou_loss = K.sum(giou_loss) / batch_size_f
            location_loss = giou_loss
        elif use_diou_loss or use_ciou_loss:
            # Calculate DIoU loss as location loss
            raw_true_box = y_true[i][...,0:4]
            diou = box_diou(raw_true_box, pred_box, use_ciou=use_ciou_loss)
            diou_loss = object_mask * box_loss_scale * (1 - diou)
            diou_loss = K.sum(diou_loss) / batch_size_f
            location_loss = diou_loss
        else:
            # Standard YOLOv3 location loss
            # K.binary_crossentropy is helpful to avoid exp overflow.
            xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
            wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
            xy_loss = K.sum(xy_loss) / batch_size_f
            wh_loss = K.sum(wh_loss) / batch_size_f
            location_loss = xy_loss + wh_loss

        # only involve class loss for multiple classes
        if num_classes == 1:
            class_loss = K.constant(0)
        else:
            class_loss = K.sum(class_loss) / batch_size_f
        confidence_loss = K.sum(confidence_loss) / batch_size_f
        loss += location_loss + confidence_loss + class_loss
        total_location_loss += location_loss
        total_confidence_loss += confidence_loss
        total_class_loss += class_loss

    # Fit for tf 2.0.0 loss shape
    loss = K.expand_dims(loss, axis=-1)

    return loss, total_location_loss, total_confidence_loss, total_class_loss