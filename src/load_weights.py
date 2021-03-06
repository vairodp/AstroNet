import os
import numpy as np

from yolo_v4 import YOLOv4
from anchors import YOLOV4_ANCHORS
from configs.train_config import IMG_SIZE, BATCH_SIZE

PRETRAINED_WEIGHTS = '../checkpoints/yolov4.weights'
INPUT_SHAPE = (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 1)

def load_darknet_weights_in_yolo(yolo, darknet_weights_path=PRETRAINED_WEIGHTS):
    model_layers = (
        yolo.model.get_layer("CSPDarknet53").layers
        + yolo.model.get_layer("YOLOv4_neck").layers
        + yolo.model.get_layer("YOLOv3_head").layers
    )

    # Get all trainable layers: convolutions and batch normalization
    conv_layers = [layer for layer in model_layers if "conv2d" in layer.name]
    batch_norm_layers = [
        layer for layer in model_layers if "batch_normalization" in layer.name
    ]

    # Sort them by order of appearance.
    # The first apparition does not have an index, hence the [[0]] trick to avoid an error
    conv_layers = [conv_layers[0]] + sorted(
        conv_layers[1:], key=lambda x: int(x.name[7:] if 'pred' not in x.name else int(x.name[15:]))
    )
    batch_norm_layers = [batch_norm_layers[0]] + sorted(
        batch_norm_layers[1:], key=lambda x: int(x.name[20:])
    )

    # Open darknet file and read headers
    darknet_weight_file = open(darknet_weights_path, "rb")
    # First elements of file are major, minor, revision, seen, _
    _ = np.fromfile(darknet_weight_file, dtype=np.int32, count=5)

    # Keep an index of which batch norm should be considered.
    # If batch norm is used with a convolution (meaning conv has no bias), the index is incremented
    # Otherwise (conv has a bias), index is kept still.
    current_matching_batch_norm_index = 0

    for layer in conv_layers:
        kernel_size = layer.kernel_size
        input_filters = layer.input_shape[-1]
        filters = layer.filters
        use_bias = layer.bias is not None

        if use_bias:
            # Read bias weight
            conv_bias = np.fromfile(
                darknet_weight_file, dtype=np.float32, count=filters
            )
        else:
            # Read batch norm
            # Reorder from darknet (beta, gamma, mean, var) to TF (gamma, beta, mean, var)
            batch_norm_weights = np.fromfile(
                darknet_weight_file, dtype=np.float32, count=4 * filters
            ).reshape((4, filters))[[1, 0, 2, 3]]

        # Read kernel weights
        # Reorder from darknet (filters, input_filters, kernel_size[0], kernel_size[1]) to
        # TF (kernel_size[0], kernel_size[1], input_filters, filters)
        conv_size = kernel_size[0] * kernel_size[1] * input_filters * filters
        conv_weights = (
            np.fromfile(darknet_weight_file, dtype=np.float32, count=conv_size)
            .reshape((filters, input_filters, kernel_size[0], kernel_size[1]))
            .transpose([2, 3, 1, 0])
        )

        if use_bias:
            # load conv weights and bias, increase batch_norm offset
            layer.set_weights([conv_weights, conv_bias])
        else:
            # load conv weights, load batch norm weights
            layer.set_weights([conv_weights])
            batch_norm_layers[current_matching_batch_norm_index].set_weights(
                batch_norm_weights
            )
            current_matching_batch_norm_index += 1

    #  Check if we read the entire darknet file.
    remaining_chars = len(darknet_weight_file.read())
    darknet_weight_file.close()
    assert remaining_chars == 0

    return yolo    

def load_weights(yolo, folder_path):
    if 'yolov4.h5' not in os.listdir(folder_path):
        yolo_coco = YOLOv4(input_shape=(128,128,3), num_classes=80, anchors=YOLOV4_ANCHORS)
        yolo_coco = load_darknet_weights_in_yolo(yolo_coco)
        yolo_coco.model.save_weights(folder_path + 'yolov4.h5')
        yolo.model.load_weights(folder_path + 'yolov4.h5', by_name=True, skip_mismatch=True)
    else:
        yolo.model.load_weights(folder_path + 'yolov4.h5', by_name=True, skip_mismatch=True)
    return yolo