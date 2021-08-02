from typing import Counter
import numpy as np
import tensorflow as tf
from yolo_v4 import YoloV4, CSPBlock
from configs.yolo_v4 import NUM_CLASSES, IMG_SIZE, BATCH_SIZE

PRETRAINED_WEIGHTS = '../checkpoints/yolov4.weights'
INPUT_SHAPE = (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3)

def get_backbone_layer_list(yolo):
    darknet = yolo.backbone
    layers = []
    for submodel in darknet:
        if 'cnn_block' in submodel.name:
            layers.extend(submodel.layers)
        else:
            for elem in submodel.layers:
                if 'cnn_block' in elem.name:
                    layers.extend(elem.layers)
                else:
                    for subelem in elem.layers:
                        if 'cnn_block' in subelem.name:
                            layers.extend(subelem.layers)
                        else:
                            # It's a residual block
                            for seq in subelem.layers:
                                for submodel in seq.layers:
                                    layers.extend(submodel.layers)
    return layers

def get_neck_layer_list(yolo):
    neck = yolo.neck
    layers = []
    for submodel in neck.layers:
        if 'cnn_block' in submodel.name:
            print(submodel)
            layers.extend(submodel.layers)
        elif 'spp_block' in submodel.name or \
                'spatial_attention' in submodel.name:
            for block in submodel.layers:
                if 'cnn_block' in block.name:
                    layers.extend(block.layers)
                else:
                    layers.append(block)
        elif 'up_sampling' in submodel.name:
            seq = submodel.layers[0]
            for block in seq.layers:
                if 'cnn_block' in block.name:
                    layers.extend(block.layers)
                else:
                    layers.append(block)
    return layers

def get_head_layer_list(yolo):
    head = yolo.head
    layers = []
    for submodel in head.layers:
        if 'cnn_block' in submodel.name:
            layers.extend(submodel.layers)
        elif 'scale_prediction' in submodel.name:
            seq = submodel.layers[0]
            for layer in seq.layers:
                layers.extend(layer.layers)
    return layers

def load_darknet_weights_in_yolo(yolo, trainable=False, darknet_weights_path=PRETRAINED_WEIGHTS):
    """
    Load the yolov4.weights file into our YOLOv4 model.
    Args:
        yolo_model (tf.keras.Model): YOLOv4 model
        darknet_weights_path (str): Path to the yolov4.weights darknet file
    Returns:
        tf.keras.Model: YOLOv4 model with Darknet weights loaded.
    """
    model_layers = (get_backbone_layer_list(yolo) + 
                    get_neck_layer_list(yolo) +
                    get_head_layer_list(yolo))
    print(len(model_layers))
    print(Counter([type(layer) for layer in model_layers]))

    # Get all trainable layers: convolutions and batch normalization
    conv_layers = [layer for layer in model_layers if "conv2d" in layer.name]
    batch_norm_layers = [
        layer for layer in model_layers if "batch_normalization" in layer.name
    ]

    # Sort them by order of appearance.
    # The first apparition does not have an index, hence the [[0]] trick to avoid an error
    conv_layers = [conv_layers[0]] + sorted(
        conv_layers[1:], key=lambda x: int(x.name[7:])
    )
    batch_norm_layers = [batch_norm_layers[0]] + sorted(
        batch_norm_layers[1:], key=lambda x: int(x.name[20:])
    )

    #print([l.name for l in conv_layers])

    # Open darknet file and read headers
    darknet_weight_file = open(darknet_weights_path, "rb")
    # First elements of file are major, minor, revision, seen, _
    _ = np.fromfile(darknet_weight_file, dtype=np.int32, count=5)

    # Keep an index of which batch norm should be considered.
    # If batch norm is used with a convolution (meaning conv has no bias), the index is incremented
    # Otherwise (conv has a bias), index is kept still. 
    # TODO: NOPE! In out case, we have a batch normalization for every conv, 
    # even if they are not used, thus this index should always be increased
    current_matching_batch_norm_index = 0

    for layer in conv_layers:
        print('LAYER: ', layer.name)
        kernel_size = layer.kernel_size
        input_filters = layer.input_shape[-1]
        filters = layer.filters
        use_bias = layer.bias is not None


        if use_bias:
            # Read bias weight
            conv_bias = np.fromfile(
                darknet_weight_file, dtype=np.float32, count=filters
            )
            print('USED BIAS')
        else:
            # Read batch norm
            # Reorder from darknet (beta, gamma, mean, var) to TF (gamma, beta, mean, var)
            batch_norm_weights = np.fromfile(
                darknet_weight_file, dtype=np.float32, count=4 * filters
            ).reshape((4, filters))[[1, 0, 2, 3]]
            print('USED BATCH NORM')

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
    
    conv_layers.extend(batch_norm_layers)
    for layer in conv_layers:
        layer.trainable = trainable

    #  Check if we read the entire darknet file.
    remaining_chars = len(darknet_weight_file.read())
    darknet_weight_file.close()
    print(remaining_chars)

    return yolo

#yolo = YoloV4(num_classes=NUM_CLASSES)
