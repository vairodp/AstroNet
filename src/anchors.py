"""
Contains original YOLO anchors and associated utils
"""
import numpy as np
from configs.train_config import IMG_SIZE

YOLOV4_ANCHORS = [
    np.array([(12, 16), (19, 36), (40, 28)], np.float32),
    np.array([(36, 75), (76, 55), (72, 146)], np.float32),
    np.array([(142, 110), (192, 243), (459, 401)], np.float32),
]

CUSTOM_ANCHORS = [
    np.array([(4,7), (4,3), (6,6), (8,11), (10,7), (13, 21), (18, 9)]),
    np.array([(33, 22), (52, 55)]),
    np.array([(100, 110)])
]

CUSTOM_ANCHORS_TINY = [
    np.array([(4,7), (4,3), (6,6), (8,11), (10,7), (13, 21), (18, 9)])
]

YOLOLITE_ANCHORS = [
    np.array([(10, 14), (23, 27), (37, 58)], np.float32),
    np.array([(81, 82), (135, 169), (344, 319)], np.float32),
]


def resize_achors(base_anchors, target_shape=IMG_SIZE, base_shape=416):
    """
    Original anchor size is clustered for the COCO dataset with input shape
    (416, 416). To get better results we should resize it to our train input
    images' size.
    """
    return [np.around(anchor*target_shape/base_shape) for anchor in base_anchors]

def compute_normalized_anchors(anchors, input_shape):
    """
    Compute anchors resizing based on the architecture input shapes
    Args:
        anchors (List[numpy.array[int, 2]]): List of 3 numpy arrays containing the anchor sizes used for
            each stage. The first and second columns of the numpy arrays respectively contain the anchors width and
            height.
        input_shape (Tuple[int]): Input shape of the Network

    Returns:
        (List[numpy.array[int, 2]]): anchors resized based on the input shape of the Network.
    """
    height, width = input_shape[:2]
    if anchors[-1][-1][1] <= IMG_SIZE:
        anchors = resize_achors(anchors)
    return [anchor / np.array([width, height]) for anchor in anchors]

compute_normalized_anchors(YOLOV4_ANCHORS, input_shape=(IMG_SIZE, IMG_SIZE, 3))