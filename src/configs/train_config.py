from configs.yolo_v4 import anchor_dict as anchors_yolo
from configs.small_yolo import anchor_dict as anchors_small

NUM_CLASSES = 3

IMG_SIZE = 128

SCORE_THRESHOLD = 0.5

BATCH_SIZE = 16
BUFFER_SIZE = 100
PREFETCH_SIZE = 2
MAX_NUM_BBOXES = 30

NUM_EPOCHS = 250
ITER_PER_EPOCH = 1000

loss_params = {
    'sensitivity_factor': 1.1,
    'iou_threshold': 0.55,
    'smooth_factor': 0.1
}

def get_anchors(model='yolo'):
    anchors = anchors_yolo
    if model == 'small_yolo':
        anchors = anchors_small
    return anchors