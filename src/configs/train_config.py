NUM_CLASSES = 3

IMG_SIZE = 128

SCORE_THRESHOLD = 0.01

WEIGHT_DECAY = 0

INITIAL_LR = 0.0013 

USE_COSINE_DECAY = False
USE_TENSORBOARD = True
USE_EARLY_STOPPING = True
USE_TELEGRAM_CALLBACK = False

LOAD_WEIGHTS = False
DARKNET_WEIGHTS = True

USE_CUSTOM_ANCHORS = False

DARKNET_WEIGHTS_PATH = '../checkpoints/'

BATCH_SIZE = 16
BUFFER_SIZE = 100
PREFETCH_SIZE = 2
MAX_NUM_BBOXES = 100

NUM_EPOCHS = 1000
ITER_PER_EPOCH = 70

loss_params = {
    'sensitivity_factor': 1.1,
    'iou_threshold': 0.5,
    'smooth_factor': 0.1,
    'focal_gamma': 4.3,
    'use_focal_loss': True,
    'use_focal_obj_loss': True,
    'use_vf_loss': False,
    'use_diou': True,
    'use_giou': False,
    'use_ciou': False,
    'elim_grid_sense': True
}
