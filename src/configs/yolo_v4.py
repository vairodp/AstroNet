import numpy as np

""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""

NUM_CLASSES = 3

IMG_SIZE = 128

ANCHORS = np.array([(12,16),  (19,36),  (40,28),  (36,75),  (76,55),  
                    (72,146),  (142,110),  (192,243),  (459,401)],
                    np.float32) / IMG_SIZE

ANCHORS_MASKS =  np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

BATCH_SIZE = 32
BUFFER_SIZE = 100
PREFETCH_SIZE = 2
MAX_NUM_BBOXES = 300

#Backbone
cspdarknet53 = [
    (32, 3, 1, 'same', 'mish'),
    ["CSP", 1],
    (64, 3, 1, 'same', 'mish'),
    ["CSP", 2],
    (128, 1, 1, 'same', 'mish'),
    ["CSP", 8],
    (256, 1, 1, 'same', 'mish'),
    ["CSP", 8],
    (512, 1, 1, 'same', 'mish'),
    ["CSP", 4],
    (1024, 1, 1, 'same', 'mish'),
]

#Neck
panet = [
    'S', # output small
    (512, 1, 1, 'same', 'leaky'),
    (1024, 3, 1, 'same', 'leaky'),
    (512, 1, 1, 'same', 'leaky'),
    ["SPP"],
    (512, 1, 1, 'same', 'leaky'),
    (1024, 3, 1, 'same', 'leaky'),
    (512, 1, 1, 'same', 'leaky'),
    ["U", 256],
    'M', # output medium
    ["C", 256, 1, 1, 'same', 'leaky'],
    (256, 1, 1, 'same', 'mish'),
    (512, 3, 1, 'same', 'mish'),
    (256, 1, 1, 'same', 'mish'),
    (512, 3, 1, 'same', 'mish'),
    (256, 1, 1, 'same', 'mish'),
    ["U", 128],
    'L', # output large
    ["C", 128, 1, 1, 'same', 'leaky'],
    (128, 1, 1, 'same', 'leaky'),
    (256, 3, 1, 'same', 'mish'),
    (128, 1, 1, 'same', 'leaky'),
    (256, 3, 1, 'same', 'mish'),
    (128, 1, 1, 'same', 'leaky'),
    ['A', 3]
]

head = [
    'L',
    ["S", 256],# input_3 -> output_1
    (256, 3, 2, 'valid', 'leaky'),# input_3
    'M',
    ["C"], # concatanate with input_2
    (256, 1, 1, 'same', 'leaky'),
    (512, 3, 1, 'same', 'leaky'),
    (256, 1, 1, 'same', 'leaky'),
    (512, 3, 1, 'same', 'leaky'),
    (256, 1, 1, 'same', 'leaky'),
    ["S", 256],# output_2
    (512, 3, 2, 'valid', 'leaky'),
    'S',
    ["C"], # concatanate with input_1
    (512, 1, 1, 'same', 'leaky'),
    (1024, 3, 1, 'same', 'leaky'),
    (512, 1, 1, 'same', 'leaky'),# output_2
    (1024, 3, 1, 'same', 'leaky'),
    (512, 1, 1, 'same', 'leaky'),
    ["S", 512],# output_3
]

loss_params = {
    'sensitivity_factor': 1.1,
    'iou_threshold': 0.45
}