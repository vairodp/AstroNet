""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""

NUM_CLASSES = 3

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

head = []