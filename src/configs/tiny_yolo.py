import numpy as np

""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""

#anchor_dict = {
#    'anchors': np.array([(0.47,0.48), (0.53,1.05), (0.92,0.89), (0.97,0.53), (1.09,1.81), (2.05,1.15), (2.60,3.01), (5.93,4.51), (12.06,12.36)]),
#    'anchor_masks':  np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]]),
#}

anchor_dict = {
    'anchors': np.array([(0.03,0.06), (0.04,0.03), (0.06,0.05), (0.06,0.10), (0.11,0.18), (0.11,0.07), (0.24,0.14), (0.34,0.36), (0.81,0.77)]),
    'anchor_masks':  np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]]),
}

#Backbone
backbone = [
    (8, 3, 1, 'same', 'mish'),
    ["CSP", 1],
    (16, 1, 1, 'same', 'mish'),
    ["CSP", 2],
    (32, 1, 1, 'same', 'mish'),
    ["CSP", 4],
    (64, 1, 1, 'same', 'mish')
]

head = [
    'L',
    ["S", 32],# input_3 -> output_1
    (32, 3, 2, 'valid', 'leaky'),# input_3
    'M',
    ["C"], # concatanate with input_2
    (32, 1, 1, 'same', 'leaky'),
    (64, 3, 1, 'same', 'leaky'),
    (32, 1, 1, 'same', 'leaky'),
    (64, 3, 1, 'same', 'leaky'),
    (32, 1, 1, 'same', 'leaky'),
    ["S", 32],# output_2
    (64, 3, 2, 'valid', 'leaky'),
    'S',
    ["C"], # concatanate with input_1
    (64, 1, 1, 'same', 'leaky'),
    (128, 3, 1, 'same', 'leaky'),
    (64, 1, 1, 'same', 'leaky'),# output_2
    (128, 3, 1, 'same', 'leaky'),
    (64, 1, 1, 'same', 'leaky'),
    ["S", 64],# output_3
]