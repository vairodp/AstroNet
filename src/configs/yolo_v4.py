import numpy as np
import tensorflow as tf

""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""
#anchor_dict = {
#    'anchors': np.array([(0.12,0.23), (0.16,0.11), (0.22,0.21), (0.24,0.40), (0.41,0.25), (0.44,0.73), (0.77,0.42), (1.20,1.07), (2.90,2.86)]),
#    'anchor_masks':  np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
#}


anchor_dict = {
    'anchors': np.array([(0.03,0.06), (0.04,0.03), (0.06,0.05), (0.06,0.10), (0.11,0.18), (0.11,0.07), (0.24,0.14), (0.34,0.36), (0.81,0.77)]),
    'anchor_masks':  np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]]),
}

#Backbone
cspdarknet53 = [
    (32, 3, 1, 'same', 'mish'),
    ["CSP", 1], #Also, this may not be completely correct, as we only ever use half_filters in one conv and not in all the others, everything else is good though
    (64, 1, 1, 'same', 'mish'), #According to the pattern I saw on sicara/tf2-yolov4, this has both kernel size and strides = 1
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
    (512, 1, 1, 'same', 'leaky'), # IT seems like this conv is already included in the SPP block
    ["SPP"],
    (512, 1, 1, 'same', 'leaky'),
    (1024, 3, 1, 'same', 'leaky'),
    (512, 1, 1, 'same', 'leaky'), # It seems like a conv ["C", 256, 1, 1, 'same', 'leaky'] is also done on 'S', before the upsampling
    ["U", 256], # (1) Check the upsampling, I think the convolution should be done before and not after upsampling2D
    'M', # output medium
    ["C", 256, 1, 1, 'same', 'leaky'], 
    (256, 1, 1, 'same', 'mish'),
    (512, 3, 1, 'same', 'mish'), #ALso these activation functions seem to be all 'leaky'
    (256, 1, 1, 'same', 'mish'),
    (512, 3, 1, 'same', 'mish'),
    (256, 1, 1, 'same', 'mish'), # It seems like a conv ["C", 128, 1, 1, 'same', 'leaky'] is also done after this one, before the updampling
    ["U", 128], # Same as (1)
    'L', # output large
    ["C", 128, 1, 1, 'same', 'leaky'], 
    (128, 1, 1, 'same', 'leaky'),
    (256, 3, 1, 'same', 'mish'),
    (128, 1, 1, 'same', 'leaky'),
    (256, 3, 1, 'same', 'mish'),
    (128, 1, 1, 'same', 'leaky'),
    ['A', 3] # These seem not to be there (??)
]


#Head seems good
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
