import os

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras import backend
import math

import datasets.ska
from configs.yolo_v4 import IMG_SIZE, BUFFER_SIZE, BATCH_SIZE, PREFETCH_SIZE
from configs.yolo_v4 import MAX_NUM_BBOXES, ANCHORS, ANCHORS_MASKS, NUM_CLASSES

SPLITS = {
    'train': 'train[:80%]',
    'test': 'train[-20%:]'
}


class SKADataset:
    def __init__(self, mode='train'):
        self.mode = mode
        data_dir = "../data"
        download_dir = data_dir + "\\raw"
        self.dataset = tfds.load('ska', split=SPLITS[mode], 
                                shuffle_files=True, 
                                data_dir=data_dir,
                                #download=False,
                                download_and_prepare_kwargs={'download_dir': download_dir})
    
    def transform_bbox(self, bbox):
        # bbox = [ymin, xmin, ymax, xmax] ---> [x, y, width, height]
        ymin, xmin = bbox[..., 0:2]
        ymax, xmax = bbox[..., 2:4]
        x = (xmin + xmax) / 2
        y = (ymin + ymax) / 2
        height = ymax - ymin
        width = xmax - xmin

        return np.concatenate([x,y,width, height], axis=-1)
    
    def map_label(self, bbox, label):
        # bbox = [x, y, width, height] values in [0, IMG_SIZE]
        bbox = bbox / IMG_SIZE
        bbox = np.clip(bbox, a_min=0.0, a_max=1 - backend.epsilon())

        grid_size = math.ceil(IMG_SIZE / 32)
        
        # find best anchor
        anchor_area = ANCHORS[..., 0] * ANCHORS[..., 1]
        box_wh = bbox[..., 2:4]
        box_wh = np.tile(np.expand_dims(box_wh, -2), (1, 1, ANCHORS.shape[0], 1))
        box_area = box_wh[..., 0] * box_wh[..., 1]
        intersection = np.minimum(box_wh[..., 0], ANCHORS[..., 0]) * np.minimum(box_wh[..., 1],
                                                                                     ANCHORS[..., 1])
        iou = intersection / (box_area + anchor_area - intersection)
        anchor_idx = np.argmax(iou, axis=-1)  # shape = (1, n)

        label_small = self.yolo_label(bbox, label, ANCHORS_MASKS[0], anchor_idx, grid_size)
        label_medium = self.yolo_label(bbox, label, ANCHORS_MASKS[1], anchor_idx, grid_size * 2)
        label_large = self.yolo_label(bbox, label, ANCHORS_MASKS[2], anchor_idx, grid_size * 4)

        return label_small, label_medium, label_large

    def yolo_label(self, bbox, label, anchor_masks, anchor_idx, grid_size):
        
        # grids.shape: (grid_size, grid_size, 3, NUMCLASSES + 5)
        grids = np.zeros((grid_size, grid_size, anchor_masks.shape[0], (5 + NUM_CLASSES)), dtype=np.float32)

        for box, class_id, anchor_id in zip(bbox, label, anchor_idx):
            if anchor_id in anchor_masks:
                box = box[0:4]
                box_xy = box[0:2]

                grid_xy = (box_xy // (1 / grid_size)).astype(int)

                box_index = np.where(anchor_masks == anchor_id)[0][0]

                grid_array = np.zeros((5 + NUM_CLASSES))
                grid_array[0:5] = np.append(box, [1])
                class_index = int(5 + class_id)
                grid_array[class_index] = 1

                # grid[y][x][anchor] = [tx, ty, bw, bh, obj, ...class_id]
                grids[grid_xy[1]][grid_xy[0]][box_index] = grid_array

        return grids
        
    def get_dataset(self):
        data = self.dataset.map(self.map_features) \
            .shuffle(BUFFER_SIZE) \
            .batch(BATCH_SIZE) \
            .prefetch(PREFETCH_SIZE)

        return data

    def map_features(self,feature):
        
        image = feature["image"]

        # limit the number of bounding box and label
        bbox = feature["objects"]["bbox"]
        bbox = bbox[:MAX_NUM_BBOXES]

        num_of_bbox = tf.shape(bbox)[0]
        label = tf.zeros(num_of_bbox, dtype=tf.int32)
        label = label[:MAX_NUM_BBOXES]

        bbox = tf.numpy_function(self.transform_bbox, inp=[bbox], Tout=tf.float32)

        label_small, label_medium, label_large = tf.numpy_function(self.map_label, inp=[bbox, label],
                                                                   Tout=[tf.float32, tf.float32, tf.float32])
        # normalize to [-1, 1]
        image = image / 127.5 - 1

        bbox = self.concat_class(bbox, label)
        bbox = self.pad_bbox(bbox)

        feature_dict = {
            "image": image,
            "label": (label_small, label_medium, label_large),
            "bbox": bbox,
            "num_of_bbox": num_of_bbox
        }

        return feature_dict

    def pad_bbox(self,bbox):
        # bbox.shape = (n, 5)
        bbox = tf.expand_dims(bbox, axis=-1)  # bbox.shape = (n, 5, 1)
        bbox = tf.image.pad_to_bounding_box(bbox, 0, 0, self.max_bbox_size, tf.shape(bbox)[1])

        bbox = tf.squeeze(bbox)

        return bbox
    
    def concat_class(self, bbox, label):
        # bbox.shape = (n, 4)
        # label.shape = (n)
        label = tf.cast(tf.reshape(label, (-1, 1)), tf.float32)
        label = tf.concat([bbox, label], axis=-1)

        return label