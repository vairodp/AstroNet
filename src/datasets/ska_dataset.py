import os

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras import backend
from sklearn.utils.class_weight import compute_class_weight
import math
from utils import bbox_to_x1y1x2y2, bbox_to_xywh

import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

import matplotlib.pyplot as plt
from datetime import datetime

import datasets.ska
from configs.train_config import IMG_SIZE, BUFFER_SIZE, BATCH_SIZE, PREFETCH_SIZE
from configs.train_config import MAX_NUM_BBOXES, NUM_CLASSES, get_anchors

SPLITS = {
    'train': 'train[:80%]',
    'validation': 'train[80%:90%]',
    'test': 'train[-10%:]'
}


class SKADataset:
    def __init__(self, mode='train', grid_size=32, anchors=get_anchors()['anchors'], anchor_masks=get_anchors()['anchor_masks']):
        self.mode = mode
        data_dir = "../data"
        download_dir = data_dir + "/raw"
        self.grid_size = grid_size
        self.dataset = tfds.load('ska', split=SPLITS[mode], 
                                shuffle_files=True, 
                                data_dir=data_dir,
                                #download=False,
                                download_and_prepare_kwargs={'download_dir': download_dir})
        self.anchors=anchors
        self.anchor_masks = anchor_masks
        #self.mean, self.std = self._get_mean_and_std()
    
    def get_class_weights(self):
        labels = self.dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x['objects']['label']))
        labels = list(labels.as_numpy_iterator())
        label_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
        return label_weights
    
    def _get_mean_and_std(self):
        images = self.dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x['image']))
        images = list(images.as_numpy_iterator())
        mean = np.mean(images)
        std = np.std(images)
        return mean, std

    def transform_bbox(self, bbox):
        # bbox = [ymin, xmin, ymax, xmax] ---> [x, y, width, height]
        ymin = bbox[..., 0]
        xmin = bbox[..., 1]
        ymax = bbox[..., 2]
        xmax = bbox[..., 3]
        x = (xmin + xmax) / 2
        y = (ymin + ymax) / 2
        height = ymax - ymin
        width = xmax - xmin
        
        return np.vstack([x,y,width, height]).T
    
    def map_label(self, bbox, label):
        # bbox = [x, y, width, height] values in [0, IMG_SIZE]
        bbox = np.clip(bbox, a_min=0.0, a_max=1 - backend.epsilon())

        grid_size = math.ceil(IMG_SIZE / self.grid_size)
        
        # find best anchor
        anchor_area = self.anchors[..., 0] * self.anchors[..., 1]

        box_wh = bbox[..., 2:4]
        box_wh = np.tile(np.expand_dims(box_wh, -2), (1, 1, self.anchors.shape[0], 1))
        box_area = box_wh[..., 0] * box_wh[..., 1]
        intersection = np.minimum(box_wh[..., 0], self.anchors[..., 0]) * np.minimum(box_wh[..., 1],
                                                                                     self.anchors[..., 1])
        iou = intersection / (box_area + anchor_area - intersection)
        
        anchor_idx = np.argmax(iou, axis=-1).reshape((-1))  # shape = (1, n) --> (n,)

        label_small = self.yolo_label(bbox, label, self.anchor_masks[0], anchor_idx, grid_size)
        label_medium = self.yolo_label(bbox, label, self.anchor_masks[1], anchor_idx, grid_size * 2)
        label_large = self.yolo_label(bbox, label, self.anchor_masks[2], anchor_idx, grid_size * 4)

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
        data = self.dataset.map(self.map_features).shuffle(BUFFER_SIZE) 
        if self.mode == 'train':
            data = data.repeat()
        data = data.batch(BATCH_SIZE, drop_remainder=True) \
                    .prefetch(PREFETCH_SIZE)

        return data
    
    def augment_image_and_bbox(self, image, bbox):
        # bbox = [x, y, w, h] with values normalized in [0, 1]
        bbox = bbox_to_x1y1x2y2(bbox)
        bbox = bbox * IMG_SIZE
        bbox = [BoundingBox(*box_coords) for box_coords in bbox]
        bbox = BoundingBoxesOnImage(bbox, shape=image.shape)

        #Apply augmenter to 50% of the images
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        seq = iaa.Sequential([
            iaa.Fliplr(0.3),  # horizontally flip 30% of all images
			iaa.Flipud(0.3),  # vertically flip 30% of all images
            sometimes(iaa.Rot90((1,3))),
            sometimes(iaa.TranslateX(px=(-20, 20))),
            sometimes(iaa.TranslateX(px=(-20, 20))),
            sometimes(iaa.Affine(scale=2.0)),
            sometimes(iaa.OneOf([
                iaa.RemoveSaturation(1.0),
                iaa.AddToHueAndSaturation((-50, 50), per_channel=True)
            ]))
        ])
        image_aug, bbox_aug = seq(image=image, bounding_boxes=bbox)
        #plt.imshow(image_aug)
        #plt.savefig(f'prova/img{datetime.now().strftime("%Y_%m_%d-%I-%M-%S_%p")}.png')
        bbox_aug = tf.convert_to_tensor([tf.concat([box.coords[0], box.coords[1]], axis=-1) for box in bbox_aug.items])
        bbox_aug = bbox_to_xywh(bbox_aug) / IMG_SIZE
        return image_aug, bbox_aug

    def map_features(self,feature):
        
        image = feature["image"]

        # limit the number of bounding box and label
        bbox = feature["objects"]["bbox"]
        bbox = bbox[:MAX_NUM_BBOXES]

        if self.mode == 'train':
            image, bbox = tf.numpy_function(self.augment_image_and_bbox, inp=[image, bbox], Tout=[np.uint8, tf.float32])

        num_of_bbox = tf.shape(bbox)[0]
        label = tf.zeros(num_of_bbox, dtype=tf.int32)
        label = label[:MAX_NUM_BBOXES]

        bbox = tf.numpy_function(self.transform_bbox, inp=[bbox], Tout=tf.float32)

        label_small, label_medium, label_large = tf.numpy_function(self.map_label, inp=[bbox, label],
                                                                   Tout=[tf.float32, tf.float32, tf.float32])
        # normalize to [-1, 1]
        image = tf.cast(image, tf.float32)
        image = image / 127.5 - 1.0
        #image = (image / self.mean) - self.std

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
        bbox = tf.image.pad_to_bounding_box(bbox, 0, 0, MAX_NUM_BBOXES, tf.shape(bbox)[1])

        bbox = tf.squeeze(bbox)

        return bbox
    
    def concat_class(self, bbox, label):
        # bbox.shape = (n, 4)
        # label.shape = (n)
        label = tf.cast(tf.reshape(label, (-1, 1)), tf.float32)
        label = tf.concat([bbox, label], axis=-1)

        return label