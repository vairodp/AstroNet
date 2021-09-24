import os
from anchors import YOLOV4_ANCHORS
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
from configs.train_config import MAX_NUM_BBOXES, NUM_CLASSES

from anchors import compute_normalized_anchors, YOLOV4_ANCHORS

MAX_BOXES = 100

SPLITS = {
    'train': 'train[:80%]',
    'validation': 'train[80%:90%]',
    'test': 'train[-10%:]'
}

class ConvoSKA:
    def __init__(self, mode='train'):
        self.mode = mode
        self.data_dir = "../data"
        self.download_dir = self.data_dir + "/raw"
        self.dataset = tfds.load('ska', split=SPLITS[mode],
                                data_dir=self.data_dir,
                                download_and_prepare_kwargs={'download_dir': self.download_dir})
        self.weights = self.get_class_weights()

    def get_class_weights(self):
        dataset = tfds.load('ska', split=SPLITS['train'],
                                data_dir=self.data_dir,
                                download_and_prepare_kwargs={'download_dir': self.download_dir})
        labels = dataset.map(self.create_mask)
        masks = []
        for mask in labels:
            masks.append(mask['label'])
        masks = np.ravel(masks)
        weights = compute_class_weight(class_weight='balanced', classes=np.unique(masks), y=masks)
        weights = weights.astype(np.float32)
        return weights

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

        ymins = np.rint(bbox[..., 0]*IMG_SIZE).astype(np.int32)
        xmins = np.rint(bbox[..., 1]*IMG_SIZE).astype(np.int32)
        ymaxs = np.rint(bbox[..., 2]*IMG_SIZE).astype(np.int32)
        xmaxs = np.rint(bbox[..., 3]*IMG_SIZE).astype(np.int32)
        #classes = np.zeros((MAX_BOXES, 3), dtype=np.float32) # classes one hot encoded + confidence
        #coords = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.float32) # sources + confidence
        mask = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
        mask.fill(3.0)
        for index, ymin, xmin, ymax, xmax, class_id in zip(tf.range(MAX_BOXES), ymins, xmins, ymaxs, xmaxs, label):
            #classes[index][class_id] = 1
            #coords[y_coord][x_coord] = 1
            mask[ymin:ymax, xmin:xmax] = class_id
        
        #return coords, classes
        return mask
        
    def get_dataset(self):
        data = self.dataset.map(self.map_features).shuffle(BUFFER_SIZE) 
        if self.mode == 'train':
            data = data.repeat()
        data = data.batch(BATCH_SIZE, drop_remainder=True) \
                    .prefetch(PREFETCH_SIZE)

        return data
    
    def augment_image_and_bbox(self, image, bbox):
        # bbox = [y1, x1, y2, x2] with values normalized in [0, 1]
        # we need to go from bottom-left and top-right corners to 
        # bottom-right and top-left corners
        bbox = bbox * IMG_SIZE
        ymin = bbox[..., 0]
        xmin = bbox[..., 1]
        ymax = bbox[..., 2]
        xmax = bbox[..., 3]
        bbox = [BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2) for y1, x1, y2, x2 in zip(ymin, xmin, ymax, xmax)]
        bbox = BoundingBoxesOnImage(bbox, shape=image.shape)
        #Apply augmenter to 50% of the images
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        seq = iaa.Sequential([
            iaa.OneOf([
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
			    iaa.Flipud(0.5)   # vertically flip 50% of all images
            ]),
            sometimes(iaa.Rot90((1,3))),
        ])
        image_aug, bbox_aug = seq(image=image, bounding_boxes=bbox)

        #plt.imshow(new_image)
        #plt.savefig(f'prova/img{datetime.now().strftime("%Y_%m_%d-%I-%M-%S_%p")}.png')

        bbox_aug = tf.convert_to_tensor([tf.concat([box.coords[0][1], box.coords[0][0], box.coords[1][1], box.coords[1][0]], axis=-0) for box in bbox_aug.items])
        #bbox_aug = bbox_to_xywh(bbox_aug) / IMG_SIZE
        bbox_aug = bbox_aug / IMG_SIZE
        return image_aug, bbox_aug
    
    def aug_img(self, img):
        p_brightness = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        p_constrast = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        p_hue = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        p_sat = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        if p_brightness >= 0.3:
            img = tf.image.random_brightness(img, max_delta=0.25)
        if p_constrast >= 0.3:
            img = tf.image.random_contrast(img, lower=0.4, upper=1.3)
        if p_hue >= 0.3:
            img = tf.image.random_hue(img, max_delta=0.2)
        if p_sat >= 0.3:
            img = tf.image.random_saturation(img, lower=0, upper=4)
        #plt.imshow(img)
        #plt.savefig(f'prova/img{datetime.now().strftime("%Y_%m_%d-%I-%M-%S_%p")}.png')
        return img

    def add_sample_weights(self, label):
        # The weights for each class, with the constraint that:
        #     sum(class_weights) == 1.0
        #class_weights = tf.numpy_function(self.get_class_weights, inp=[], Tout=tf.float32)
        class_weights = tf.constant(self.weights)
        class_weights = class_weights/tf.reduce_sum(class_weights)

        # Create an image of `sample_weights` by using the label at each pixel as an 
        # index into the `class weights` .
        sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))

        return sample_weights
    
    def create_mask(self, feature):
        # limit the number of bounding box and label
        bbox = feature["objects"]["bbox"]

        #coords, classes = tf.numpy_function(self.map_label, inp=[bbox, feature['objects']['label']], Tout=[tf.float32, tf.float32])
        mask = tf.numpy_function(self.map_label, inp=[bbox, feature['objects']['label']], Tout=tf.float32)

        feature_dict = {
            "label": mask
        }

        return feature_dict

    def map_features(self,feature):
        
        image = feature["image"]

        # limit the number of bounding box and label
        bbox = feature["objects"]["bbox"]

        #if self.mode == 'train':
            #image, bbox = tf.numpy_function(self.augment_image_and_bbox, inp=[image, bbox], Tout=[tf.uint8, tf.float32])
            #image = self.aug_img(image)]
            

        num_of_bbox = tf.shape(bbox)[0]

        #coords, classes = tf.numpy_function(self.map_label, inp=[bbox, feature['objects']['label']], Tout=[tf.float32, tf.float32])
        mask = tf.numpy_function(self.map_label, inp=[bbox, feature['objects']['label']], Tout=tf.float32)

        sample_weights = self.add_sample_weights(mask)

        #coords.set_shape([IMG_SIZE, IMG_SIZE, 1])
        #classes.set_shape([MAX_NUM_BBOXES, NUM_CLASSES])
        mask.set_shape([IMG_SIZE, IMG_SIZE, 1])

        image = (image - tf.math.reduce_mean(image)) / tf.math.reduce_std(image)
        image = tf.expand_dims(image, axis=-1)


        bbox = self.concat_class(bbox, feature['objects']['label'])
        bbox = self.pad_bbox(bbox)


        feature_dict = {
            "image": image,
            "label": mask,
            "weights": sample_weights,
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