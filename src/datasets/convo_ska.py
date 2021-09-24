import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import backend

from configs.train_config import IMG_SIZE, BUFFER_SIZE, BATCH_SIZE, PREFETCH_SIZE, MAX_NUM_BBOXES
from sklearn.utils.class_weight import compute_class_weight

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
    
    def map_label(self, bbox, label):
        # bbox = [ymin, xmin, ymax, xmax] values in [0, 1]
        bbox = np.clip(bbox, a_min=0.0, a_max=1 - backend.epsilon())

        ymins = np.rint(bbox[..., 0]*IMG_SIZE).astype(np.int32)
        xmins = np.rint(bbox[..., 1]*IMG_SIZE).astype(np.int32)
        ymaxs = np.rint(bbox[..., 2]*IMG_SIZE).astype(np.int32)
        xmaxs = np.rint(bbox[..., 3]*IMG_SIZE).astype(np.int32)
        mask = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
        mask.fill(3.0)
        for ymin, xmin, ymax, xmax, class_id in zip(ymins, xmins, ymaxs, xmaxs, label):
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
    

    def add_sample_weights(self, label):
        # The weights for each class, with the constraint that:
        #     sum(class_weights) == 1.0
        class_weights = tf.constant(self.weights)
        class_weights = class_weights/tf.reduce_sum(class_weights)

        # Create an image of `sample_weights` by using the label at each pixel as an 
        # index into the `class weights` .
        sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))

        return sample_weights
    
    def create_mask(self, feature):
        # limit the number of bounding box and label
        bbox = feature["objects"]["bbox"]

        mask = tf.numpy_function(self.map_label, inp=[bbox, feature['objects']['label']], Tout=tf.float32)

        feature_dict = {
            "label": mask
        }

        return feature_dict

    def map_features(self,feature):
        
        image = feature["image"]

        # limit the number of bounding box and label
        bbox = feature["objects"]["bbox"]

        num_of_bbox = tf.shape(bbox)[0]

        mask = tf.numpy_function(self.map_label, inp=[bbox, feature['objects']['label']], Tout=tf.float32)

        sample_weights = self.add_sample_weights(mask)


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