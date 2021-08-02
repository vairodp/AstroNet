import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np

import datasets
#from load_weights import load_darknet_weights_in_yolo
import tensorflow as tf
from yolo_v4 import YoloV4
from loss import YoloLoss
from datasets.ska_dataset import SKADataset
from configs.yolo_v4 import ANCHORS, ANCHORS_MASKS, NUM_CLASSES, loss_params

import tensorflow_datasets as tfds

#Loading Bar
#from tqdm import tqdm

checkpoint_filepath = '../checkpoints/model.{epoch:02d}-{loss:.2f}.h5'

dataset_train = SKADataset(mode='train')
class_weights = dataset_train.get_class_weights()
dataset_train = dataset_train.get_dataset()
#class_weights = None

yolo = YoloV4(num_classes=NUM_CLASSES)

yolo.summary(line_length=160)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.003, clipvalue=1.0)
yolo_loss = [YoloLoss(anchors=ANCHORS[mask],
                    class_weights=class_weights,
                    iou_threshold=loss_params['iou_threshold'], 
                    smooth_factor=loss_params['smooth_factor'], 
                    use_ciou=True) for mask in ANCHORS_MASKS]

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='../log')

reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(min_lr=1e-7)

val_data = SKADataset(mode='validation').get_dataset()

yolo.compile(optimizer=optimizer, 
            loss=yolo_loss, 
            run_eagerly=True)

#yolo.load_weights(filepath='../checkpoints/cspdarknet53.h5')
#yolo = load_darknet_weights_in_yolo(yolo, trainable=True)
#yolo.summary()

yolo.fit(dataset_train, epochs=60, callbacks=[model_checkpoint_callback, tensorboard_callback, reduce_on_plateau], validation_data=val_data)
