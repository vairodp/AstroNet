import os

from tensorflow.keras import callbacks
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np

import datasets
#from load_weights import load_darknet_weights_in_yolo
from callbacks.telegram_callback import TelegramCallback
import tensorflow as tf
from yolo_v4 import YoloV4
from small_yolo import SmallYolo
from loss import YoloLoss
from datasets.ska_dataset import SKADataset
from configs.train_config import NUM_CLASSES, loss_params, get_anchors

import tensorflow_datasets as tfds

SMALL = True

#Loading Bar
#from tqdm import tqdm

anchor_dict = get_anchors()

if SMALL:
    anchor_dict = get_anchors(model='small_yolo')
    yolo = SmallYolo(num_classes=NUM_CLASSES)
    dataset_train = SKADataset(mode='train', grid_size=8, **anchor_dict)
    val_data = SKADataset(mode='validation', grid_size=8, **anchor_dict).get_dataset()
    checkpoints_path = '../checkpoints/small_yolo/'

else:
    yolo = YoloV4(num_classes=NUM_CLASSES)
    dataset_train = SKADataset(mode='train')
    val_data = SKADataset(mode='validation').get_dataset()
    checkpoints_path = '../checkpoints/yolo/'

checkpoint_filepath = checkpoints_path + 'model.{epoch:02d}-{loss:.2f}.h5'
class_weights = dataset_train.get_class_weights()
dataset_train = dataset_train.get_dataset()
#class_weights = None

anchors = anchor_dict['anchors']
anchor_masks = anchor_dict['anchor_masks']
yolo_loss = [YoloLoss(anchors=anchors[mask],
                    class_weights=class_weights,
                    iou_threshold=loss_params['iou_threshold'], 
                    smooth_factor=loss_params['smooth_factor'], 
                    use_ciou=True) for mask in anchor_masks]

optimizer = tf.keras.optimizers.Adam(learning_rate=0.003, clipvalue=1.0)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='../log')

telegram_callback = TelegramCallback()

reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(min_lr=1e-7)

#yolo = SmallYolo(num_classes=NUM_CLASSES)

yolo.summary(line_length=170)

yolo.compile(optimizer=optimizer, 
            loss=yolo_loss, 
            run_eagerly=True)

#yolo.load_weights(filepath='../checkpoints/cspdarknet53.h5')
#yolo = load_darknet_weights_in_yolo(yolo, trainable=True)
#yolo.summary()

yolo.fit(dataset_train, epochs=60, callbacks=[model_checkpoint_callback, tensorboard_callback, reduce_on_plateau, telegram_callback], 
        validation_data=val_data, steps_per_epoch=2)
