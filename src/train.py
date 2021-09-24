import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import tensorflow as tf

from yolo_v4 import YOLOv4
from load_weights import load_weights
from datasets.ska_dataset import SKADataset
from callbacks.telegram_callback import TelegramCallback
from callbacks.lr_scheduler import LinearWarmupCosineDecay
from configs.train_config import DARKNET_WEIGHTS_PATH, IMG_SIZE, LOAD_WEIGHTS, NUM_CLASSES, NUM_EPOCHS, ITER_PER_EPOCH, USE_EARLY_STOPPING
from configs.train_config import INITIAL_LR, USE_COSINE_DECAY, USE_TENSORBOARD, DARKNET_WEIGHTS, USE_CUSTOM_ANCHORS, USE_TELEGRAM_CALLBACK

from anchors import CUSTOM_ANCHORS, YOLOV4_ANCHORS


if USE_CUSTOM_ANCHORS:
    anchors = CUSTOM_ANCHORS
    anchor_masks = [np.array([0,1,2,3,4,5,6]), np.array([7,8]), np.array([9])]
else:
    anchors = YOLOV4_ANCHORS
    anchor_masks = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

yolo = YOLOv4(input_shape=(128,128,1), num_classes=NUM_CLASSES, anchors=anchors)
dataset_train = SKADataset(mode='train', anchors=anchors, anchor_masks=anchor_masks)
val_data = SKADataset(mode='validation', anchors=anchors, anchor_masks=anchor_masks).get_dataset()
checkpoint_filepath = '../checkpoints/yolo/model-best.h5'

yolo.model.summary(line_length=200)
dataset_train = dataset_train.get_dataset()

optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LR, clipvalue=1.0)

if DARKNET_WEIGHTS:
    yolo = load_weights(yolo, folder_path=DARKNET_WEIGHTS_PATH)

yolo.compile(optimizer=optimizer, loss={'output_1': lambda y_true, y_pred: y_pred}) # use custom yolo_loss Lambda layer

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

if USE_COSINE_DECAY:
    warmup_steps = int(0.10 * NUM_EPOCHS) * ITER_PER_EPOCH
    max_decay_steps = NUM_EPOCHS * ITER_PER_EPOCH - warmup_steps
    lr_scheduler = LinearWarmupCosineDecay(initial_lr=INITIAL_LR, 
                    final_lr=INITIAL_LR/100, 
                    warmup_steps=warmup_steps, 
                    max_decay_steps=max_decay_steps)
else:
    def scheduler(epoch, lr):
        if epoch == 14:
            return 0.0013
        elif epoch == 5714:
            return lr * 0.1
        elif epoch == 6428:
            return lr * 0.1
        else:
            return lr
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

callback = [telegram_callback, model_checkpoint_callback]

if USE_TELEGRAM_CALLBACK:
    telegram_callback = TelegramCallback()
    callback.append(telegram_callback)

if USE_TENSORBOARD:
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='../log')
    callback.append(tensorboard_callback)

if USE_EARLY_STOPPING:
    early_stop_callback = tf.keras.callbacks.EarlyStopping(patience=10)
    callback.append(early_stop_callback)

if LOAD_WEIGHTS:
    y_true = [np.zeros((1, int(IMG_SIZE/(32/l)), int(IMG_SIZE/(32/l)), 3, NUM_CLASSES+5)) for l in [4,2,1]]
    yolo([np.zeros((1,128,128,1)), *y_true])
    yolo.load_weights(filepath=checkpoint_filepath)


yolo.fit(dataset_train, epochs=NUM_EPOCHS, callbacks=callback,
        validation_data=val_data, steps_per_epoch=ITER_PER_EPOCH)