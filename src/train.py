import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np

import datasets
#from load_weights import load_darknet_weights_in_yolo
from callbacks.telegram_callback import TelegramCallback
from callbacks.lr_scheduler import CyclicLR, LinearWarmupCosineDecay
import tensorflow as tf
from yolo_v4 import YOLOv4
#from small_yolo import SmallYolo
from loss import YoloLoss
from datasets.ska_dataset import SKADataset
from configs.train_config import NUM_CLASSES, NUM_EPOCHS, ITER_PER_EPOCH, loss_params, get_anchors


from anchors import YOLOV4_ANCHORS,  compute_normalized_anchors

SMALL = False

#Loading Bar
#from tqdm import tqdm

anchor_dict = get_anchors()

"""
if SMALL:
    anchor_dict = get_anchors(model='small_yolo')
    yolo = SmallYolo(num_classes=NUM_CLASSES)
    dataset_train = SKADataset(mode='train', grid_size=8, **anchor_dict)
    val_data = SKADataset(mode='validation', grid_size=8, **anchor_dict).get_dataset()
    checkpoints_path = '../checkpoints/small_yolo/'

else:
"""
yolo = YOLOv4(input_shape=(128,128,3), num_classes=NUM_CLASSES, anchors=YOLOV4_ANCHORS, weights=None, training=True)
dataset_train = SKADataset(mode='train')
val_data = SKADataset(mode='validation').get_dataset()
checkpoints_path = '../checkpoints/yolo/'

checkpoint_filepath = checkpoints_path + 'model-best.h5'
#class_weights = dataset_train.get_class_weights()
dataset_train = dataset_train.get_dataset()
class_weights = None

anchors = compute_normalized_anchors(YOLOV4_ANCHORS, (128,128,3))

yolo_loss = [YoloLoss(anchors=anchor,
                    class_weights=class_weights,
                    iou_threshold=loss_params['iou_threshold'], 
                    smooth_factor=loss_params['smooth_factor'], 
                    use_ciou=True) for anchor in anchors]

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003, clipvalue=1.0)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True)

#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='../log')

telegram_callback = TelegramCallback()

warmup_steps = int(0.10 * NUM_EPOCHS) * ITER_PER_EPOCH

max_decay_steps = NUM_EPOCHS * ITER_PER_EPOCH - warmup_steps

lr_scheduler = LinearWarmupCosineDecay(initial_lr=0.01, final_lr = 1e-3, warmup_steps= warmup_steps, max_decay_steps=max_decay_steps)

#lr_scheduler = CyclicLR(base_lr=1e-5, max_lr=1e-2,step_size=ITER_PER_EPOCH*4)

#reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(min_lr=1e-6,factor=0.001)

#yolo = SmallYolo(num_classes=NUM_CLASSES)

yolo.model.summary(line_length=170)

#yolo.load_weights(filepath='../checkpoints/yolo/best.h5')
#yolo = load_darknet_weights_in_yolo(yolo, trainable=True)

yolo.compile(optimizer=optimizer, 
            loss=yolo_loss, 
            run_eagerly=True)
#yolo.summary()

yolo.fit(dataset_train, epochs=NUM_EPOCHS, callbacks=[model_checkpoint_callback, telegram_callback, lr_scheduler],
        validation_data=val_data, steps_per_epoch=ITER_PER_EPOCH)