import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np

import datasets
import tensorflow as tf
from yolo_v4 import YoloV4
from loss import YoloLoss
from datasets.ska_dataset import SKADataset
from configs.yolo_v4 import ANCHORS, ANCHORS_MASKS, NUM_CLASSES, loss_params

#Loading Bar
from tqdm import tqdm

checkpoint_filepath = '../checkpoints/model.{epoch:02d}-{loss:.2f}.h5'

dataset_train = SKADataset(mode='train')
class_weights = dataset_train.get_class_weights()
dataset_train = dataset_train.get_dataset()

class_weights = None
yolo = YoloV4(num_classes=NUM_CLASSES)
yolo.build((None, 128, 128, 3))
yolo.summary()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.003, clipvalue=1.0)
loss_small = YoloLoss(anchors=ANCHORS[ANCHORS_MASKS[0]],
                    class_weights=class_weights,
                    iou_threshold=loss_params['iou_threshold'], 
                    smooth_factor=loss_params['smooth_factor'], 
                    use_ciou=True, reduction="sum")
loss_med = YoloLoss(anchors=ANCHORS[ANCHORS_MASKS[1]],
                    class_weights=class_weights,
                    iou_threshold=loss_params['iou_threshold'], 
                    smooth_factor=loss_params['smooth_factor'], 
                    use_ciou=True, reduction="sum")
loss_large = YoloLoss(anchors=ANCHORS[ANCHORS_MASKS[2]],
                    class_weights=class_weights,
                    iou_threshold=loss_params['iou_threshold'], 
                    smooth_factor=loss_params['smooth_factor'], 
                    use_ciou=True, reduction="sum")

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True)

val_data = SKADataset(mode='validation').get_dataset()

yolo.compile(optimizer=optimizer, 
            loss=[loss_small, loss_med, loss_large], 
            run_eagerly=True)
#yolo.load_weights(filepath='../checkpoints/model.02-0.68.h5')
yolo.fit(dataset_train, epochs=60, callbacks=[model_checkpoint_callback], validation_data=val_data)

#def train_one_step(x, y):
#    with tf.GradientTape() as tape:
#        pred = yolo(x, training=True)
#        true_s, true_m, true_l = y
#        pred_s, pred_m, pred_l = pred
#        loss_s = loss_large(y_pred=pred_l, y_true=true_l)
#        tf.print("GLOBAL LOSS: ", loss_s)

#    grads = tape.gradient(total_loss, yolo.trainable_variables)
#    optimizer.apply_gradients(
#        zip(grads, yolo.trainable_variables))

#    return pred_loss

#for epoch in tqdm(range(1), desc='Training epochs..'):
#    for data in dataset_train:
#        print(len(data['image']))
#        train_one_step(data['image'], data['label'])
#        break

#        tf.print(loss)