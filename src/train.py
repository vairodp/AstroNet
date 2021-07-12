import datasets
import tensorflow as tf
from yolo_v4 import YoloV4
from loss import YoloLoss
from datasets.ska_dataset import SKADataset
from configs.yolo_v4 import NUM_CLASSES, loss_params

#Loading Bar
from tqdm import tqdm

yolo = YoloV4(num_classes=NUM_CLASSES)
yolo.build((32, 128, 128, 3))
yolo.summary()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.003, clipnorm=1.0)
yolo_loss = YoloLoss(iou_threshold=loss_params['iou_threshold'], use_ciou=True)
#print(type(loss))

dataset = SKADataset(mode='train').get_dataset()

#yolo.compile(optimizer=optimizer, loss=yolo_loss)

def train_one_step(x, y):
    with tf.GradientTape() as tape:
        pred = yolo(x, training=True)
        pred_loss = yolo_loss(y_pred=pred, y_true=y)
        regularization_loss = tf.reduce_sum(yolo.losses)
        total_loss = pred_loss + regularization_loss

    grads = tape.gradient(total_loss, yolo.trainable_variables)
    optimizer.apply_gradients(
        zip(grads, yolo.trainable_variables))

    return pred_loss

for epoch in tqdm(range(100), desc='Training epochs..'):
    for data in dataset:
        print(len(data['image']))
        loss = train_one_step(data['image'], data['label'])
        tf.print(loss)

