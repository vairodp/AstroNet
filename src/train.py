import datasets
import tensorflow as tf
from yolo_v4 import YoloV4
from loss import YoloLoss
from datasets.ska_dataset import SKADataset
from configs.yolo_v4 import NUM_CLASSES, loss_params

yolo = YoloV4(num_classes=NUM_CLASSES)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.03)
loss = YoloLoss(iou_threshold=loss_params['iou_threshold'])

dataset = SKADataset(mode='train')

def train_one_step(x, y):
    with tf.GradientTape() as tape:
        pred = yolo(x, training=True)
        pred_loss = loss(y_pred=pred, y_true=y)
        regularization_loss = tf.reduce_sum(yolo.losses)
        total_loss = pred_loss + regularization_loss

    grads = tape.gradient(total_loss, yolo.trainable_variables)
    optimizer.apply_gradients(
        zip(grads, yolo.trainable_variables))

    return pred_loss

for epoch in range(100):
    for data in dataset:
        loss = train_one_step(data['image'], data['label'])
        print(loss)

