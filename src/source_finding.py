import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import MaxPool2D, Input, Lambda, Concatenate, Conv2D, Reshape
from tensorflow.python.keras.layers.convolutional import Conv, UpSampling2D
from configs.train_config import NUM_CLASSES, MAX_NUM_BBOXES, loss_params, INITIAL_LR, NUM_EPOCHS, ITER_PER_EPOCH
from datasets.ska_dataset import SKADataset
from datasets.convo_ska import ConvoSKA
from loss import yolo3_loss
from anchors import CUSTOM_ANCHORS1, CUSTOM_ANCHORS_TINY, compute_normalized_anchors, YOLOLITE_ANCHORS

from layers import DropBlock, cnn_block, residual_block, route_group, scale_prediction, upsample

def backbone_tiny(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    input_data = cnn_block(inputs, num_filters=16, kernel_size=1, strides=1)
    input_data = MaxPool2D()(input_data)
    input_data = DropBlock()(input_data)
    input_data = cnn_block(input_data, num_filters=32, kernel_size=1, strides=1)
    input_data = MaxPool2D()(input_data)
    input_data = DropBlock()(input_data)
    input_data = cnn_block(input_data, num_filters=64, kernel_size=1, strides=1)

    return tf.keras.Model(inputs, input_data, name="Backbone")

class SourceDetection(tf.keras.Model):
    def __init__(self, input_shape, num_classes):
        super().__init__(name='SourceDetection')
        self.num_classes = num_classes

        backbone = backbone_tiny(input_shape)
        print(backbone.summary())
        print(backbone.outputs)

        inputs = tf.keras.Input(shape=input_shape)
        x = backbone(inputs)


        # Detection head
        out_1 = cnn_block(x, num_filters=32, kernel_size=1, strides=1)
        out_1 = UpSampling2D(size=(2,2))(out_1)
        out_1 = cnn_block(out_1, num_filters=16, kernel_size=1, strides=1)
        out_1 = UpSampling2D(size=(2,2))(out_1)
        out_1 = Conv2D(
            filters=1,
            kernel_size=1,
            strides=1,
            padding="same",
            activation='sigmoid',
            use_bias=True)(out_1)

        # Classification head
        out_2 = MaxPool2D(pool_size=((x.shape[1], x.shape[2])))(x)
        out_2 = Conv2D(
            filters=300,
            kernel_size=1,
            strides=1,
            padding="same",
            activation='softmax',
            use_bias=True)(out_2)
        out_2 = Reshape((MAX_NUM_BBOXES, 3))(out_2)
        
        self.model = tf.keras.Model([inputs], out_2)
    
    def call(self, x, training=False):
        return self.model(x, training)
    
    def train_step(self, data):
        x, y = data['image'], list(data['label'])[1]

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        #mAP_tracker.update_state(y_pred, data['bbox'], data['num_of_bbox'])
        metrics = {m.name: m.result() for m in self.metrics}
        #metrics['reg_loss'] = (loss - y_pred)[0]
        return metrics
    
    def test_step(self, data):
        x, y = data['image'], list(data['label'])[1]
        
        y_pred = self(x, training=True)
        loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)

        #mAP_tracker.update_state(y_pred, data['bbox'], data['num_of_bbox'])
        metrics = {m.name: m.result() for m in self.metrics}
        #metrics['reg_loss'] = (loss - y_pred)[0]

        return metrics


yolo = SourceDetection(input_shape=(128,128,1), num_classes=NUM_CLASSES)
#yolo.model.summary(line_length=200)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.003, clipvalue=1.0)

checkpoint_filepath = '../checkpoints/convo-best1.h5'

def sigmoid_focal_loss(gamma=4.3, alpha=0.25):

    def focal_loss(y_true, y_pred):
        sigmoid_loss = K.binary_crossentropy(y_true, y_pred, from_logits=True)

        #pred_prob = tf.sigmoid(y_pred)
        pred_prob = y_pred
        p_t = ((y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob)))
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        alpha_weight_factor = (y_true * alpha + (1 - y_true) * (1 - alpha))

        sigmoid_focal_loss = modulating_factor * alpha_weight_factor * sigmoid_loss
        #sigmoid_focal_loss = tf.reduce_sum(sigmoid_focal_loss, axis=-1)

        return sigmoid_focal_loss
    
    return focal_loss

"""
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

yolo.compile(optimizer=optimizer, loss=sigmoid_focal_loss(),
            metrics='accuracy')
dataset_train = ConvoSKA(mode='train').get_dataset()
val_data = ConvoSKA(mode='validation').get_dataset()
yolo.fit(dataset_train, epochs=NUM_EPOCHS, callbacks=[model_checkpoint_callback],
        validation_data=val_data, steps_per_epoch=ITER_PER_EPOCH)
"""