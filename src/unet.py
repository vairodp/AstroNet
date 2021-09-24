import tensorflow as tf
from tensorflow.keras import Model
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, MaxPool2D, Concatenate, Conv2D, Conv2DTranspose
from configs.train_config import ITER_PER_EPOCH, NUM_CLASSES, NUM_EPOCHS
from layers import cnn_block, decoder_block
from callbacks.telegram_callback import TelegramCallback
from callbacks.display_callback import DisplayCallback
from datasets.convo_ska import ConvoSKA

def unet_body(input_shape, num_classes=NUM_CLASSES+1):
    inputs = Input(shape=input_shape)
    # Encoder body
    x = cnn_block(inputs, num_filters=64, kernel_size=3, strides=1, activation='relu')
    skip64 = cnn_block(x, num_filters=64, kernel_size=3, strides=1, activation='relu')
    x = MaxPool2D((2,2))(skip64)
    x = cnn_block(x, num_filters=128, kernel_size=3, strides=1, activation='relu')
    skip128 = cnn_block(x, num_filters=128, kernel_size=3, strides=1, activation='relu')
    x = MaxPool2D((2,2))(skip128)
    x = cnn_block(x, num_filters=256, kernel_size=3, strides=1, activation='relu')
    skip256 = cnn_block(x, num_filters=256, kernel_size=3, strides=1, activation='relu')
    x = MaxPool2D((2,2))(skip256)
    x = cnn_block(x, num_filters=512, kernel_size=3, strides=1, activation='relu')
    skip512 = cnn_block(x, num_filters=512, kernel_size=3, strides=1, activation='relu')
    x = MaxPool2D((2,2))(skip512)
    x = cnn_block(x, num_filters=1024, kernel_size=3, strides=1, activation='relu')
    x = cnn_block(x, num_filters=1024, kernel_size=3, strides=1, activation='relu')

    # Decoder

    x = decoder_block(inputs=x, skip_connection=skip512, num_filters=512)
    x = decoder_block(inputs=x, skip_connection=skip256, num_filters=256)
    x = decoder_block(inputs=x, skip_connection=skip128, num_filters=128)
    x = decoder_block(inputs=x, skip_connection=skip64, num_filters=64)

    outputs = Conv2D(filters=num_classes, kernel_size=1, padding='same')(x)

    return Model(inputs=inputs, outputs=outputs, name='U-Net')

def unet_body_small(input_shape, num_classes=NUM_CLASSES+1):
    inputs = Input(shape=input_shape)
    # Encoder body
    x = cnn_block(inputs, num_filters=64, kernel_size=3, strides=1, activation='relu')
    skip64 = cnn_block(x, num_filters=64, kernel_size=3, strides=1, activation='relu')
    x = MaxPool2D((2,2))(skip64)
    x = cnn_block(x, num_filters=128, kernel_size=3, strides=1, activation='relu')
    skip128 = cnn_block(x, num_filters=128, kernel_size=3, strides=1, activation='relu')
    x = MaxPool2D((2,2))(skip128)
    x = cnn_block(x, num_filters=256, kernel_size=3, strides=1, activation='relu')
    skip256 = cnn_block(x, num_filters=256, kernel_size=3, strides=1, activation='relu')
    x = MaxPool2D((2,2))(skip256)
    x = cnn_block(x, num_filters=512, kernel_size=3, strides=1, activation='relu')

    # Decoder

    x = decoder_block(inputs=x, skip_connection=skip256, num_filters=256)
    x = decoder_block(inputs=x, skip_connection=skip128, num_filters=128)
    x = decoder_block(inputs=x, skip_connection=skip64, num_filters=64)

    outputs = Conv2D(filters=num_classes, kernel_size=1, padding='same')(x)

    return Model(inputs=inputs, outputs=outputs, name='U-Net')

class SourceSegmentation(tf.keras.Model):
    def __init__(self, input_shape, use_class_weights=True, tiny=False, num_classes=NUM_CLASSES+1):
        super().__init__(name='SourceSegmentation')
        self.num_classes = num_classes
        self.use_class_weights = use_class_weights
        if tiny:
            self.model = unet_body_small(input_shape, num_classes)
        else:
            self.model = unet_body(input_shape, num_classes)  
    
    def call(self, x, training=False):
        return self.model(x, training)
    
    def train_step(self, data):
        x, y, sample_weights = data['image'], data['label'], data['weights']

        if not self.use_class_weights:
            sample_weights = None

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.compiled_loss(y, y_pred, sample_weight=sample_weights, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weights)
        #mAP_tracker.update_state(y_pred, data['bbox'], data['num_of_bbox'])
        metrics = {m.name: m.result() for m in self.metrics}
        #metrics['reg_loss'] = (loss - y_pred)[0]
        return metrics
    
    def test_step(self, data):
        x, y, sample_weights = data['image'], data['label'], data['weights']

        if not self.use_class_weights:
            sample_weights = None
        
        y_pred = self(x, training=True)
        self.compiled_loss(y, y_pred, sample_weight=sample_weights, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weights)

        #mAP_tracker.update_state(y_pred, data['bbox'], data['num_of_bbox'])
        metrics = {m.name: m.result() for m in self.metrics}
        #metrics['reg_loss'] = (loss - y_pred)[0]

        return metrics

unet = SourceSegmentation((128,128,1), use_class_weights=True)
unet.model.summary()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.003, clipvalue=1.0)
checkpoint_filepath = '../checkpoints/unet-best.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='../log')
early_stop_callback = tf.keras.callbacks.EarlyStopping(patience=10)
unet.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
dataset_train = ConvoSKA(mode='train').get_dataset()
val_data = ConvoSKA(mode='validation').get_dataset()
display_callback = DisplayCallback(val_data)
unet.fit(dataset_train, epochs=NUM_EPOCHS, validation_data=val_data, 
        callbacks=[model_checkpoint_callback, display_callback, tensorboard_callback, early_stop_callback], steps_per_epoch=ITER_PER_EPOCH)
