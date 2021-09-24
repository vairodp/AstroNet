import tensorflow as tf
from tensorflow.keras import Model
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, MaxPool2D, Concatenate, Conv2D, Conv2DTranspose
from configs.train_config import ITER_PER_EPOCH, NUM_CLASSES, NUM_EPOCHS
from layers import cnn_block
from callbacks.telegram_callback import TelegramCallback
from datasets.convo_ska import ConvoSKA

def decoder_block(inputs, skip_connection, num_filters):
    x = Conv2DTranspose(num_filters, kernel_size=2, padding='same', strides=2)(inputs)
    x = Concatenate()([x, skip_connection])
    x = cnn_block(x, num_filters=num_filters, kernel_size=3, strides=1, activation='relu')
    x = cnn_block(x, num_filters=num_filters, kernel_size=3, strides=1, activation='relu')

    return x


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

class SourceSegmentation(tf.keras.Model):
    def __init__(self, input_shape, use_class_weights=True, num_classes=NUM_CLASSES+1):
        super().__init__(name='SourceSegmentation')
        self.num_classes = num_classes
        self.use_class_weights = use_class_weights
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

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask

class DisplayCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data):
        self.val_data = val_data

    def on_epoch_end(self, epoch, logs=None):
        for batch in self.val_data:
            predictions = self.model.model.predict(batch['image'])
            labels = batch['label']
            images = batch['image']
            for prediction, label, image in zip(predictions, labels, images):
                pred = create_mask(prediction)

                _, (ax1,ax2,ax3) = plt.subplots(ncols=3, nrows=1)
                ax1.imshow(label)
                ax1.set_title('LABEL')

                ax2.imshow(pred)
                ax2.set_title('PREDICTION')

                ax3.imshow(image)
                ax3.set_title('REAL IMAGE')
                plt.savefig(f'../data/results/image_at_epoch_{epoch+1}')
                break
            break
        
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

unet = SourceSegmentation((128,128,1))
unet.model.summary()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.003, clipvalue=1.0)
checkpoint_filepath = '../checkpoints/unet-best.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
unet.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
dataset_train = ConvoSKA(mode='train').get_dataset()
val_data = ConvoSKA(mode='validation').get_dataset()
display_callback = DisplayCallback(val_data)
unet.fit(dataset_train, epochs=NUM_EPOCHS, validation_data=val_data, 
        callbacks=[model_checkpoint_callback, display_callback], steps_per_epoch=ITER_PER_EPOCH)
