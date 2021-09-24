import tensorflow as tf
import matplotlib.pyplot as plt

from utils import create_mask

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

                for ax in (ax1, ax2, ax3):
                    ax.get_xaxis().set_ticks([])
                    ax.get_yaxis().set_ticks([])

                plt.savefig(f'../data/results/image_at_epoch_{epoch+1}')
                break
            break
        
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))