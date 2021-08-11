import tensorflow as tf
import math

class LinearWarmupCosineDecay(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr, final_lr, warmup_steps, max_decay_steps):
        super().__init__()
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.max_decay_steps = max_decay_steps
        self.lr_schedule = self._create_lr_schedule()
    
    def _create_lr_schedule(self):
        lr_warmup = tf.linspace(0., self.initial_lr, self.warmup_steps)
        lr_steps = tf.range(self.max_decay_steps, dtype=tf.float32)
        pi = tf.constant(math.pi)
        lr_cosine = tf.convert_to_tensor([self.final_lr + 0.5 * (self.initial_lr + self.final_lr) * \
                (1 + tf.math.cos(pi * step / self.max_decay_steps)) for step in lr_steps])
        return tf.concat([lr_warmup, lr_cosine], axis=0)

    def on_train_batch_begin(self, batch, logs):
        if batch < len(self.lr_schedule):
            new_lr = self.lr_schedule[batch]
        else:
            new_lr = self.lr_schedule[-1]
        self.model.optimizer.lr = new_lr