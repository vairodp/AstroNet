import tensorflow as tf
from tensorflow.keras import backend as K

import numpy as np
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


class CyclicLR(tf.keras.callbacks.Callback):    
    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        #Resets cycle iterations.
        #Optional boundary/step size adjustment.
        
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())