import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pathlib
import signal

from functions.training.config import *
from functions.training.dataloader import *
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, ReLU, MaxPool2D, UpSampling2D
from lib.u2net import *

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    try: tf.config.experimental.set_memory_growth(device, True)
    except: pass


def set_args(
    bs = None,
    lr = None,
    si = None):
    batch_size = bs if bs != None else batch_size
    learning_rate = lr if lr != None else 0.001
    save_interval = si if si != None else 250

# Overwrite the default optimizer
adam = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=.9, beta_2=.999, epsilon=1e-08)

def calculate_batchsize(loss, batch_size):
    if loss > 1: return batch_size
    elif loss <= 1 and loss > 0.5: return batch_size // 2
    elif loss <= 0.5 and loss > 0.25: return batch_size // 4
    elif loss <= 0.25 and loss > 0.125: return batch_size // 8
    elif loss <= 0.125 and loss > 0.0625: return batch_size // 16
    elif loss <= 0.0625 and loss > 0.03125: return batch_size // 32
    else: return batch_size // 64

def train(batch_size, num_epochs = None, resume = None, model = None):
    bs = batch_size
    if num_epochs != None:
        epochs = num_epochs
    
    if model is None:
        inputs = keras.Input(shape=default_in_shape)
        net = U2NET()
        out = net(inputs)
        model = keras.Model(inputs=inputs, outputs=out, name='u2netmodel')
        model.compile(optimizer=adam, loss=bce_loss, metrics=None)
        model.summary()

    if resume:
        print('[ OK ] Loading weights from %s' % resume)
        model.load_weights(resume)

    # helper function to save state of model
    def save_weights():
        print('[ OK ] Saving state of model to %s' % weights_file)
        model.save_weights(str(weights_file))
    
    # signal handler for early abortion to autosave model state
    def autosave(sig, frame):
        print('[ OK ] Training aborted early... Saving weights.')
        save_weights()
        exit(0)

    for sig in [signal.SIGABRT, signal.SIGINT, signal.SIGTSTP]:
        signal.signal(sig, autosave)

    # start training
    print('[ OK ] Starting training')
    for e in range(epochs):
        try:
            feed, out = load_training_batch(batch_size=batch_size)
            loss = model.train_on_batch(feed, out)
        except KeyboardInterrupt:
            save_weights()
            return
        except ValueError:
            continue

        if e % 10 == 0:
            print('[%05d] Loss: %.4f || Batch Size: %d' % (e, loss, batch_size))

        if save_interval and e > 0 and e % save_interval == 0:
            save_weights()

    return model