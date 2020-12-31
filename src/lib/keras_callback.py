import tensorflow as tf

class U2NETCallback(tf.keras.callbacks.Callback):

    def __init__(self, weights_file = None):
        self.weights_file = weights_file
        super(U2NETCallback, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        if epoch % 8 == 0:
            print('\n[ OK ] Saving weights for epoch: {:,}'.format(epoch))
            self.model.save_weights(self.weights_file)