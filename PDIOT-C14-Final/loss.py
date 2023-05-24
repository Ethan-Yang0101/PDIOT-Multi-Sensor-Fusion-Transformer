import tensorflow as tf


def smoothed_sparse_categorical_crossentropy(label_smoothing: float = 0.0):
    def loss_fn(y_true, y_pred):
        num_classes = tf.shape(y_pred)[-1]
        y_true = tf.one_hot(y_true, num_classes)
        y_pred = tf.expand_dims(y_pred, axis=1)
        loss = tf.keras.losses.categorical_crossentropy(
            y_true, y_pred, label_smoothing=label_smoothing)
        return tf.reduce_mean(loss)
    return loss_fn
