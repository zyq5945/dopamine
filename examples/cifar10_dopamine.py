import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from tensorflow import keras
import os
import numpy as np

import sys

sys.path.append("../src")
from dopamine import Dopamine, dopamine

assert tf.__version__.startswith('2.')

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size = 200
num_classes = 10
shuffle_size = 60000
epochs = 20
validation_freq = 1
dopamine_batch_size = batch_size


def preprocess(x, y):
    """
    x is a simple image, not a batch
    """
    x = tf.cast(x, dtype=tf.float32) / 255.
	# x = tf.reshape(x, [28*28])
    y = tf.cast(y, dtype=tf.int32)
	# y = tf.one_hot(y, depth=num_classes)
    return x, y


def load_data():
    (x, y), (x_val, y_val) = datasets.cifar10.load_data()
    db = tf.data.Dataset.from_tensor_slices((x, y))
    db = db.map(preprocess).shuffle(shuffle_size).batch(batch_size)
    ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    ds_val = ds_val.map(preprocess).batch(batch_size)
    return db, ds_val


def main():
    db_train, db_test = load_data()

    model = Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        # Dopamine(input_shape=[30, 30, 32], batch_size= dopamine_batch_size),
        layers.Conv2D(64, (3, 3), activation='relu'),
        # Dopamine(input_shape=[28, 28, 64], batch_size= dopamine_batch_size),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        Dopamine(input_shape=[64], batch_size=dopamine_batch_size, use_bias=True),
        layers.Dense(10, activation='softmax')
    ])

    # optimizer= tf.keras.optimizers.Adam(lr=0.015)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.252, momentum=0.5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy']
                  )
    model.summary()

    model.fit(db_train, epochs=epochs, validation_data=db_test,
              validation_freq=validation_freq)
    model.evaluate(db_test)


if __name__ == '__main__':
    main()
