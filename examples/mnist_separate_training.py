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
validation_freq = 2
dopamine_batch_size = batch_size


def preprocess(x, y):
    """
    x is a simple image, not a batch
    """
    x = tf.cast(x, dtype=tf.float32) / 255.
    # x = tf.reshape(x, [28*28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=num_classes)
    return x, y


def load_data():
    (x, y), (x_val, y_val) = datasets.mnist.load_data()
    db = tf.data.Dataset.from_tensor_slices((x, y))
    db = db.map(preprocess).shuffle(shuffle_size).batch(batch_size)
    ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    ds_val = ds_val.map(preprocess).batch(batch_size)
    return db, ds_val


def create_params():
    optimizer = tf.keras.optimizers.SGD(lr=0.25, momentum=0.6)
    losser = tf.losses.CategoricalCrossentropy(from_logits=True)
    acc = tf.keras.metrics.CategoricalAccuracy()
    return optimizer, losser, acc


def tran_step(epoch, model, db_train, trainable_variables):
    optimizer, losser, acc = create_params()

    for step, (x, y) in enumerate(db_train):
        with tf.GradientTape() as tape:
            logits = model(x)
            loss = losser(y, logits)
        # print('---ff', logits)

        grads = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(grads, trainable_variables))
        acc.update_state(y, logits)

        print('epoch:', epoch, 'step:', step, 'loss:', loss.numpy(), 'acc:', acc.result().numpy())

    return losser, acc


def split_dopamine_trainable_variables(layers):
    dtv = []
    otv = []
    for l in layers:
        vs = l.trainable_variables
        if isinstance(l, Dopamine):
            dtv.extend(vs)
        else:
            otv.extend(vs)

    return dtv, otv


def main():
    db_train, db_test = load_data()

    lys = [
        layers.Reshape(target_shape=(784,), input_shape=(28, 28)),
        layers.Dense(256, activation='relu'),
        Dopamine(input_shape=[256], batch_size=dopamine_batch_size),
        layers.Dense(64, activation='relu'),
        Dopamine(input_shape=[64], batch_size=dopamine_batch_size, use_bias=True),
        layers.Dense(10)
    ]
    model = Sequential(lys)

    # optimizer= tf.keras.optimizers.Adam(lr=0.025)
    optimizer, losser, acc = create_params()
    model.compile(optimizer=optimizer,
                  loss=losser,
                  metrics=[acc]
                  )
    model.summary()

    # dopamine_variables = lys[2].trainable_variables + lys[4].trainable_variables
    # other_variables = lys[0].trainable_variables + lys[1].trainable_variables + lys[3].trainable_variables
    # + lys[5].trainable_variables
    dopamine_variables, other_variables = split_dopamine_trainable_variables(lys)
    print(len(dopamine_variables), len(other_variables))
    for task in range(8):

        for epoch in range(3):
            tran_step(epoch, model, db_train, other_variables)
        for epoch in range(1):
            tran_step(epoch, model, db_train, dopamine_variables)

    model.evaluate(db_test)


if __name__ == '__main__':
    main()
