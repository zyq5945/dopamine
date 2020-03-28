import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential, regularizers
from resnet import ResNet18
import numpy as np
import random

import os
import sys

sys.path.append("../src")
from dopamine import Dopamine, dopamine


assert tf.__version__.startswith('2.')

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batchsz = 250
epochs = 180
validation_freq = 1
input_shape = (batchsz, 32, 32, 3)
model_path = 'data/'
initial_epoch = 116
dopamine_batch_size = batchsz

# 1. 归一化函数实现；cifar100 均值和方差，自己计算的。
img_mean = tf.constant([0.50736203482434500, 0.4866895632914611, 0.4410885713465068])
img_std = tf.constant([0.26748815488001604, 0.2565930997269337, 0.2763085095510783])


def normalize(x, mean=img_mean, std=img_std):
    x = (x - mean) / std
    return x


# 2. 数据预处理，仅仅是类型的转换。    [-1~1]
def preprocess(x, y):
    x = tf.pad(x, [[4, 4], [4, 4], [0, 0]])  # 上下填充4个0，左右填充4个0，变为[40, 40, 3]
    x = tf.image.random_crop(x, [32, 32, 3])
    x = tf.image.random_flip_left_right(x)
    # x: [0,255]=> -1~1   其次：normalizaion
    x = tf.cast(x, dtype=tf.float32) / 255.
    # 0~1 => D(0,1) 调用函数；
    x = normalize(x)
    y = tf.cast(y, dtype=tf.int32)
    return x, y


# 3. 学习率调整测率200epoch
def lr_schedule_300ep(epoch):
    if epoch < 60:
        return 0.1
    if epoch < 130:
        return 0.02
    if epoch < 160:
        return 0.004
    if epoch < 200:
        return 0.0008
    if epoch < 250:
        return 0.0003
    if epoch < 300:
        return 0.0001
    else:
        return 0.00006


def load_data():
    # 数据集的加载
    (x, y), (x_test, y_test) = datasets.cifar100.load_data()
    y = tf.squeeze(y)  # 或者tf.squeeze(y, axis=1)把1维度的squeeze掉。
    y_test = tf.squeeze(y_test)  # 或者tf.squeeze(y, axis=1)把1维度的squeeze掉。
    print(x.shape, y.shape, x_test.shape, y_test.shape)

    # 训练集和标签包装成Dataset对象
    train_db = tf.data.Dataset.from_tensor_slices((x, y))
    # 测试集和标签包装成Dataset对象
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # train_db = train_db.take(1000).shuffle(50000).map(preprocess).batch(batchsz)
    # test_db = test_db.take(1000).map(preprocess).batch(batchsz)

    train_db = train_db.shuffle(50000).map(preprocess).batch(batchsz)
    test_db = test_db.map(preprocess).batch(batchsz)

    # 我们来取一个样本，测试一下sample的形状。
    sample = next(iter(train_db))
    print('sample:', sample[0].shape, sample[1].shape,
          tf.reduce_min(sample[0]),
          tf.reduce_max(sample[0]))  # 值范围为[0,1]

    return train_db, test_db


def create_params():
    optimizer = tf.keras.optimizers.SGD(lr=0.02, momentum=0.9, decay=5e-4)
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    acc = tf.keras.metrics.SparseCategoricalAccuracy()
    return optimizer, loss, acc


def create_models():
    lys = [
        layers.GlobalAveragePooling2D(),
        layers.Lambda(lambda x: x),
        layers.Dense(100, activation=None, kernel_regularizer=regularizers.l2(5e-4)),
    ]
    return ResNet18(), Sequential(lys)


def save_models(resnet_model, class_model, epoch, acc):
    acc = int(acc*10000)
    save_model(resnet_model, '%s%05d_%03d_my_model1.h5' % (model_path, acc, epoch))
    save_model(class_model, '%s%05d_%03d_my_model2.h5' % (model_path, acc, epoch))


def load_models(resnet_model, class_model):
    load_model(resnet_model, 'data/07551_my_model1.h5')
    # load_model(class_model, 'data/07551_my_model2.h5')


def fill_models(class_model):
    # vv = Dopamine(input_shape=[250, 512], batch_size=dopamine_batch_size, use_bias=True)
    vv = layers.Dropout(0.5)
    la = class_model.get_layer(index=1)
    class_model.pop()
    class_model.add(vv)
    class_model.add(la)


def load_model(model, file):
    if os.path.isfile(file):
        model.load_weights(file)


def save_model(model, file):
    model.save_weights(file)


def create_callbacks(resnet_model, class_model):
    # def on_epoch_begin(epoch, logs):
    #     print('----on_epoch_begin', epoch, logs)

    def on_epoch_end(epoch, logs):
        key = 'val_sparse_categorical_accuracy'
        acc = logs[key] if key in logs else 0.0
        save_models(resnet_model, class_model, epoch, acc)
        # print('----on_epoch_end', epoch, logs)

    # def on_batch_begin(batch, logs):
    #     print('----on_batch_begin', batch, logs)
    #
    # def on_batch_end(batch, logs):
    #     acc = logs['sparse_categorical_accuracy']
    #
    #     print('----on_batch_end%.5f' % acc, batch, logs)
    #
    # def on_train_begin(logs):
    #     print('----on_train_begin', logs)
    #
    # def on_train_end(logs):
    #     print('----on_train_end', logs)

    task_callback = tf.keras.callbacks.LambdaCallback(
        # on_epoch_begin=on_epoch_begin,
        on_epoch_end=on_epoch_end,
        # on_batch_begin=on_batch_begin,
        # on_batch_end=on_batch_end,
        # on_train_begin=on_train_begin,
        # on_train_end=on_train_end,
    )

    lrs_cb = tf.keras.callbacks.LearningRateScheduler(lr_schedule_300ep)
    csv_logger = tf.keras.callbacks.CSVLogger('data/training.log')
    tb_logger = tf.keras.callbacks.TensorBoard(
        log_dir='logs',
        histogram_freq=0,
        write_graph=False,
        write_images=False,
        update_freq='batch',  # epoch or batch
        profile_batch=2,
        embeddings_freq=0, )

    # callbacks = [lrs_cb, csv_logger, tb_logger]
    callbacks = [lrs_cb, tb_logger, csv_logger, task_callback]
    return callbacks


def main():
    train_db, test_db = load_data()

    resnet_model, class_model = create_models()
    model = Sequential([resnet_model, class_model])
    optimizer, loss, acc = create_params()
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[acc]
    )
    model.build(input_shape=input_shape)
    load_models(resnet_model, class_model)
    # fill_models(class_model)
    # model.build(input_shape=input_shape)
    model.summary()

    callbacks = create_callbacks(resnet_model, class_model)
    model.fit(
        train_db,
        epochs=epochs,
        validation_data=test_db,
        initial_epoch=initial_epoch,
        validation_freq=validation_freq,
        callbacks=callbacks
    )
    model.evaluate(test_db)
    # save_models(resnet_model, class_model)


if __name__ == '__main__':
    main()
