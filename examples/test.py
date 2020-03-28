import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from tensorflow import keras
import os
import numpy as np

import sys


def main():
    print("tensorflow version:", tf.__version__)
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")

    if tf.test.is_built_with_cuda():
        print('Yes!!! is_built_with_cuda')
    else:
        print('No!!! is_built_with_cuda')


if __name__ == '__main__':
    main()
