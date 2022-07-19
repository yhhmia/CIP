from tqdm import tqdm
from PIL import Image
import pandas as pd
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split

def load_CH_MNIST(model_mode):
    """
    Loads CH_MNIST dataset and maps it to Target Model and Shadow Model.
    :param model_mode: one of "TargetModel" and "ShadowModel".
    :return: Tuple of numpy arrays:'(x_train, y_train, l_train), (x_test, y_test, l_test)'.
    :raise: ValueError: in case of invalid `model_mode`.
    """

    if model_mode not in ['Target', 'Shadow']:
        raise ValueError('model_mode must be one of TargetModel, ShadowModel.')

    # Initialize Data
    images, labels = tfds.load('colorectal_histology', split='train', batch_size=-1, as_supervised=True)

    x_train, x_test, y_train, y_test = train_test_split(images.numpy(), labels.numpy(), train_size=0.5,
                                                        random_state=1,
                                                        stratify=labels.numpy())
    if model_mode == "Shadow":
        x_test, x_train, y_test, y_train = train_test_split(images.numpy(), labels.numpy(), train_size=0.5,
                                                            random_state=1,
                                                            stratify=labels.numpy())
    x_train = tf.image.resize((x_train)/255, (64, 64))
    y_train = tf.keras.utils.to_categorical(y_train-1, num_classes=8)
    m_train = tf.ones(y_train.shape[0])

    x_test = tf.image.resize(x_test/255, (64, 64))
    y_test = tf.keras.utils.to_categorical(y_test-1, num_classes=8)
    m_test = tf.zeros(y_test.shape[0])

    member = np.r_[m_train, m_test]
    return (x_train, y_train), (x_test, y_test), member


def load_New_CIFAR(model_mode):
    """
    Loads CIFAR-100 or CIFAR-10 dataset and maps it to Target Model and Shadow Model.
    :param model_mode: one of "TargetModel" and "ShadowModel".
    :param num_classes: one of 10 and 100 and the default value is 100
    :return: Tuple of numpy arrays:'(x_train, y_train, l_train), (x_test, y_test, l_test)'.
    :raise: ValueError: in case of invalid `model_mode`.
    """
    if model_mode not in ['Target', 'Shadow']:
        raise ValueError('model_mode must be one of TargetModel, ShadowModel.')

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
    images, labels = x_train[:20000], y_train[:20000]
    #
    # x_train, x_test, y_train, y_test = train_test_split(images, labels, train_size=0.5,
    #                                                     random_state=1,
    #                                                     stratify=labels)
    x_train, x_test, y_train, y_test = train_test_split(images, labels, train_size=0.5,
                                                        random_state=1,
                                                        stratify=labels)

    if model_mode == "Shadow":
        (x_train, y_train), (x_test, y_test) = (x_test, y_test), (x_train, y_train)

    x_train = tf.cast(x_train/255, dtype=tf.float32)
    x_train = tf.image.resize(x_train, [80, 80])
    x_train = tf.map_fn(lambda im: tf.image.random_crop(im, [64, 64, 3]), x_train)
    x_train = tf.image.flip_left_right(x_train)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=100)
    m_train = np.ones(y_train.shape[0])



    x_test = tf.cast(x_test/255, dtype=tf.float32)
    x_test = tf.image.resize(x_test, [64, 64])
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=100)
    m_test = np.zeros(y_test.shape[0])

    member = np.r_[m_train, m_test]
    return (x_train, y_train), (x_test, y_test), member


def load_CIFAR(model_mode):
    """
    Loads CIFAR-100 or CIFAR-10 dataset and maps it to Target Model and Shadow Model.
    :param model_mode: one of "TargetModel" and "ShadowModel".
    :param num_classes: one of 10 and 100 and the default value is 100
    :return: Tuple of numpy arrays:'(x_train, y_train, l_train), (x_test, y_test, l_test)'.
    :raise: ValueError: in case of invalid `model_mode`.
    """
    if model_mode not in ['Target', 'Shadow']:
        raise ValueError('model_mode must be one of TargetModel, ShadowModel.')

    (x_train, y_train), (_, _) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
    images, labels = x_train[:20000], y_train[:20000]

    x_train, x_test, y_train, y_test = train_test_split(images, labels, train_size=0.5,
                                                        random_state=1,
                                                        stratify=labels)

    if model_mode == "Shadow":
        (x_train, y_train), (x_test, y_test) = (x_test, y_test), (x_train, y_train)

    x_train = tf.cast(x_train / 255, dtype=tf.float32)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=100)
    m_train = np.ones(y_train.shape[0])

    x_test = tf.cast(x_test / 255, dtype=tf.float32)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=100)
    m_test = np.zeros(y_test.shape[0])

    member = np.r_[m_train, m_test]
    return (x_train, y_train), (x_test, y_test), member


def load_Ratio_CIFAR(model_mode, ratio):
    """
    Loads CIFAR-100 or CIFAR-10 dataset and maps it to training set or testing set of Shadow Model.
    :param model_mode: one of "TargetModel" and "ShadowModel".
    :param num_classes: one of 10 and 100 and the default value is 100
    :return: Tuple of numpy arrays:'(x_train, y_train, (x_test, y_test), member'.
    :raise: ValueError: in case of invalid `model_mode`.
    """
    if model_mode not in ['Target', 'Shadow']:
        raise ValueError('model_mode must be one of TargetModel, ShadowModel.')
    (x_train, y_train), (_, _) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
    images, labels = x_train[:20000], y_train[:20000]
    x_1, x_2, y_1, y_2 = train_test_split(images, labels, train_size=0.5,
                                                        random_state=1, stratify=labels)
    x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(x_1, y_1, train_size=ratio,
                                                                    random_state=1, stratify=y_1)
    x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(x_2, y_2, train_size=1-ratio,
                                                                    random_state=1, stratify=y_2)

    x_train, y_train= tf.concat([x_train_1, x_train_2], 0), tf.concat([y_train_1, y_train_2], 0)
    x_test, y_test = tf.concat([x_test_1, x_test_2], 0), tf.concat([y_test_1, y_test_2], 0)
    x_train = tf.cast(x_train / 255, dtype=tf.float32)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=100)
    m_train = np.ones(y_train.shape[0])

    x_test = tf.cast(x_test / 255, dtype=tf.float32)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=100)
    m_test = np.zeros(y_test.shape[0])

    member = np.r_[m_train, m_test]
    return (x_train, y_train), (x_test, y_test), member


def load_Ratio_New_CIFAR(model_mode, ratio):
    """
    Loads CIFAR-100 or CIFAR-10 dataset and maps it to training set or testing set of Shadow Model.
    :param model_mode: one of "TargetModel" and "ShadowModel".
    :param num_classes: one of 10 and 100 and the default value is 100
    :return: Tuple of numpy arrays:'(x_train, y_train, (x_test, y_test), member'.
    :raise: ValueError: in case of invalid `model_mode`.
    """
    if model_mode not in ['Target', 'Shadow']:
        raise ValueError('model_mode must be one of TargetModel, ShadowModel.')
    (x_train, y_train), (_, _) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
    images, labels = x_train[:20000], y_train[:20000]
    x_1, x_2, y_1, y_2 = train_test_split(images, labels, train_size=0.5,
                                                        random_state=1, stratify=labels)
    x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(x_1, y_1, train_size=ratio,
                                                                    random_state=1, stratify=y_1)
    x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(x_2, y_2, train_size=1-ratio,
                                                                    random_state=1, stratify=y_2)

    x_train, y_train= tf.concat([x_train_1, x_train_2], 0), tf.concat([y_train_1, y_train_2], 0)
    x_test, y_test = tf.concat([x_test_1, x_test_2], 0), tf.concat([y_test_1, y_test_2], 0)

    x_train = tf.cast(x_train / 255, dtype=tf.float32)
    x_train = tf.image.resize(x_train, [80, 80])
    x_train = tf.map_fn(lambda im: tf.image.random_crop(im, [64, 64, 3]), x_train)
    # x_train = tf.keras.applications.imagenet_utils.preprocess_input(x_train)
    x_train = tf.image.flip_left_right(x_train)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=100)
    m_train = np.ones(y_train.shape[0])

    x_test = tf.cast(x_test / 255, dtype=tf.float32)
    x_test = tf.image.resize(x_test, [64, 64])
    # x_test = tf.keras.applications.imagenet_utils.preprocess_input(x_test)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=100)
    m_test = np.zeros(y_test.shape[0])

    member = np.r_[m_train, m_test]
    return (x_train, y_train), (x_test, y_test), member


def load_Ratio_CH_MNIST(model_mode, ratio):
    """
    Loads CH_MNIST dataset and maps it to Target Model and Shadow Model.
    :param model_mode: one of "TargetModel" and "ShadowModel".
    :return: Tuple of numpy arrays:'(x_train, y_train, l_train), (x_test, y_test, l_test)'.
    :raise: ValueError: in case of invalid `model_mode`.
    """

    if model_mode not in ['Target', 'Shadow']:
        raise ValueError('model_mode must be one of TargetModel, ShadowModel.')

    images, labels = tfds.load('colorectal_histology', split='train', batch_size=-1, as_supervised=True)

    x_1, x_2, y_1, y_2 = train_test_split(images.numpy(), labels.numpy(), train_size=0.5,
                                                        random_state=1,
                                                        stratify=labels.numpy())
    x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(x_1, y_1, train_size=ratio,
                                                                random_state=1, stratify=y_1)
    x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(x_2, y_2, train_size=1 - ratio,
                                                                random_state=1, stratify=y_2)

    x_train, y_train = tf.concat([x_train_1, x_train_2], 0), tf.concat([y_train_1, y_train_2], 0)
    x_test, y_test = tf.concat([x_test_1, x_test_2], 0), tf.concat([y_test_1, y_test_2], 0)

    x_train = tf.image.resize((x_train)/255, (64, 64))
    #x_train = tf.keras.applications.imagenet_utils.preprocess_input(x_train)
    y_train = tf.keras.utils.to_categorical(y_train-1, num_classes=8)
    m_train = tf.ones(y_train.shape[0])

    x_test = tf.image.resize(x_test/255, (64, 64))
    #x_test = tf.keras.applications.imagenet_utils.preprocess_input(x_test)
    y_test = tf.keras.utils.to_categorical(y_test-1, num_classes=8)
    m_test = tf.zeros(y_test.shape[0])

    member = np.r_[m_train, m_test]
    return (x_train, y_train), (x_test, y_test), member


def load_CIFAR_Clients(num_clients):
    """
    Loads CIFAR-100 or CIFAR-10 dataset and maps it to Target Model and Shadow Model.
    :param model_mode: one of "TargetModel" and "ShadowModel".
    :param num_classes: one of 10 and 100 and the default value is 100
    :return: Tuple of numpy arrays:'(x_train, y_train, l_train), (x_test, y_test, l_test)'.
    :raise: ValueError: in case of invalid `model_mode`.
    """
    data = tfds.load("cifar100", split=["train", "test"], batch_size=-1)
    x_train = [tf.keras.applications.resnet_v2.preprocess_input(tf.cast(data[0]['image'][data[0]['coarse_label']==i], dtype=tf.float32)) for i in range(num_clients)]
    y_train = [tf.one_hot(data[0]['label'][data[0]['coarse_label']==i], depth=100) for i in range(num_clients)]

    x_test = [tf.keras.applications.resnet_v2.preprocess_input(tf.cast(data[1]['image'][data[1]['coarse_label']==i], dtype=tf.float32)) for i in range(num_clients)]
    y_test = [tf.one_hot(data[1]['label'][data[1]['coarse_label']==i], depth=100) for i in range(num_clients)]

    m_train = np.ones(sum([i.shape[0] for i in y_train]))
    m_test = np.zeros(sum([i.shape[0] for i in y_test]))

    member = np.r_[m_train, m_test]

    return (x_train, y_train), (x_test, y_test), member


def load_CIFAR_ACTIVE_ATTACK(num_clients, ratio=1):
    """
    Loads CIFAR-100 or CIFAR-10 dataset and maps it to Target Model and Shadow Model.
    :param model_mode: one of "TargetModel" and "ShadowModel".
    :param num_classes: one of 10 and 100 and the default value is 100
    :return: Tuple of numpy arrays:'(x_train, y_train, l_train), (x_test, y_test, l_test)'.
    :raise: ValueError: in case of invalid `model_mode`.
    """
    data = tfds.load("cifar100", split=["train", "test"], batch_size=-1)
    x_train = tf.keras.applications.resnet_v2.preprocess_input(tf.cast(data[0]['image'][data[0]['coarse_label'] < num_clients][:500*ratio*num_clients], dtype=tf.float32))
    y_train = tf.one_hot(data[0]['label'][data[0]['coarse_label'] < num_clients][:500*ratio*num_clients], depth=100)

    x_test = tf.keras.applications.resnet_v2.preprocess_input(tf.cast(data[1]['image'][data[1]['coarse_label'] < num_clients][:500*ratio*num_clients], dtype=tf.float32))
    y_test = tf.one_hot(data[1]['label'][data[1]['coarse_label'] < num_clients][:500*ratio*num_clients], depth=100)

    m_train = np.ones(y_train.shape[0])
    m_test = np.zeros(y_test.shape[0])

    member = np.r_[m_train, m_test]
    return (x_train, y_train), (x_test, y_test), member