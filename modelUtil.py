from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Activation,Conv2D, MaxPooling2D,Flatten
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, ResNet101, VGG16, VGG19, DenseNet121, ResNet50V2

def create_ResNet50_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        ResNet50(include_top=False,
                 weights='imagenet',
                 input_shape=input_shape),
        GlobalAveragePooling2D(),
        Dense(num_classes),
        Activation("softmax")
    ])
    model.summary()
    return model


def create_ResNet50V2_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        ResNet50V2(include_top=False,
                 weights='imagenet',
                 input_shape=input_shape),
        GlobalAveragePooling2D(),
        Dense(num_classes),
        Activation("softmax")
    ])
    model.summary()
    return model


def create_ResNet101_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        ResNet101(include_top=False,
                 weights='imagenet',
                 input_shape=input_shape),
        GlobalAveragePooling2D(),
        Dense(num_classes),
        Activation("softmax")
    ])
    model.summary()
    return model


def create_VGG16_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        VGG16(include_top=False,
                 weights='imagenet',
                 input_shape=input_shape),
        GlobalAveragePooling2D(),
        Dense(num_classes),
        Activation("softmax")
    ])
    model.summary()
    return model


def create_VGG19_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        VGG19(include_top=False,
                 weights='imagenet',
                 input_shape=input_shape),
        GlobalAveragePooling2D(),
        Dense(num_classes),
        Activation("softmax")
    ])
    model.summary()
    return model


def create_DenseNet121_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        DenseNet121(include_top=False,
                 weights='imagenet',
                 input_shape=input_shape),
        GlobalAveragePooling2D(),
        Dense(num_classes),
        Activation("softmax")
    ])
    model.summary()
    return model


def create_CNN_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes),
        Activation('softmax')
    ])
    model.summary()
    return model


def blend_noise(x_, trigger, alpha):
    noise = tf.squeeze(trigger)
    return tf.map_fn(fn=lambda x : tf.clip_by_value(x*(1-alpha)+noise*alpha, 0, 1), elems=x_), \
           tf.map_fn(fn=lambda x : tf.clip_by_value(x*(1+alpha)+noise*(-alpha), 0, 1), elems=x_)


def gen_random_initial(input_shape):
    noise = tf.random.uniform(input_shape, minval=-1, maxval=1)
    base = tf.Variable(noise[tf.newaxis, :], trainable=True)
    return base
