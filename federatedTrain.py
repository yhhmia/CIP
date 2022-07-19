import os
import cv2 as cv
import numpy as np
from tqdm import tqdm
from modelUtil import *
from dataLoader import *
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices("GPU")[0], True)

NUM_CLIENTS = 20
DATA_NAME = "CIFAR"
MODEL = "ResNet50"
EPOCHS = 1000
BATCH_SIZE = 64
WEIGHTS_PATH_TEMPLATE = "weights/Target{}/Client{}/{}_{}_{}_{}.hdf5"
boundaries = [20000, 40000]
values = [0.005, 0.0025, 0.001]

(x_train, y_train), (x_test, y_test), _ = globals()['load_' + DATA_NAME + '_Clients'](NUM_CLIENTS)

train_datasets = [tf.data.Dataset.from_tensor_slices(
    (x, y)).batch(BATCH_SIZE) for x, y in zip(x_train, y_train)]
test_datasets = [tf.data.Dataset.from_tensor_slices(
    (x, y)).batch(BATCH_SIZE) for x, y in zip(x_test, y_test)]

models = [globals()['create_' + MODEL + '_model']((32, 32, 3), 100) for _ in range(NUM_CLIENTS)]

loss_fn = tf.keras.losses.CategoricalCrossentropy()
learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
discriminator_optimizer = tf.keras.optimizers.SGD(learning_rate_fn)

train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
train_acc_metric_agg = tf.keras.metrics.CategoricalAccuracy()
test_acc_metric = tf.keras.metrics.CategoricalAccuracy()
test_acc_metric_agg = tf.keras.metrics.CategoricalAccuracy()
overall_loss = tf.keras.metrics.CategoricalCrossentropy()

@tf.function
def train_step(model, images, labels):
    with tf.GradientTape() as disc_tape:
        outputs = model(images)
        loss = loss_fn(labels, outputs)

    disc_gradients = disc_tape.gradient(loss, model.trainable_variables)
    discriminator_optimizer.apply_gradients(
        zip(disc_gradients, model.trainable_variables))

    train_acc_metric.update_state(labels, outputs)
    overall_loss.update_state(labels, outputs)


def train():
    best_acc = 0
    for epoch in range(EPOCHS):
        for index in range(NUM_CLIENTS):
            with tqdm(enumerate(train_datasets[index])) as tBatch:
                for step, (images, labels) in tBatch:
                    train_step(models[index], images, labels)
                    tBatch.set_description(
                        f"Model: {index} Epoch:{epoch + 1}, Step:{step}, Training Loss:{overall_loss.result()}, Training Acc:{train_acc_metric.result()}")
            train_acc_metric.reset_states()
            overall_loss.reset_states()

        for index in range(NUM_CLIENTS):
            for images, labels in test_datasets[index]:
                y_pred = models[index](images, training=False)
                test_acc_metric.update_state(labels, y_pred)
            with test_summary_writer.as_default():
                tf.summary.scalar(f'model {index} Acc', test_acc_metric.result(), step=epoch + 1)
            tf.print(f"Model: {index} Testing Acc: {test_acc_metric.result()}")
            test_acc_metric.reset_states()

        if (epoch+1) % 1 == 0:
            print(f"Epoch {epoch+1} update weights")
            aggregate_weights(models)

            for index in range(NUM_CLIENTS):
                for images, labels in train_datasets[index]:
                    y_pred = models[index](images, training=False)
                    train_acc_metric_agg.update_state(labels, y_pred)
            tf.print(f"Aggregated Training Acc: {train_acc_metric_agg.result()}")
            train_acc_metric_agg.reset_states()

            for index in range(NUM_CLIENTS):
                for images, labels in test_datasets[index]:
                    y_pred = models[index](images, training=False)
                    test_acc_metric_agg.update_state(labels, y_pred)
            tf.print(f"Aggregated Testing Acc: {test_acc_metric_agg.result()}")
            test_acc_metric_agg.reset_states()
train()